# Copyright (c) Meta Platforms, Inc. and affiliates.
# ruff: noqa


import torch
import triton
import triton.language as tl

# PagedAttention helper kernel to fill paged kv-caches
#
# Instead of using plain contiguous tensors, PagedAttention
# structures the caches as sequences of pages with a layout
# described by a page (or block) table.
#
# The kernel paged_memcpy defined in this file is a primitive
# that lets us fill paged caches efficiently and in a CUDA-
# graphable way.


@triton.jit
def paged_memcpy_kernel(
    src_ptr,
    dst_ptr,
    page_ptr,  # int32 data
    dst_pos_ptr,  # int32 data
    src_batch_ptr_opt,  # int32 data
    batch_size_ptr_opt,  # int32 data
    BLOCK_SIZE: tl.constexpr,
    ELEM_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    SHARD_INDEX: tl.constexpr,
    SHARD_COUNT: tl.constexpr,
):
    """
    Copy memory from ``src_ptr`` to ``dst_ptr``. The logical
    location within ``dst_ptr`` to write to is identified by
    ``dst_pos_ptr`` and the program id. The physical memory
    location is given by indirection through the mapping
    defined by a row of ``page_ptr``. The row to consider
    for each source element is given by ``src_batch_ptr_opt``.
    In case the latter pointer is not provided, it is assumed
    to point to ``arange(src.shape[0])``.

    Constant arguments:
      - ``BLOCK_SIZE``: how many scalars to load/store at once;
      - ``ELEM_DIM``: dimension of elements to copy, and stride of
        the ``{src,dst}_ptr`` tensors;
      - ``PAGE_SIZE``: number of elements per page;
      - ``PAGE_STRIDE``: stride of the ``page_ptr`` tensor.
      - ``DST_STRIDE``: stride of the ``dst_ptr`` tensor.
      - ``SHARD_INDEX``: shard index of the destination tensor.
      - ``SHARD_COUNT``: shard count of the destination tensor.
    """
    elem_idx = tl.program_id(0).to(tl.int64)
    if batch_size_ptr_opt is not None:
        tl.static_assert(src_batch_ptr_opt is None)
        batch_size = tl.load(batch_size_ptr_opt)
        if elem_idx >= batch_size:
            return

    dst_pos = tl.load(dst_pos_ptr + elem_idx)
    if SHARD_COUNT > 1:
        if dst_pos % SHARD_COUNT != SHARD_INDEX:
            return
        dst_pos //= SHARD_COUNT

    src_ptr += elem_idx * ELEM_DIM

    if src_batch_ptr_opt is None:
        page_ptr += elem_idx * PAGE_STRIDE
    else:
        batch_id = tl.load(src_batch_ptr_opt + elem_idx)
        page_ptr += batch_id * PAGE_STRIDE

    page_idx = tl.load(page_ptr + dst_pos // PAGE_SIZE)
    dst_ofs = dst_pos % PAGE_SIZE

    # now set to the location we need to copy our element to
    dst_ptr += (page_idx * PAGE_SIZE + dst_ofs) * DST_STRIDE

    for offset in range(0, ELEM_DIM, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < ELEM_DIM
        data = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, data, mask=mask)


def paged_memcpy(
    src: torch.Tensor,
    dst: torch.Tensor,
    page_tbl: torch.Tensor,
    dst_pos: torch.Tensor,
    dst_shard: tuple[int, int] | None,
    src_batch: torch.Tensor | None,
    batch_size: torch.Tensor | None,
    page_size: int,
) -> None:
    """
    Copy memory to a paged target. The structure of
    destination tensor is given by a page table that
    maps logical positions in the tensor to physical
    positions. This is the same mapping scheme as the
    one used in paged-attention.

    Args:
        src (torch.Tensor):
            the source tensor of shape (N, D) where
            N is the number of elements to copy and
            D is the size of one element.
        dst (torch.Tensor):
            the destination tensor, of shape (?, D).
        page_tbl (torch.Tensor):
            an integer tensor of shape (B, P) where
            B is a batch dimension and P is the number
            of pages available; the element with
            logical index ``n`` in the ``dst`` tensor
            will be found at physical index
            ``page_tbl[n // page_size] * page_size
              + n % page_size``.
        dst_pos (torch.Tensor):
            an integer tensor of shape (N,) giving
            the destination logical index of each
            source element.
        dst_shard (tuple[int, int] | None):
            if specified as (K, S), only the items at
            positions p = K (mod S) will be copied to
            the destination tensor.
        src_batch (torch.Tensor, optional):
            an integer tensor of shape (N,) giving
            for each source element the row of the
            ``page_tbl`` tensor to use; if not
            provided, it defaults to``arange(N)``.
        batch_size (torch.Tensor, optional):
            when provided, this scalar tensor will
            limit the processing to the first
            ``batch_size`` elements of ``src``;
            it is useful in graphed contexts.
        page_size (int): number of elements in a page.
    """

    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError(
            "src and dst tensors must be 2-D tensors; "
            f"got this instead: {src.ndim=} {dst.ndim=}"
        )
    if src.shape[1] != dst.shape[1]:
        raise ValueError(
            "src and dst must have identical dimension 1",
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            "src and dst must have identical dtype; "
            f"got this instead: {src.dtype=} {dst.dtype=}"
        )
    if not src.is_contiguous():
        raise ValueError("src must be contiguous")
    if dst.stride(1) != 1:
        raise ValueError(
            "dst must be contiguous on its last dimension",
        )

    N = src.shape[0]

    if page_tbl.ndim != 2:
        raise ValueError(
            "page_tbl must be a 2-D integer tensor; "
            f"got this instead: {page_tbl.shape=}"
        )
    if page_tbl.dtype != torch.int:
        raise ValueError("page_tbl must be an int tensor")

    if dst_pos.ndim != 1 or dst_pos.shape[0] != N:
        raise ValueError(
            f"dst_pos must be a 1-D tensor with size {N}; "
            f"got this instead: {dst_pos.shape=}"
        )
    if dst_pos.dtype != torch.int:
        raise ValueError(
            "dst_pos must be a 1-D integer tensor; "
            f"got this instead: {dst_pos.dtype=}"
        )
    if not dst_pos.is_contiguous():
        raise ValueError("dst_pos must be contiguous")

    if src_batch is not None:
        if src_batch.ndim != 1 or src_batch.shape[0] != N:
            raise ValueError(
                f"src_batch must be a 1-D tensor with size {N}; "
                f"got this instead: {src_batch.shape=}"
            )
        if src_batch.dtype != torch.int:
            raise ValueError(
                "src_batch must be an integer tensor; "
                f"got this instead: {src_batch.dtype=}",
            )
        if not src_batch.is_contiguous():
            raise ValueError("src_batch must be contiguous")

    if batch_size is not None:
        if batch_size.ndim != 0:
            raise ValueError(
                f"batch_size must be a scalar tensor; "
                f"got this instead: {batch_size.shape=}"
            )
        if batch_size.dtype not in (torch.int, torch.int64):
            raise ValueError(
                f"batch_size must be an integer tensor; "
                f"got this instead: {batch_size.dtype=}"
            )

    BLOCK_SIZE = 1024
    paged_memcpy_kernel[(N,)](
        src,
        dst,
        page_tbl,
        dst_pos,
        src_batch,
        batch_size,
        BLOCK_SIZE,
        src.shape[1],
        page_size,
        page_tbl.stride(0),
        dst.stride(0),
        dst_shard[0] if dst_shard else 0,
        dst_shard[1] if dst_shard else 1,
    )
