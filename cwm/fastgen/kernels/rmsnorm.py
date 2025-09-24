# Copyright (c) Meta Platforms, Inc. and affiliates.
# ruff: noqa

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    eps,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute the root mean square normalization of the
    ``x_ptr`` tensor and stores it in ``out_ptr``.
    The two tensors must be of identical shape (N, DIM)
    and the weight tensor ``w_ptr`` must be of shape
    (DIM,).

    Constant arguments:
      - ``DIM``: dimension of the elements.
      - ``BLOCK_SIZE``: how many elements to process
        at once in a block.
    """
    idx = tl.program_id(0).to(tl.int64)
    x_ptr += idx * DIM
    out_ptr += idx * DIM

    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, DIM, BLOCK_SIZE):
        ofs = offset + tl.arange(0, BLOCK_SIZE)
        # evict_last because we will reload the same
        # data in the next loop
        a = tl.load(
            x_ptr + ofs,
            mask=ofs < DIM,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        mean += a * a

    rstd = tl.rsqrt((tl.sum(mean, axis=0) / DIM) + eps)

    for offset in range(0, DIM, BLOCK_SIZE):
        ofs = offset + tl.arange(0, BLOCK_SIZE)
        mask = ofs < DIM
        # evict_first because it is the last time we
        # read this data
        a = tl.load(
            x_ptr + ofs,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + ofs, mask=mask)
        tl.store(out_ptr + ofs, a * rstd * w, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute the root mean square normalization of the
    input tensor.

    Args:
        x (torch.Tensor):
            the tensor to normalize, with shape (N, D)
            where N is a batch dimension and D is the
            dimension of the elements to normalize.
        w (torch.Tensor):
            multiplicative weight for the output.
        eps (float):
            a small value used in the denominator of
            the normalization division for numerical
            stability.

    Returns:
        torch.Tensor: the normalized tensor.
    """
    if x.ndim != 2:
        raise ValueError(
            "x must be a 2-D tensor; got {x.ndim=}",
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    N, D = x.shape

    if tuple(w.shape) != (D,):
        raise ValueError(
            f"w must have shape ({D},); got: {w.shape=}",
        )
    if not w.is_contiguous():
        raise ValueError("w must be contiguous")

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    BLOCK_SIZE = min(8192, max(BLOCK_SIZE, 128))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    out = torch.empty_like(x)
    rmsnorm_kernel[(N,)](
        x,
        out,
        w,
        eps,
        D,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out
