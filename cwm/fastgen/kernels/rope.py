# Copyright (c) Meta Platforms, Inc. and affiliates.
# ruff: noqa

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    pos_ptr,
    freqs_ptr,
    n_heads,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM2: tl.constexpr,
    HEADS_PER_ELEM: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    APPROX_TRIGO: tl.constexpr,
):
    """
    Apply RoPE to the ``x_ptr`` tensor. The frequencies
    to use per element are provided in the ``freqs_ptr``
    tensor, this way the kernel does not have to know
    about the various scaling schemes and RoPE params.
    The number of heads total in ``x_ptr`` is given by
    ``n_heads``.

    The RoPE computations are performed in float32.

    Constant arguments:
      - ``HEADS_PER_ELEM``: each input element in the
        ``x_ptr`` tensor is composed of this many heads;
      - ``HEAD_DIM``: the dimension of a head, and also
        double the length of the ``freqs_ptr`` tensor
        (must be even);
      - ``HEAD_DIM2``: ``next_power_of_2(HEAD_DIM) // 2``;
      - ``HEAD_GROUP_SIZE``: loop iteration size in heads;
      - ``HEADS_PER_BLOCK``: each kernel block will process
        this many heads;
      - ``APPROX_TRIGO``: whether to use fast approximate
        sin/cos functions (only safe for inputs in
        [-100pi, +100pi]).
    """
    block_idx = tl.program_id(0).to(tl.int64)
    freqs_ofs = tl.arange(0, HEAD_DIM2)
    head_mask = 2 * freqs_ofs < HEAD_DIM
    freqs = tl.load(freqs_ptr + freqs_ofs, mask=head_mask)

    # iterate on head groups of size HEAD_GROUP_SIZE
    for head_idx in range(
        block_idx * HEADS_PER_BLOCK,
        (block_idx + 1) * HEADS_PER_BLOCK,
        HEAD_GROUP_SIZE,
    ):
        grp_ofs = head_idx + tl.arange(0, HEAD_GROUP_SIZE)
        head_pos = tl.load(pos_ptr + grp_ofs // HEADS_PER_ELEM)
        angles = head_pos[:, None] * freqs[None, :]
        tl.static_assert(angles.dtype == tl.float32)

        if APPROX_TRIGO:
            sines, cosines = tl.inline_asm_elementwise(
                asm="""
                sin.approx.f32  $0, $2;
                cos.approx.f32  $1, $2;
                """,
                constraints="=r,=r,r",
                args=[angles],
                dtype=(tl.float32, tl.float32),
                is_pure=True,
                pack=1,
            )
        else:
            sines = tl.sin(angles)
            cosines = tl.cos(angles)

        re_ofs = grp_ofs[:, None] * HEAD_DIM + 2 * freqs_ofs[None, :]
        im_ofs = re_ofs + 1

        mask = (grp_ofs < n_heads)[:, None] & head_mask[None, :]

        re_x = tl.load(x_ptr + re_ofs, mask=mask).to(tl.float32)
        im_x = tl.load(x_ptr + im_ofs, mask=mask).to(tl.float32)

        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines

        tl.store(x_ptr + re_ofs, re_out, mask=mask)
        tl.store(x_ptr + im_ofs, im_out, mask=mask)


def apply_rope(
    x: torch.Tensor,
    pos: torch.Tensor,
    freqs: torch.Tensor,
    approx_trigo: bool = False,
) -> None:
    """
    Apply RoPE in place to the argument tensor.

    Args:
        x (torch.Tensor):
            the tensor to apply RoPE to, with shape
            (N, H, D) where H is the number of heads
            per element and D is the dimension of
            the individual heads modified by RoPE.
        pos (torch.Tensor):
            an integer tensor of shape (N,) giving
            the sequence position of each element
            in ``x``.
        freqs (torch.Tensor):
            the frequencies to use in the RoPE
            computation; this must be a float
            tensor of shape (D/2,).
    """
    if x.ndim != 3:
        raise ValueError(
            f"x must be a 3-D tensor; got: {x.ndim=}",
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    N, H, D = x.shape

    if pos.ndim != 1 or pos.shape[0] != N:
        raise ValueError(
            f"pos must be a 1-D tensor with size {N}; "
            f"got this instead : {pos.shape=}",
        )
    if not pos.is_contiguous():
        raise ValueError("pos must be contiguous")

    if freqs.ndim != 1 or freqs.shape[0] * 2 != D:
        raise ValueError(
            f"freqs must be a 1-D tensor with size {D/2}; "
            f"got this instead: {freqs.shape=}"
        )
    if not freqs.is_contiguous():
        raise ValueError("freqs must be contiguous")

    n_heads = N * H
    if n_heads < 2048:
        HEADS_PER_BLOCK = 1
        HEAD_GROUP_SIZE = 1
    else:
        head_size = D * x.element_size()
        HEADS_PER_BLOCK = max(1, 2048 // head_size)
        HEAD_GROUP_SIZE = triton.next_power_of_2(512 // head_size)

    n_blocks = triton.cdiv(n_heads, HEADS_PER_BLOCK)
    rope_kernel[(n_blocks,)](
        x,
        pos,
        freqs,
        n_heads,
        D,
        triton.next_power_of_2(D) // 2,
        H,
        HEAD_GROUP_SIZE,
        HEADS_PER_BLOCK,
        approx_trigo,
    )
