# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import TYPE_CHECKING, TypeAlias

import torch
import torch.nn

if TYPE_CHECKING:
    from cwm.model.transformer import Transformer as CWMModel

    # the code below works for multiple model classes
    TransformerModel: TypeAlias = CWMModel


def consecutive(*tensors: torch.Tensor) -> torch.Tensor | None:
    """
    Check whether the input tensors are consecutive in memory.

    If the input tensors are consecutive and part of a single
    storage, return a tensor that is the concatenation of all the
    input tensors on dimension 0.

    Returns:
        The concatenation of the input tensors, or None if the
        input tensors are not consecutive in memory.
    """
    if not tensors:
        return torch.Tensor()

    if not all(t.is_contiguous() for t in tensors):
        return None

    t = tensors[0]
    storage = t.untyped_storage()
    dtype = t.dtype
    sptr = storage.data_ptr()
    soff = t.storage_offset()
    ptr = t.data_ptr()
    stride = t.stride()
    d0, ds = t.shape[0], t.shape[1:]

    # from PyTorch 2.1 we can use t.nbytes() instead
    ptr += t.numel() * t.element_size()

    for t in tensors[1:]:
        if (
            t.untyped_storage().data_ptr() != sptr
            or t.dtype != dtype
            or t.data_ptr() != ptr
            or t.stride() != stride
            or t.shape[1:] != ds
        ):
            return None
        d0 += t.shape[0]
        ptr += t.numel() * t.element_size()

    cat = torch.empty(0, dtype=dtype, device=storage.device)
    return cat.set_(storage, soff, (d0, *ds), stride)
