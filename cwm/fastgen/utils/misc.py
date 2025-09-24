# Copyright (c) Meta Platforms, Inc. and affiliates.

import struct

import numpy as np
import xxhash


def hashints(vals: list[int]) -> int:
    """
    Compute a hash of an int list and return it as
    a non-negative integer less than ``2**32``.
    """
    vala = np.array(vals, dtype=np.uint32)
    h = xxhash.xxh32_digest(vala.view(np.uint8))  # type: ignore
    return struct.unpack("<I", h)[0] & 0x7FFFFFFF
