# Copyright (c) Meta Platforms, Inc. and affiliates.

# Patricia/Radix tree implementation of ordered int sets
# Ref: "Fast Mergeable Integer Maps", Okasaki and Gill
# Ref: https://en.wikipedia.org/wiki/Radix_tree

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

_L = 32
_M = 2**_L - 1


class SetNode:
    def __contains__(self, elm: int) -> bool:
        if elm > _M:
            return False
        return self._contains(elm)

    def __len__(self) -> int:
        return self._len(_M)

    def __iter__(self) -> Iterable[int]:
        return iter(self.tolist())

    def add(self, elm: int) -> "SetNode":
        assert elm <= _M, f"{elm} out of bounds (>{_M})"
        return self._union(singleton(elm))

    def union(self, other: "SetNode") -> "SetNode":
        return self._union(other)

    def take(self, count: int) -> tuple["SetNode", "SetNode"]:
        return self._take(count, _M)

    def tolist(self) -> list[int]:
        items: list[int] = []
        self._tolist(items, 0, _M)
        return items

    @abstractmethod
    def _len(self, mask: int) -> int: ...

    @abstractmethod
    def _contains(self, elm: int) -> bool: ...

    @abstractmethod
    def _union(self, other: "SetNode") -> "SetNode": ...

    @abstractmethod
    def _take(self, count: int, mask: int) -> tuple["SetNode", "SetNode"]: ...

    @abstractmethod
    def _tolist(self, acc: list[int], prefix: int, mask: int) -> None: ...


def singleton(elm: int) -> "SetNode":
    assert elm <= _M, f"{elm} out of bounds (>{_M})"
    return Chk._mk(_M, elm, Mem.t())


def interval(a: int, b: int, mask: int = _M) -> SetNode:
    "Make an interval set; ``a`` inclusive, ``b`` exclusive."
    assert a >= 0, f"{a=} must not be negative"
    assert b <= mask + 1, f"{b=} out of bounds (>{mask + 1})"
    if b <= a:
        return _empty
    # largest power of 2 that is <= b-a
    p = 1 << ((b - a).bit_length() - 1)
    aa = (a + p - 1) & -p  # aligned a
    if not (aa + p <= b):
        p >>= 1
        aa = (a + p - 1) & -p
        assert p and aa + p <= b
    r0 = interval(a, aa, mask)
    r1 = Chk._mk(mask & -p, aa, Mem.t())
    r2 = interval(aa + p, b, mask)
    return r0.union(r1).union(r2)


@dataclass(frozen=True)
class Mem(SetNode):
    mem: bool

    def __repr__(self) -> str:
        return "T" if self.mem else "F"

    @cache
    @staticmethod
    def t() -> SetNode:
        return Mem(True)

    @cache
    @staticmethod
    def f() -> SetNode:
        return Mem(False)

    def _len(self, mask: int) -> int:
        assert mask & (mask + 1) == 0, bin(mask)
        if self.mem:
            return mask + 1
        return 0

    def _contains(self, elm: int) -> bool:
        return self.mem

    def _union(self, other: SetNode) -> SetNode:
        return self if self.mem else other

    def _take(self, count: int, mask: int) -> tuple[SetNode, SetNode]:
        assert mask & (mask + 1) == 0, bin(mask)
        if not self.mem:
            return _empty, _empty
        if count >= mask + 1:
            return self, _empty
        a = interval(0, count, mask)
        b = interval(count, mask + 1, mask)
        return a, b

    def _tolist(self, acc: list[int], prefix: int, mask: int) -> None:
        assert mask & (mask + 1) == 0, bin(mask)
        if self.mem:
            acc.extend(range(prefix, prefix + mask + 1))


_empty = Mem.f()


@dataclass(frozen=True)
class Chk(SetNode):
    mask: int
    bits: int
    chld: SetNode

    def __repr__(self) -> str:
        return f"Chk({bin(self.mask)[2:]}, {self.bits}, {self.chld})"

    @staticmethod
    def _mk(mask: int, bits: int, chld: SetNode) -> SetNode:
        if not mask or chld is _empty:
            return chld
        bits &= mask
        if isinstance(chld, Chk):
            assert (mask & chld.mask) == 0
            mask |= chld.mask
            bits |= chld.bits
            chld = chld.chld
        return Chk(mask, bits, chld)

    def _len(self, mask: int) -> int:
        return self.chld._len(mask & ~self.mask)

    def _contains(self, elm: int) -> bool:
        if (elm & self.mask) == self.bits:
            return self.chld._contains(elm)
        return False

    def _union(self, other: SetNode) -> SetNode:
        if not isinstance(other, Chk):
            return other._union(self)

        if other.mask < self.mask:
            return other._union(self)
        assert self.mask & other.mask == self.mask
        assert other.mask < 2 * self.mask, (self, other)

        mask = self.mask
        diff = (self.bits ^ other.bits) & mask

        # prefix of identical bits
        ones = mask
        while diff & ones:
            ones <<= 1

        mask0 = ones & mask
        omask1 = other.mask & ~mask0
        smask1 = self.mask & ~mask0

        if mask0 == 0:
            # differ on the first bit; branch on it
            high = (ones >> 1) & mask
            c0 = Chk._mk(omask1 & ~high, other.bits, other.chld)
            c1 = Chk._mk(smask1 & ~high, self.bits, self.chld)
            chlds = (c0, c1) if self.bits & high else (c1, c0)
            return Bit._mk(high, chlds)

        so = Chk._mk(omask1, other.bits, other.chld)
        ss = Chk._mk(smask1, self.bits, self.chld)
        # call into the other branch
        chld = so._union(ss)
        return Chk._mk(mask0, self.bits, chld)

    def _take(self, count: int, mask: int) -> tuple[SetNode, SetNode]:
        if count >= self._len(mask):
            return self, _empty
        a, b = self.chld._take(count, mask & ~self.mask)
        a = Chk._mk(self.mask, self.bits, a)
        b = Chk._mk(self.mask, self.bits, b)
        return a, b

    def _tolist(self, acc: list[int], prefix: int, mask: int) -> None:
        assert mask and (mask & (mask + 1) == 0)
        self.chld._tolist(acc, prefix | self.bits, mask & ~self.mask)


@dataclass(frozen=True)
class Bit(SetNode):
    bit: int
    chld: tuple[SetNode, SetNode]

    def __repr__(self) -> str:
        return f"Bit({bin(self.bit)[2:]}, {self.chld})"

    @staticmethod
    def _mk(bit: int, chld: tuple[SetNode, SetNode]) -> SetNode:
        assert bit.bit_count() == 1
        c0, c1 = chld
        if c0 is c1:
            return c0
        if c0 is _empty:
            return Chk._mk(bit, bit, c1)
        if c1 is _empty:
            return Chk._mk(bit, 0, c0)
        return Bit(bit, chld)

    def _len(self, mask: int) -> int:
        mask &= ~self.bit
        c0, c1 = self.chld
        return c0._len(mask) + c1._len(mask)

    def _contains(self, elm: int) -> bool:
        bit = int((elm & self.bit) != 0)
        return self.chld[bit]._contains(elm)

    def _union(self, other: SetNode) -> SetNode:
        match other:
            case Bit(bit, (c0, c1)):
                assert self.bit == bit
                c0 = c0._union(self.chld[0])
                c1 = c1._union(self.chld[1])
                return Bit._mk(bit, (c0, c1))
            case Chk(mask, bits, chld):
                bit = self.bit
                assert mask < bit * 2 and mask & bit
                s = Chk._mk(mask & ~bit, bits, chld)
                c0, c1 = self.chld
                if bits & bit:
                    c1 = c1._union(s)
                else:
                    c0 = c0._union(s)
                return Bit._mk(bit, (c0, c1))
            case Mem():
                return other._union(self)
        raise RuntimeError(f"unhandled {other}")

    def _take(self, count: int, mask: int) -> tuple[SetNode, SetNode]:
        if count >= self._len(mask):
            return self, _empty
        mask &= ~self.bit
        c0, c1 = self.chld
        a0, b0 = c0._take(count, mask)
        count -= a0._len(mask)
        if count > 0:
            a1, b1 = c1._take(count, mask)
            a = Bit._mk(self.bit, (a0, a1))
            assert b0 is _empty
            # b = Bit._mk(self.bit, (b0, b1))
            b = Chk._mk(self.bit, self.bit, b1)
        else:
            a = Chk._mk(self.bit, 0, a0)
            b = Bit._mk(self.bit, (b0, c1))
        return a, b

    def _tolist(self, acc: list[int], prefix: int, mask: int) -> None:
        assert mask and (mask & (mask + 1) == 0)
        self.chld[0]._tolist(acc, prefix, mask & ~self.bit)
        self.chld[1]._tolist(acc, prefix | self.bit, mask & ~self.bit)


@dataclass
class ISet:
    node: SetNode = _empty

    @staticmethod
    def interval(a: int, b: int) -> "ISet":
        return ISet(interval(a, b))

    @staticmethod
    def singleton(elm: int) -> "ISet":
        return ISet(singleton(elm))

    def __iter__(self) -> Iterable[int]:
        return iter(self.node.tolist())

    def __len__(self) -> int:
        return len(self.node)

    def tolist(self) -> list[int]:
        return self.node.tolist()

    def add(self, elm: int) -> None:
        self.node = self.node.add(elm)

    def merge(self, other: "ISet") -> None:
        self.node = self.node.union(other.node)

    def take(self, count: int) -> tuple["ISet", "ISet"]:
        a, b = self.node.take(count)
        return ISet(a), ISet(b)

    def popleft(self) -> int:
        elm, self.node = self.node.take(1)
        assert isinstance(elm, Chk)
        return elm.bits

    def clear(self) -> None:
        self.node = _empty

    def __ior__(self, other: "ISet") -> "ISet":
        self.merge(other)
        return self
