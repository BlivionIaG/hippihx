"""gfx1030 DOT atoms for the FlyDSL compiler backend.

The packed-DOT names and LLVM intrinsics come from ``hippihx._lib.isa``.
This module only adds the FlyDSL atom name. Never WMMA, MFMA,
``fdot2.bf16``, or ``sudot4`` (mixed-sign DOT is gfx11+).
"""

from __future__ import annotations

from dataclasses import dataclass

from hippihx._lib.isa import FDOT2 as ISA_FDOT2
from hippihx._lib.isa import FORBIDDEN_ISA
from hippihx._lib.isa import SDOT4 as ISA_SDOT4


@dataclass(frozen=True, slots=True)
class DotAtom:
    name: str
    llvm: str
    isa: tuple[str, ...]
    a: str
    b: str
    acc: str
    k: int
    clamp: bool = False


FDOT2 = DotAtom(
    name="fly_fdot2",
    llvm=ISA_FDOT2.llvm,
    isa=ISA_FDOT2.isa,
    a="half2",
    b="half2",
    acc="f32",
    k=ISA_FDOT2.k,
)

SDOT4 = DotAtom(
    name="fly_sdot4",
    llvm=ISA_SDOT4.llvm,
    isa=ISA_SDOT4.isa,
    a="i32",  # packed little-endian i8x4
    b="i32",
    acc="i32",
    k=ISA_SDOT4.k,
)

ATOMS: tuple[DotAtom, ...] = (FDOT2, SDOT4)

FORBIDDEN_ATOMS: tuple[str, ...] = FORBIDDEN_ISA
