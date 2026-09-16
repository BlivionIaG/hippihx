"""gfx1030 DOT atoms for the FlyDSL compiler backend.

Prefer LLVM intrinsics. Never WMMA, MFMA, ``fdot2.bf16``, or ``sudot4``
(mixed-sign DOT is gfx11+).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    llvm="llvm.amdgcn.fdot2",
    isa=("v_dot2_f32_f16", "v_dot2c_f32_f16"),
    a="half2",
    b="half2",
    acc="f32",
    k=2,
)

SDOT4 = DotAtom(
    name="fly_sdot4",
    llvm="llvm.amdgcn.sdot4",
    isa=("v_dot4_i32_i8", "v_dot4c_i32_i8"),
    a="i32",  # packed little-endian i8x4
    b="i32",
    acc="i32",
    k=4,
)

ATOMS: tuple[DotAtom, ...] = (FDOT2, SDOT4)

FORBIDDEN_ATOMS: tuple[str, ...] = (
    "WMMA",
    "MFMA",
    "sudot4",
    "fdot2.bf16",
    "llvm.amdgcn.fdot2.bf16",
)
