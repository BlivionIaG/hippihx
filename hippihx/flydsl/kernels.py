"""FlyDSL kernel contracts. Bodies land here, not as ROCm MFMA/WMMA ports."""

from __future__ import annotations

from dataclasses import dataclass

from hippihx._lib.fatbin import DEFAULT_ARCH, DOT_WAVE


@dataclass(frozen=True, slots=True)
class FlyKernel:
    name: str
    arch: str
    wave: int
    summary: str


VEC_ADD = FlyKernel(
    name="vec_add",
    arch=DEFAULT_ARCH,
    wave=DOT_WAVE,
    summary="gfx1030 vector-add; wave32; no gfx11/12 opcodes",
)

KERNELS: tuple[FlyKernel, ...] = (VEC_ADD,)
