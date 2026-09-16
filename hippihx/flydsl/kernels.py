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
    summary="dest gfx1030 vector-add (FlyDSL 01-vectorAdd; wave32)",
)

KERNELS: tuple[FlyKernel, ...] = (VEC_ADD,)
