"""FlyDSL compiler backend for the hippihx zoo.

HIP fatbins live in ``tiles/``. FlyDSL kernels live here. Same plan / bind /
run verbs; extras V1 consume is still HIP until graph-safe JIT lands.
"""

from __future__ import annotations

from hippihx._lib.backend import (
    FORBIDDEN_FLYDSL_KERNELS,
    FLYDSL_DEST,
    FLYDSL_V1_CONSUME,
    Backend,
)

from .atoms import ATOMS, FDOT2, FORBIDDEN_ATOMS, SDOT4, DotAtom
from .kernels import KERNELS, VEC_ADD, FlyKernel
from .runtime import available, env_arch, refuse_shipped_kernel, require
from .vec_add import launch as launch_vec_add

__all__ = [
    "ATOMS",
    "Backend",
    "DotAtom",
    "FDOT2",
    "FORBIDDEN_ATOMS",
    "FORBIDDEN_FLYDSL_KERNELS",
    "FLYDSL_DEST",
    "FLYDSL_V1_CONSUME",
    "FlyKernel",
    "KERNELS",
    "SDOT4",
    "VEC_ADD",
    "available",
    "env_arch",
    "launch_vec_add",
    "refuse_shipped_kernel",
    "require",
]
