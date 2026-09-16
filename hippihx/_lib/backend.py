"""Zoo backends: HIP fatbins and the FlyDSL compiler.

Both are dest *zoo* backends (plan / bind / run). extras V1 consume is still
the HIP fatbin (`hippihx_v1_*`). FlyDSL extras consume waits on graph-safe
JIT. Do not port ROCm/FlyDSL MFMA/WMMA GEMM/MoE/FA kernels. FlyDSL is an
optional extra, not a required dependency.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    HIP = "hip"
    FLYDSL = "flydsl"


FATBIN_BACKEND = Backend.HIP
DEST_BACKEND = FATBIN_BACKEND
FLYDSL_DEST = True
FLYDSL_V1_CONSUME = False

# Kernel-readiness in *this* tree. Caps does not wait on these.
FLYDSL_GATE0_OBJECT = False
FLYDSL_GATE1_DOT_WRAPPERS = False
FLYDSL_GATE2_SKINNY = False
FLYDSL_GATE3_TABLE = False

FORBIDDEN_FLYDSL_KERNELS: tuple[str, ...] = (
    "preshuffle_gemm",
    "moe_gemm_2stage",
    "rdna3_f16_gemm",
    "rdna_f16_gemm",
    "rdna_fp8_preshuffle_gemm",
    "flash_attn_generic",
)


def is_zoo_backend(backend: Backend | str) -> bool:
    try:
        return Backend(backend) in (Backend.HIP, Backend.FLYDSL)
    except ValueError:
        return False


def is_dest_backend(backend: Backend | str) -> bool:
    return is_zoo_backend(backend)


def flydsl_ready() -> bool:
    """Graph-safe extras consume. Zoo Caps do not wait on this."""
    return FLYDSL_V1_CONSUME
