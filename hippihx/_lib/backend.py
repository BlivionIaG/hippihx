"""HIP is dest. FlyDSL is a research compiler substrate, not a dest backend.

rdna-hip-wiki ``engine/flydsl.md``: FlyDSL may emit gfx1030 objects, but no
optimized GEMM/MoE/FA kernel there is RDNA2. Gates 0–3 (object, fdot2/sdot4
wrappers, skinny proof, fair table) must pass before any vLLM path. Do not
port MFMA/WMMA FlyDSL pipelines. Do not add a FlyDSL wheel dependency.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    HIP = "hip"
    FLYDSL = "flydsl"


DEST_BACKEND = Backend.HIP
FLYDSL_DEST = False

# Wiki gates. All false until silicon evidence lands in this tree.
FLYDSL_GATE0_OBJECT = False
FLYDSL_GATE1_DOT_WRAPPERS = False
FLYDSL_GATE2_SKINNY = False
FLYDSL_GATE3_TABLE = False


def is_dest_backend(backend: Backend | str) -> bool:
    return Backend(backend) is DEST_BACKEND


def flydsl_ready() -> bool:
    return (
        FLYDSL_DEST
        and FLYDSL_GATE0_OBJECT
        and FLYDSL_GATE1_DOT_WRAPPERS
        and FLYDSL_GATE2_SKINNY
        and FLYDSL_GATE3_TABLE
    )
