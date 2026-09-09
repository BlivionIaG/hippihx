"""Mirror of include/hippihx/v1.h for host tests / serve planners.

Stable C ABI lives in the fatbin (``hippihx_v1_*``). This module is the
Python-side id table so ``list_ops()`` and the C enum cannot drift without
a test failure. Torch registration stays in rdna_extras.
"""

from __future__ import annotations

from enum import IntEnum

# Keep in lockstep with HIPPIHX_V1_ABI_REVISION in include/hippihx/v1.h.
ABI_REVISION = 1


class V1OpId(IntEnum):
    ATTN_FA_FDOT2 = 0
    ATTN_GDN_SCAN = 1
    ATTN_KDA_SCAN = 2
    ATTN_QSA_INDEXER = 3
    ATTN_DSA_NOPE = 4
    GEMM_W4A16_FDOT2 = 5
    GEMM_EXL3_3INST = 6
    MOE_ROUTED = 7
    MOE_SHARED = 8
    MOE_LEFTOVER_BF16 = 9
    SEQUENCE_CAUSAL_CONV = 10
    COMM_PCIE = 11


# Order must match V1OpId values, include/hippihx/v1.h, and list_ops().
V1_OP_NAMES: tuple[str, ...] = (
    "attn.fa_fdot2",
    "attn.gdn_scan",
    "attn.kda_scan",
    "attn.qsa_indexer",
    "attn.dsa_nope",
    "gemm.w4a16_fdot2",
    "gemm.exl3_3inst",
    "moe.routed",
    "moe.shared",
    "moe.leftover_bf16",
    "sequence.causal_conv",
    "comm.pcie",
)

_DOT_OPS: frozenset[str] = frozenset(
    {
        "attn.fa_fdot2",
        "gemm.w4a16_fdot2",
        "gemm.exl3_3inst",
        "moe.shared",
    }
)


def v1_op_name(op: V1OpId | int) -> str:
    return V1_OP_NAMES[int(op)]


def v1_op_is_dot(op: V1OpId | int) -> bool:
    return V1_OP_NAMES[int(op)] in _DOT_OPS


__all__ = [
    "ABI_REVISION",
    "V1OpId",
    "V1_OP_NAMES",
    "v1_op_is_dot",
    "v1_op_name",
]
