"""Mirror of include/hippihx/v1.h. Ids come from ``catalog.OPS``."""

from __future__ import annotations

from enum import IntEnum

from .catalog import OPS

# Keep in lockstep with HIPPIHX_V1_ABI_REVISION in include/hippihx/v1.h.
# Rev 3: qualnames attn.* → attention.* (ids unchanged).
ABI_REVISION = 3

V1OpId = IntEnum("V1OpId", {spec.enum: spec.v1_id for spec in OPS})

V1_OP_NAMES: tuple[str, ...] = tuple(spec.qualname for spec in OPS)

_DOT_OPS: frozenset[str] = frozenset(spec.qualname for spec in OPS if spec.dot)
_FP16_ACT_OPS: frozenset[str] = frozenset(
    spec.qualname for spec in OPS if spec.fp16_act
)


def v1_op_name(op: V1OpId | int) -> str:
    return V1_OP_NAMES[int(op)]


def v1_op_is_dot(op: V1OpId | int) -> bool:
    return V1_OP_NAMES[int(op)] in _DOT_OPS


def v1_op_fp16_act(op: V1OpId | int) -> bool:
    return V1_OP_NAMES[int(op)] in _FP16_ACT_OPS


__all__ = [
    "ABI_REVISION",
    "V1OpId",
    "V1_OP_NAMES",
    "v1_op_fp16_act",
    "v1_op_is_dot",
    "v1_op_name",
]
