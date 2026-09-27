"""Mirror of include/hippihx/v1.h (ABI rev 4). Ids and schemas come from ``catalog.OPS``.

Rev 4 is plan, then run:

- ``hippihx_v1_plan`` is host-only and runs before capture, once per
  capture bucket. It checks caps and params, sizes scratch, picks the
  explore variant and reports ``ready``.
- ``hippihx_v1_run`` takes that plan, tensor descriptors in slot order,
  the zeroed scratch block and a stream. It never allocates and never
  reads device memory.

The helpers below are the same checks in Python. ``StubOp.plan`` uses them
when params are given, so a sized Python plan and the C plan agree.
``include/hippihx/v1.h`` and ``tiles/v1_abi.cpp`` are generated from this
module and the catalog.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

from .catalog import OPS, TENSOR_DTYPES, Cond, OpSpec, ParamSpec
from .explore import ranked_explore
from .fabric import HOPS, SWITCHES, Fabric, custom_ar_fabric_ok
from .protocol import ScratchSpec

# Keep in lockstep with HIPPIHX_V1_ABI_REVISION in include/hippihx/v1.h.
# Rev 2: caps.dtype + HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE (no fdot2.bf16).
# Rev 3: qualnames attn.* → attention.* (ids unchanged).
# Rev 4: caller-owned plan (ready, variant, scratch offsets), params,
#        tensor descriptors in slot order, stream, caps.fabric, and the
#        per-slot code object loader (hippihx_v1_load*).
ABI_REVISION = 4

MAX_RANK = 6
MAX_PARAMS = 16
MAX_SCRATCH = 4
SCRATCH_ALIGN = 256
# Scratch byte counts and offsets are int64 in C.
INT64_MAX = 2**63 - 1


class V1Status(IntEnum):
    """``HIPPIHX_V1_*`` return codes. Append only."""

    OK = 0
    ERR_UNKNOWN_OP = 1
    ERR_UNSUPPORTED_ARCH = 2
    ERR_BAD_ARG = 3
    ERR_SCRATCH = 4
    ERR_NOT_READY = 5
    ERR_UNSUPPORTED_DTYPE = 6
    ERR_ABI = 7
    ERR_PARAM = 8
    ERR_TENSOR = 9
    ERR_UNSUPPORTED_FABRIC = 10
    ERR_CODE_OBJECT = 11
    ERR_FOREIGN_ISA = 12
    ERR_NO_HIP = 13


STATUS_NOTES: dict[V1Status, str] = {
    V1Status.ERR_UNSUPPORTED_ARCH: "slot not built for this op, or wave not allowed",
    V1Status.ERR_BAD_ARG: "NULL pointer or malformed fabric",
    V1Status.ERR_SCRATCH: "scratch too small, NULL, or misaligned",
    V1Status.ERR_NOT_READY: "contract exists; HIP body not migrated",
    V1Status.ERR_UNSUPPORTED_DTYPE: "e.g. bf16 on fdot2 / GDN HIP",
    V1Status.ERR_ABI: "struct_size / revision mismatch, or a failed plan",
    V1Status.ERR_PARAM: "param count, domain, cross-check, or scratch overflow",
    V1Status.ERR_TENSOR: "tensor count, dtype, rank, extent, layout, or NULL data",
    V1Status.ERR_UNSUPPORTED_FABRIC: "comm: not a custom-AR hop; serve keeps RCCL",
    V1Status.ERR_CODE_OBJECT: "not one raw AMDGPU ELF code object, or unreadable",
    V1Status.ERR_FOREIGN_ISA: "code object mach is not the slot's, or HSA_OVERRIDE set",
    V1Status.ERR_NO_HIP: "library built without HIP (host stub); nothing loaded",
}

# Element dtype codes. 0 is unset. Caps activation dtypes are fp16 / bf16 / fp32.
V1_DTYPES: dict[str, int] = {name: code for code, name in enumerate(TENSOR_DTYPES, 1)}
DTYPE_BYTES: dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "i32": 4,
    "i64": 8,
    "u8": 1,
    "i8": 1,
}

# Fabric class codes for hippihx_v1_fabric. 0 is unset.
V1_HOPS: dict[str, int] = {hop: code for code, hop in enumerate(HOPS, 1)}
V1_SWITCHES: dict[str, int] = {name: code for code, name in enumerate(SWITCHES, 1)}

V1OpId = IntEnum("V1OpId", {spec.enum: spec.v1_id for spec in OPS})

V1_OP_NAMES: tuple[str, ...] = tuple(spec.qualname for spec in OPS)

_DOT_OPS: frozenset[str] = frozenset(spec.qualname for spec in OPS if spec.dot)
_FP16_ACT_OPS: frozenset[str] = frozenset(
    spec.qualname for spec in OPS if spec.fp16_act
)


class V1Error(ValueError):
    """A plan the C ABI would refuse. ``status`` is the C return code."""

    def __init__(self, status: V1Status, message: str) -> None:
        super().__init__(f"{status.name}: {message}")
        self.status = status


def v1_op_name(op: V1OpId | int) -> str:
    return V1_OP_NAMES[int(op)]


def v1_op_is_dot(op: V1OpId | int) -> bool:
    return V1_OP_NAMES[int(op)] in _DOT_OPS


def v1_op_fp16_act(op: V1OpId | int) -> bool:
    return V1_OP_NAMES[int(op)] in _FP16_ACT_OPS


def _param(spec: OpSpec, name: str) -> ParamSpec:
    for param in spec.params:
        if param.name == name:
            return param
    raise KeyError(f"{spec.qualname} has no param {name!r}")


def _coerce(spec: OpSpec, param: ParamSpec, raw: Any) -> int | float:
    where = f"{spec.qualname}.{param.name}"
    if param.kind == "float":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise V1Error(V1Status.ERR_PARAM, f"{where} must be a number, got {raw!r}")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise V1Error(V1Status.ERR_PARAM, f"{where} must be finite and > 0, got {raw!r}")
        return value
    if param.kind == "enum" and isinstance(raw, str):
        if raw not in param.labels:
            raise V1Error(
                V1Status.ERR_PARAM, f"{where}: {raw!r} not in {param.labels}"
            )
        raw = param.values[param.labels.index(raw)]
    if isinstance(raw, bool):
        if param.kind != "bool":
            raise V1Error(V1Status.ERR_PARAM, f"{where} is {param.kind}, got a bool")
        raw = int(raw)
    if not isinstance(raw, int):
        raise V1Error(V1Status.ERR_PARAM, f"{where} must be an int, got {raw!r}")
    if param.kind == "bool" and raw not in (0, 1):
        raise V1Error(V1Status.ERR_PARAM, f"{where} must be 0 or 1, got {raw}")
    if param.kind == "enum" and raw not in param.values:
        raise V1Error(V1Status.ERR_PARAM, f"{where}: {raw} not in {param.values}")
    if param.kind == "int":
        if param.lo is not None and raw < param.lo:
            raise V1Error(V1Status.ERR_PARAM, f"{where}={raw} is below {param.lo}")
        if param.hi is not None and raw > param.hi:
            raise V1Error(V1Status.ERR_PARAM, f"{where}={raw} is above {param.hi}")
        if not -(2**63) <= raw <= INT64_MAX:
            raise V1Error(V1Status.ERR_PARAM, f"{where}={raw} does not fit int64")
    return raw


def param_values(spec: OpSpec, params: Mapping[str, Any]) -> tuple[int | float, ...]:
    """``params`` in schema order, checked like ``hippihx_v1_plan``.

    Enum params also accept their label (``mode="decode"``). Raises
    ``V1Error(ERR_PARAM)``.
    """

    names = [param.name for param in spec.params]
    unknown = sorted(set(params) - set(names))
    missing = [name for name in names if name not in params]
    if unknown or missing:
        raise V1Error(
            V1Status.ERR_PARAM,
            f"{spec.qualname} params: unknown {unknown}, missing {missing}; "
            f"schema {tuple(names)}",
        )
    values = tuple(_coerce(spec, param, params[param.name]) for param in spec.params)
    by_name = dict(zip(names, values))
    for check in spec.checks:
        num, by = by_name[check.num], by_name[check.by]
        if by == 0 or num % by != 0:
            raise V1Error(
                V1Status.ERR_PARAM,
                f"{spec.qualname}: {check.num}={num} is not a multiple of "
                f"{check.by}={by}",
            )
    return values


def _cond_holds(cond: Cond, value: int) -> bool:
    return {
        "==": value == cond.value,
        "!=": value != cond.value,
        ">": value > cond.value,
        ">=": value >= cond.value,
        "<": value < cond.value,
        "<=": value <= cond.value,
    }[cond.op]


def cond_any(spec: OpSpec, conds: Sequence[Cond], values: Sequence[int | float]) -> bool:
    """True when ``conds`` is empty or any cond holds."""

    if not conds:
        return True
    names = [param.name for param in spec.params]
    return any(_cond_holds(cond, int(values[names.index(cond.param)])) for cond in conds)


def align_up(nbytes: int) -> int:
    return -(-nbytes // SCRATCH_ALIGN) * SCRATCH_ALIGN


def scratch_layout(
    spec: OpSpec, values: Sequence[int | float]
) -> tuple[tuple[ScratchSpec, ...], int]:
    """Scratch specs with offsets, and the block size to allocate.

    Every spec is zeroed once by serve. Offsets and the total are
    multiples of ``SCRATCH_ALIGN``. Raises ``V1Error(ERR_PARAM)`` past the
    rule's ``max_elems`` or int64.
    """

    names = [param.name for param in spec.params]
    specs: list[ScratchSpec] = []
    cursor = 0
    for rule in spec.scratch:
        if not cond_any(spec, rule.when, values):
            continue
        elems = 1
        for dim in rule.dims:
            factor = int(dim) if dim.isdigit() else int(values[names.index(dim)])
            if factor < 1:
                raise V1Error(V1Status.ERR_PARAM, f"{spec.qualname} scratch {rule.name}: {dim} < 1")
            elems *= factor
        if rule.max_elems and elems > rule.max_elems:
            raise V1Error(
                V1Status.ERR_PARAM,
                f"{spec.qualname} scratch {rule.name}: {elems} elements exceed "
                f"{rule.max_elems} (kernel index width)",
            )
        nbytes = elems * rule.elem_bytes
        offset = align_up(cursor)
        cursor = offset + nbytes
        if cursor > INT64_MAX - SCRATCH_ALIGN:
            raise V1Error(V1Status.ERR_PARAM, f"{spec.qualname} scratch overflows int64")
        specs.append(ScratchSpec(name=rule.name, nbytes=nbytes, offset=offset))
    return tuple(specs), align_up(cursor)


def plan_variant(spec: OpSpec, arch: str, values: Sequence[int | float]) -> int:
    """Explore rank the plan picks. 0 when the op has no variant param."""

    if not spec.variant:
        return 0
    param = _param(spec, spec.variant)
    names = [p.name for p in spec.params]
    label = param.labels[param.values.index(int(values[names.index(param.name)]))]
    for shape in ranked_explore(spec.qualname, arch):
        if shape.name == label:
            return shape.rank
    raise KeyError(f"{spec.qualname}: no explore shape {label!r} on {arch}")


def variant_ranks(spec: OpSpec, arch: str) -> tuple[int, ...]:
    """Explore rank for each value of the variant param (C table order)."""

    if not spec.variant:
        return ()
    param = _param(spec, spec.variant)
    ranks = {shape.name: shape.rank for shape in ranked_explore(spec.qualname, arch)}
    return tuple(ranks[label] for label in param.labels)


def fabric_status(fabric: Fabric | None) -> V1Status:
    """``OK`` on a custom-AR hop, else ``ERR_UNSUPPORTED_FABRIC`` (RCCL)."""

    return V1Status.OK if custom_ar_fabric_ok(fabric) else V1Status.ERR_UNSUPPORTED_FABRIC


def arena_layout(sizes: Sequence[int]) -> tuple[tuple[int, ...], int]:
    """Pack several plans' scratch blocks into one zeroed arena.

    ``sizes`` are ``Plan.scratch_nbytes``. Returns each block's offset
    (aligned for ``hippihx_v1_run``) and the arena size.
    """

    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        if size < 0:
            raise ValueError("scratch size must be >= 0")
        cursor = align_up(cursor)
        offsets.append(cursor)
        cursor += size
    return tuple(offsets), align_up(cursor)


__all__ = [
    "ABI_REVISION",
    "DTYPE_BYTES",
    "MAX_PARAMS",
    "MAX_RANK",
    "MAX_SCRATCH",
    "SCRATCH_ALIGN",
    "V1Error",
    "V1OpId",
    "V1Status",
    "V1_DTYPES",
    "V1_HOPS",
    "V1_OP_NAMES",
    "V1_SWITCHES",
    "align_up",
    "arena_layout",
    "cond_any",
    "fabric_status",
    "param_values",
    "plan_variant",
    "scratch_layout",
    "v1_op_fp16_act",
    "v1_op_is_dot",
    "v1_op_name",
]
