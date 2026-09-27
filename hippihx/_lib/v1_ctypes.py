"""ctypes view of include/hippihx/v1.h (rev 4) for host tools and tests.

``load(path)`` opens any shared object that exports ``hippihx_v1_*``
(for example ``tiles/v1_abi.cpp`` built with ``-shared``). Importing this
module loads nothing. The structs mirror the header by hand; the parity
test checks them against a compiled library.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .catalog import OpSpec
from .fabric import Fabric as HostFabric
from .v1 import ABI_REVISION, MAX_PARAMS, MAX_RANK, MAX_SCRATCH, V1_HOPS, V1_SWITCHES

c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64


class Param(ctypes.Union):
    _fields_ = [("i", c_int64), ("f", ctypes.c_double)]


class Fabric(ctypes.Structure):
    _fields_ = [
        ("hop", c_int32),
        ("pcie_switch", c_int32),
        ("width", c_int32),
        ("acs_clear", c_int32),
        ("large_bar", c_int32),
    ]


class Caps(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_char_p),
        ("wave", ctypes.c_int),
        ("dtype", ctypes.c_int),
        ("fabric", ctypes.POINTER(Fabric)),
    ]


class Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", c_int32),
        ("rank", c_int32),
        ("shape", c_int64 * MAX_RANK),
        ("stride", c_int64 * MAX_RANK),
    ]


class ScratchSpec(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("nbytes", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
        ("zeroed", ctypes.c_int),
    ]


class Plan(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_revision", c_int32),
        ("op", c_int32),
        ("arch", c_int32),
        ("wave", c_int32),
        ("dtype", c_int32),
        ("ready", c_int32),
        ("variant", c_int32),
        ("nparams", c_int32),
        ("params", Param * MAX_PARAMS),
        ("nscratch", c_int32),
        ("scratch", ScratchSpec * MAX_SCRATCH),
        ("scratch_nbytes", ctypes.c_size_t),
    ]


def new_plan() -> Plan:
    plan = Plan()
    plan.struct_size = ctypes.sizeof(Plan)
    return plan


def fabric_struct(fabric: HostFabric) -> Fabric:
    return Fabric(
        hop=V1_HOPS[fabric.hop],
        pcie_switch=V1_SWITCHES[fabric.switch],
        width=fabric.width,
        acs_clear=int(fabric.acs_clear),
        large_bar=int(fabric.large_bar),
    )


def params_array(spec: OpSpec, values: Mapping[str, Any]) -> Any:
    """``hippihx_v1_param[]`` in schema order from checked plan params."""

    array = (Param * max(1, len(spec.params)))()
    for index, param in enumerate(spec.params):
        if param.kind == "float":
            array[index].f = float(values[param.name])
        else:
            array[index].i = int(values[param.name])
    return array


def tensor(data: int, dtype: int, shape: Sequence[int], stride: Sequence[int] | None = None) -> Tensor:
    """Descriptor; contiguous row-major strides when ``stride`` is None."""

    if stride is None:
        stride, step = [], 1
        for extent in reversed(shape):
            stride.insert(0, step)
            step *= extent
    out = Tensor(data=data, dtype=dtype, rank=len(shape))
    for axis, (extent, step) in enumerate(zip(shape, stride)):
        out.shape[axis] = extent
        out.stride[axis] = step
    return out


def load(path: str | Path) -> ctypes.CDLL:
    """Open a V1 library, declare signatures, refuse a revision mismatch."""

    lib = ctypes.CDLL(str(path))
    lib.hippihx_v1_abi_revision.restype = ctypes.c_int
    lib.hippihx_v1_abi_revision.argtypes = []
    lib.hippihx_v1_op_count.restype = ctypes.c_int
    lib.hippihx_v1_op_count.argtypes = []
    for name in ("op_is_dot", "op_fp16_act", "op_nparams", "op_ntensors"):
        fn = getattr(lib, f"hippihx_v1_{name}")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int]
    lib.hippihx_v1_op_name.restype = ctypes.c_char_p
    lib.hippihx_v1_op_name.argtypes = [ctypes.c_int]
    for name in ("op_param_name", "op_tensor_name"):
        fn = getattr(lib, f"hippihx_v1_{name}")
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.hippihx_v1_plan.restype = ctypes.c_int
    lib.hippihx_v1_plan.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(Caps),
        ctypes.POINTER(Param),
        ctypes.c_size_t,
        ctypes.POINTER(Plan),
    ]
    lib.hippihx_v1_run.restype = ctypes.c_int
    lib.hippihx_v1_run.argtypes = [
        ctypes.POINTER(Plan),
        ctypes.POINTER(Tensor),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    revision = lib.hippihx_v1_abi_revision()
    if revision != ABI_REVISION:
        raise RuntimeError(
            f"{path}: V1 ABI revision {revision}, this hippihx speaks {ABI_REVISION}"
        )
    return lib


__all__ = [
    "Caps",
    "Fabric",
    "Param",
    "Plan",
    "ScratchSpec",
    "Tensor",
    "fabric_struct",
    "load",
    "new_plan",
    "params_array",
    "tensor",
]
