"""hippihx — HIP/RDNA op zoo (FlyDSL research, not dest).

b12x-shaped library: ``hippihx.<group>.<op>`` owns plan/bind/run. HIP
fatbins live in ``tiles/``. ``rdna_extras`` is thin serve wiring.

Import is cheap and torch-free. Device libraries load later, per fatbin.
"""

from __future__ import annotations

import importlib
from typing import Any, Final

from ._lib.catalog import list_qualnames
from ._lib.fatbin import (
    DEFAULT_ARCH,
    DOT_ARCHES,
    GFX1030_WAVE,
    KNOWN_ARCHES,
    LATER_ARCHES,
    LATER_NONDOT_ARCHES,
    UNOPTIMIZED_DOT_ARCHES,
    VERIFY_WAVE_ARCHES,
    ROCM_PIN,
)
from ._lib.protocol import Binding, Caps, OpMeta, Plan, ScratchSpec
from ._lib.v1 import ABI_REVISION as V1_ABI_REVISION
from ._lib.v1 import V1_OP_NAMES, V1OpId, v1_op_fp16_act, v1_op_is_dot, v1_op_name

__version__ = "0.0.0"

_OPS: Final[tuple[str, ...]] = list_qualnames()
_GROUPS: Final[tuple[str, ...]] = (
    "attention",
    "comm",
    "gemm",
    "moe",
    "sequence",
)


def list_ops() -> tuple[OpMeta, ...]:
    """Return ``META`` for every registered tile contract."""
    return tuple(
        importlib.import_module(f".{qualname}", __name__).META for qualname in _OPS
    )


def find_op(qualname: str) -> OpMeta:
    if qualname not in _OPS:
        raise KeyError(f"unknown hippihx op {qualname!r}; known: {list(_OPS)}")
    return importlib.import_module(f".{qualname}", __name__).META


def __getattr__(name: str) -> Any:
    if name in _GROUPS:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_GROUPS])


__all__ = [
    "Binding",
    "Caps",
    "DEFAULT_ARCH",
    "DOT_ARCHES",
    "GFX1030_WAVE",
    "KNOWN_ARCHES",
    "LATER_ARCHES",
    "LATER_NONDOT_ARCHES",
    "UNOPTIMIZED_DOT_ARCHES",
    "VERIFY_WAVE_ARCHES",
    "OpMeta",
    "Plan",
    "ROCM_PIN",
    "ScratchSpec",
    "V1OpId",
    "V1_ABI_REVISION",
    "V1_OP_NAMES",
    "__version__",
    "find_op",
    "list_ops",
    "v1_op_fp16_act",
    "v1_op_is_dot",
    "v1_op_name",
]
