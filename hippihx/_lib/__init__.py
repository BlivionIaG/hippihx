"""Shared zoo spine: catalog, plan/bind/run, fatbin slots, backends.

Kernel guts live in ``tiles/``. Serve wiring stays in ``rdna_extras``.
"""

from .backend import DEST_BACKEND, FATBIN_BACKEND, FLYDSL_DEST, FLYDSL_V1_CONSUME, Backend
from .catalog import OPS, OpSpec, find_spec, list_qualnames
from .fatbin import (
    BIND_KEYS_ON_ARCH_AND_WAVE,
    DEFAULT_ARCH,
    DOT_ARCHES,
    DOT_WAVE,
    GFX1030_WAVE,
    GFX103X_WAVE,
    KNOWN_ARCHES,
    LATER_ARCHES,
    LATER_NONDOT_ARCHES,
    NO_FOREIGN_ISA_LOAD,
    NO_HSA_OVERRIDE,
    ROCM_PIN,
    UNOPTIMIZED_DOT_ARCHES,
    VERIFY_WAVE_ARCHES,
)
from .protocol import Binding, Caps, OpMeta, Plan, ScratchSpec

__all__ = [
    "BIND_KEYS_ON_ARCH_AND_WAVE",
    "Backend",
    "Binding",
    "Caps",
    "DEFAULT_ARCH",
    "DEST_BACKEND",
    "DOT_ARCHES",
    "DOT_WAVE",
    "FATBIN_BACKEND",
    "FLYDSL_DEST",
    "FLYDSL_V1_CONSUME",
    "GFX1030_WAVE",
    "GFX103X_WAVE",
    "KNOWN_ARCHES",
    "LATER_ARCHES",
    "LATER_NONDOT_ARCHES",
    "NO_FOREIGN_ISA_LOAD",
    "NO_HSA_OVERRIDE",
    "OPS",
    "OpMeta",
    "OpSpec",
    "Plan",
    "ROCM_PIN",
    "ScratchSpec",
    "UNOPTIMIZED_DOT_ARCHES",
    "VERIFY_WAVE_ARCHES",
    "find_spec",
    "list_qualnames",
]
