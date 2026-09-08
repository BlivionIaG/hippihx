"""hippihx — HIP kernel / op zoo (gfx1030 + gfx1100 shared DOT source).

Library, not a serve stack. ``rdna_extras`` wires one ``torch.ops`` entry
per op. Pack produce (AWQ / ``3inst``) stays outside this tree.

Import is cheap and torch-free. Device libraries load later, per fatbin.
"""

from __future__ import annotations

import importlib
from typing import Final

from .protocol import (
    DEFAULT_ARCH,
    DOT_ARCHES,
    GFX1030_WAVE,
    KNOWN_ARCHES,
    LATER_ARCHES,
    ROCM_PIN,
    Binding,
    Caps,
    OpMeta,
    Plan,
    ScratchSpec,
)

__version__ = "0.0.0"

# Lockstep with tiles/<group>/<op>/ and hippihx/<group>/<op>/.
_OPS: Final[tuple[str, ...]] = (
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


def list_ops() -> tuple[OpMeta, ...]:
    """Return ``META`` for every registered tile contract."""
    return tuple(
        importlib.import_module(f".{qualname}", __name__).META for qualname in _OPS
    )


def find_op(qualname: str) -> OpMeta:
    if qualname not in _OPS:
        raise KeyError(f"unknown hippihx op {qualname!r}; known: {list(_OPS)}")
    return importlib.import_module(f".{qualname}", __name__).META


__all__ = [
    "Binding",
    "Caps",
    "DEFAULT_ARCH",
    "DOT_ARCHES",
    "GFX1030_WAVE",
    "KNOWN_ARCHES",
    "LATER_ARCHES",
    "OpMeta",
    "Plan",
    "ROCM_PIN",
    "ScratchSpec",
    "__version__",
    "find_op",
    "list_ops",
]
