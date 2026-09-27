"""Mojo/MAX authoring surface for the hippihx zoo.

Plan and bind accept ``Caps(backend="mojo")``. ``run`` raises
``MojoNotProduce``. Mojo objects are not dest-ready. Dest produce for
vllm-rdna stays hipcc 7.14 / ``hipModuleLoad`` / ``libamdhip64`` until a
documented HIP re-emit or MAX-serve soak exists. The ABI gap is MAX
``execute`` versus ``hippihx_v1_*``.

Sources live in the repo ``mojo/`` directory (MAX ``custom_extensions``
package). They are not CMake fatbin inputs.
"""

from __future__ import annotations

from pathlib import Path

from hippihx._lib.backend import (
    MOJO_AUTHORING,
    MOJO_DEST_READY,
    MOJO_NOT_PRODUCE_MSG,
    MOJO_PRODUCE,
    MOJO_V1_CONSUME,
    MojoNotProduce,
)
from hippihx._lib.explore import ExploreShape, preferred_explore, ranked_explore
from hippihx._lib.families import (
    HOOKS,
    LEFT_WITHOUT_NEW_BRIEF,
    FamilyHook,
    TensorContract,
    hooks_for,
    require_family,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "mojo"
FA_FDOT2_MOJO = SOURCE_ROOT / "zoo" / "fa_fdot2.mojo"
ISA_MOJO = SOURCE_ROOT / "isa" / "contracts.mojo"

__all__ = [
    "FA_FDOT2_MOJO",
    "HOOKS",
    "ISA_MOJO",
    "LEFT_WITHOUT_NEW_BRIEF",
    "MOJO_AUTHORING",
    "MOJO_DEST_READY",
    "MOJO_NOT_PRODUCE_MSG",
    "MOJO_PRODUCE",
    "MOJO_V1_CONSUME",
    "SOURCE_ROOT",
    "ExploreShape",
    "FamilyHook",
    "TensorContract",
    "MojoNotProduce",
    "hooks_for",
    "preferred_explore",
    "ranked_explore",
    "refuse_produce",
    "require_family",
]


def refuse_produce() -> None:
    """Fail closed. There is no Mojo object to enqueue."""

    raise MojoNotProduce(MOJO_NOT_PRODUCE_MSG)
