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
from hippihx._lib.catalog import OPS
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
FAMILIES_MOJO = SOURCE_ROOT / "zoo" / "families.mojo"
ISA_MOJO = SOURCE_ROOT / "isa" / "contracts.mojo"
REFUSE_MOJO = SOURCE_ROOT / "zoo" / "refuse.mojo"

# Catalog ops with a MAX registration. KDA / DSA stay catalog stubs.
AUTHORING_QUALNAMES: tuple[str, ...] = tuple(
    spec.qualname for spec in OPS if spec.qualname not in LEFT_WITHOUT_NEW_BRIEF
)

__all__ = [
    "AUTHORING_QUALNAMES",
    "FA_FDOT2_MOJO",
    "FAMILIES_MOJO",
    "HOOKS",
    "ISA_MOJO",
    "LEFT_WITHOUT_NEW_BRIEF",
    "REFUSE_MOJO",
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
    "authoring_source",
    "hooks_for",
    "preferred_explore",
    "ranked_explore",
    "refuse_produce",
    "require_family",
]


def authoring_source(qualname: str) -> Path:
    """Path of the MAX registration for ``qualname``.

    Raises for catalog stubs that were left without a Mojo brief.
    """

    if qualname not in AUTHORING_QUALNAMES:
        raise KeyError(
            f"{qualname} has no Mojo registration; "
            f"left without a brief: {LEFT_WITHOUT_NEW_BRIEF}"
        )
    stem = qualname.split(".", 1)[1]
    return SOURCE_ROOT / "zoo" / f"{stem}.mojo"


def refuse_produce() -> None:
    """Fail closed. There is no Mojo object to enqueue."""

    raise MojoNotProduce(MOJO_NOT_PRODUCE_MSG)
