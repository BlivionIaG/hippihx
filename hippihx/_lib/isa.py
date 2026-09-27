"""Durable ISA contracts. Emit language does not own these.

HIP fatbins (``tiles/``), FlyDSL atoms, and the Mojo/MAX authoring tree
(``mojo/``) all read this table. A Mojo object is not a second copy of the
contract and is not vllm-rdna produce — see ``hippihx.mojo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .fatbin import DOT_ARCHES, LATER_ARCHES


@dataclass(frozen=True, slots=True)
class PackedDot:
    """gfx1030 packed DOT. ``fdot2`` / ``v_dot2c`` and ``sdot4`` / ``v_dot4``."""

    name: str
    llvm: str
    isa: tuple[str, ...]
    k: int


FDOT2 = PackedDot(
    name="fdot2",
    llvm="llvm.amdgcn.fdot2",
    isa=("v_dot2_f32_f16", "v_dot2c_f32_f16"),
    k=2,
)
SDOT4 = PackedDot(
    name="sdot4",
    llvm="llvm.amdgcn.sdot4",
    isa=("v_dot4_i32_i8", "v_dot4c_i32_i8"),
    k=4,
)
PACKED_DOT: tuple[PackedDot, ...] = (FDOT2, SDOT4)

# gfx1030 shared DOT has no matrix-core gate. WMMA is not a required path.
WMMA_GATE_GFX1030 = False
NO_WMMA_GATE = True

# RDNA LDS is 32 banks × 4 bytes. W4 A-tile pad is the locked bank-conflict
# contract (extras ``LDS_PAD=8`` on ``q_gemm`` / ``moe_q_gemm``).
LDS_BANKS = 32
LDS_BANK_BYTES = 4
LDS_PAD = 8

# DOT tiles are wave32 only. This is an occupancy gate, not a tok/s claim.
WAVE32 = 32

# QSA indexer observation on gfx1030 (6h×256 prefill): 4 warps / 128 threads.
# A 2-warp profile spills. Recorded so explore does not rediscover it.
QSA_OCCUPANCY_WARPS = 4
QSA_OCCUPANCY_THREADS = 128
QSA_SPILL_WARPS = 2

# gfx1030 DOT objects stay off these paths. gfx900 is the built mad_mix stub.
# gfx906 and gfx1013 are Later (not built). None of them load FA/EXL3 DOT.
MAD_MIX_ARCHES: tuple[str, ...] = ("gfx900",) + tuple(LATER_ARCHES)

FORBIDDEN_ISA: tuple[str, ...] = (
    "WMMA",
    "MFMA",
    "sudot4",
    "fdot2.bf16",
    "llvm.amdgcn.fdot2.bf16",
)


class ArchSwitch(str, Enum):
    """Where an object is allowed to live. One switch per arch, not per op."""

    DOT = "dot"
    MAD_MIX = "mad_mix"


def arch_switch(arch: str) -> ArchSwitch:
    """Classify ``arch`` for object placement.

    ``dot`` is the gfx1030 primary slot plus the portable DOT fatbins
    (gfx110x, gfx1151, gfx103x). ``mad_mix`` is gfx900/gfx906/gfx1013 —
    do not ship gfx1030 packed-DOT objects there.
    """

    if arch in DOT_ARCHES:
        return ArchSwitch.DOT
    if arch in MAD_MIX_ARCHES:
        return ArchSwitch.MAD_MIX
    raise ValueError(
        f"unknown arch {arch!r}; dot slots {DOT_ARCHES}; mad_mix {MAD_MIX_ARCHES}"
    )


def refuse_dot_on_mad_mix(arch: str) -> str:
    """Return ``arch`` when a DOT object may be emitted for it.

    Raises when ``arch`` is a mad_mix path. Does not allocate and does not
    launch.
    """

    switch = arch_switch(arch)
    if switch is ArchSwitch.MAD_MIX:
        raise ValueError(
            f"do not ship gfx1030 DOT objects onto {arch} mad_mix "
            "(gfx900/gfx906/gfx1013)"
        )
    return arch


def wave_occupancy_ok(*, dot: bool, wave: int) -> bool:
    """DOT tiles refuse any wave other than 32. Non-DOT may be 32 or 64."""

    if dot:
        return wave == WAVE32
    return wave in (WAVE32, 64)
