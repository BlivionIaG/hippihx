"""Public ISA contracts. Emit language does not own this module.

``import hippihx.isa`` is the host API. The C mirror is
``include/hippihx/isa.hpp``. The Mojo mirror is ``mojo/isa/contracts.mojo``.
"""

from hippihx._lib.isa import (
    FDOT2,
    FORBIDDEN_ISA,
    LDS_BANK_BYTES,
    LDS_BANKS,
    LDS_PAD,
    MAD_MIX_ARCHES,
    NO_WMMA_GATE,
    PACKED_DOT,
    QSA_OCCUPANCY_THREADS,
    QSA_OCCUPANCY_WARPS,
    QSA_SPILL_WARPS,
    SDOT4,
    WAVE32,
    WMMA_GATE_GFX1030,
    ArchSwitch,
    PackedDot,
    arch_switch,
    refuse_dot_on_mad_mix,
    wave_occupancy_ok,
)

__all__ = [
    "ArchSwitch",
    "FDOT2",
    "FORBIDDEN_ISA",
    "LDS_BANK_BYTES",
    "LDS_BANKS",
    "LDS_PAD",
    "MAD_MIX_ARCHES",
    "NO_WMMA_GATE",
    "PACKED_DOT",
    "PackedDot",
    "QSA_OCCUPANCY_THREADS",
    "QSA_OCCUPANCY_WARPS",
    "QSA_SPILL_WARPS",
    "SDOT4",
    "WAVE32",
    "WMMA_GATE_GFX1030",
    "arch_switch",
    "refuse_dot_on_mad_mix",
    "wave_occupancy_ok",
]
