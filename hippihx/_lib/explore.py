"""Ranked explore shapes. Host metadata for plan, not a benchmark.

Rank 0 is the first candidate. A ``spill`` row is recorded so a later
search does not promote it. Numbers are tile-README observations. No tok/s.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import find_spec
from .fatbin import default_wave
from .isa import WAVE32, ArchSwitch, arch_switch


@dataclass(frozen=True, slots=True)
class ExploreShape:
    qualname: str
    name: str
    rank: int
    arch_switch: str
    wave: int
    note: str
    threads: int | None = None
    lds_bytes: int | None = None
    spill: bool = False
    pinned: bool = True


def _shape(
    qualname: str,
    name: str,
    rank: int,
    note: str,
    *,
    threads: int | None = None,
    lds_bytes: int | None = None,
    spill: bool = False,
    pinned: bool = True,
    switch: str = ArchSwitch.DOT.value,
    wave: int = WAVE32,
) -> ExploreShape:
    return ExploreShape(
        qualname=qualname,
        name=name,
        rank=rank,
        arch_switch=switch,
        wave=wave,
        note=note,
        threads=threads,
        lds_bytes=lds_bytes,
        spill=spill,
        pinned=pinned,
    )


# Pinned rows follow tiles/*/README.md. Portable DOT arches share the
# gfx1030 observation and are not dest-tuned.
_PINNED: tuple[ExploreShape, ...] = (
    _shape(
        "attention.fa_fdot2",
        "decode",
        0,
        "gfx1030 observation: Br=1, Bc=64, head_dim=128, THREADS=128, "
        "extras smem ~33 KiB. Occupancy pin closed. Not dest-locked.",
        threads=128,
    ),
    _shape(
        "attention.fa_fdot2",
        "prefill",
        1,
        "gfx1030 observation: Br=16, Bc=64, head_dim=128, THREADS=128, "
        "extras smem ~48 KiB (64 KiB/CU class). Leftover launch hygiene (N,1).",
        threads=128,
    ),
    _shape(
        "attention.qsa_indexer",
        "four_warp",
        0,
        "gfx1030 occupancy gate: 4 warps / 128 threads on 6h×256 prefill. "
        "Keep this shape. Not a tok/s claim.",
        threads=128,
    ),
    _shape(
        "attention.qsa_indexer",
        "two_warp",
        1,
        "2-warp profile spills on V620. Recorded so explore does not promote it.",
        threads=64,
        spill=True,
    ),
    _shape(
        "attention.gdn_scan",
        "decode",
        0,
        "extras decode: LDS 0, GDN_THREADS=256, GDN_BV=32, GDN_K=128, "
        "fp16 activations. __launch_bounds__(256).",
        threads=256,
        lds_bytes=0,
    ),
    _shape(
        "attention.gdn_scan",
        "prefill_o",
        1,
        "extras prefill o LDS 45312 B. Not a product fuse with decode.",
        lds_bytes=45312,
    ),
    _shape(
        "gemm.w4a16_fdot2",
        "config_c",
        0,
        "Default W4 prefill branch. K_STEP=32. launch_bounds threads 128. "
        "One GEMM family. ConfigH is not a candidate.",
        threads=128,
    ),
    _shape(
        "gemm.w4a16_fdot2",
        "config_a",
        1,
        "M>256 and N>=4096: M_TILE=16, N_TILE=1024, THREADS=256, K_STEP=32. "
        "LDS_PAD=8. ConfigH (K_STEP=64) stays refused.",
        threads=256,
    ),
    _shape(
        "sequence.causal_conv",
        "state4",
        0,
        "Scalar FMA, LDS 0, 32 threads per dim-block, state_len 4. "
        "fp32 mul. Not fdot2.",
        threads=32,
        lds_bytes=0,
    ),
    _shape(
        "sequence.causal_conv",
        "state3",
        1,
        "Same conv tile with state_len 3 (width-1). Still scalar FMA, LDS 0.",
        threads=32,
        lds_bytes=0,
    ),
    _shape(
        "moe.routed",
        "w4_family",
        0,
        "Routed launch around the one W4 family. A-tile LDS_PAD=8. "
        "expert_path=routed. Not leftover BF16.",
    ),
    _shape(
        "moe.leftover_bf16",
        "cvt_or_scalar",
        0,
        "bf16 caps accepted. GEMM leftovers cvt to fp16 then fdot2; else "
        "scalar FMA. Never fdot2.bf16. expert_path=leftover_bf16.",
    ),
)


def ranked_explore(qualname: str, arch: str) -> tuple[ExploreShape, ...]:
    """Shapes for ``qualname`` on ``arch``, lowest rank first.

    DOT ops on a mad_mix arch return an empty tuple: there is no explore
    candidate because the object must not ship.
    """

    spec = find_spec(qualname)
    switch = arch_switch(arch)
    if spec.dot and switch is ArchSwitch.MAD_MIX:
        return ()
    pinned = tuple(
        shape
        for shape in _PINNED
        if shape.qualname == qualname and shape.arch_switch == switch.value
    )
    if pinned:
        return tuple(sorted(pinned, key=lambda shape: shape.rank))
    wave = default_wave(arch)
    return (
        ExploreShape(
            qualname=qualname,
            name="unpinned",
            rank=0,
            arch_switch=switch.value,
            wave=wave,
            note="No README pin yet. Rank 0 placeholder. Not a measured shape.",
            pinned=False,
        ),
    )


def preferred_explore(qualname: str, arch: str) -> tuple[ExploreShape, ...]:
    """Ranked shapes with spill rows removed."""

    return tuple(shape for shape in ranked_explore(qualname, arch) if not shape.spill)
