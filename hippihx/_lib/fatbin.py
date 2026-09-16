"""Fatbin slots. One --offload-arch per artifact. Never HSA_OVERRIDE."""

from __future__ import annotations

KNOWN_ARCHES: tuple[str, ...] = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1151",
    "gfx1031",
    "gfx1032",
    "gfx1033",
    "gfx1035",
    "gfx1036",
    "gfx900",
)
DOT_ARCHES: tuple[str, ...] = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1151",
    "gfx1031",
    "gfx1032",
    "gfx1033",
    "gfx1035",
    "gfx1036",
)
UNOPTIMIZED_DOT_ARCHES: tuple[str, ...] = (
    "gfx1151",
    "gfx1031",
    "gfx1032",
    "gfx1033",
    "gfx1035",
    "gfx1036",
)
VERIFY_WAVE_ARCHES: tuple[str, ...] = ()
GFX103X_WAVE = 32
LATER_NONDOT_ARCHES: tuple[str, ...] = ("gfx906",)
LATER_ARCHES: tuple[str, ...] = ("gfx906", "gfx1013")
LATER_ARCH_NOTES: dict[str, str] = {
    "gfx906": (
        "Later non-DOT (real Vega20/MI50). Not BC-250 — BC-250 is gfx1013 "
        "(Cyan Skillfish, also Later; not true RDNA2). Never load FA/EXL3 DOT "
        "objects"
    ),
    "gfx1013": (
        "Later. BC-250 / Cyan Skillfish is not true RDNA2 — not dest, not a "
        "portable DOT fatbin with gfx1030. Never HSA_OVERRIDE a gfx1030/Deck "
        "object onto it. Not gfx906/Vega20"
    ),
}
DEFAULT_ARCH = "gfx1030"
ROCM_PIN = "7.14"
GFX1030_WAVE = 32
DOT_WAVE = 32
NO_FDOT2_BF16 = True
NO_WMMA_ON_SHARED_DOT = True
NO_HSA_OVERRIDE = True
NO_FOREIGN_ISA_LOAD = True
BIND_KEYS_ON_ARCH_AND_WAVE = True

ONE_TORCH_OP_PER_KERNEL = True
NO_TRITON_DOUBLE_FIRE = True
SCRATCH_FROM_PLAN = True
SCRATCH_ZEROED_FOR_PAGE_COMMIT = True
NO_D2H_UNDER_CAPTURE = True
FP16_ACT_ONLY_ON_DOT = True
FP16_ACT_ONLY_ON_GDN = True


def later_note(arch: str) -> str:
    return LATER_ARCH_NOTES.get(arch, "Later fatbin slot — not built yet")


def require_single_arch(arch: str) -> str:
    if "," in arch or " " in arch:
        raise ValueError("one fatbin slot per artifact; no multi-arch objects")
    if arch in LATER_ARCHES:
        raise ValueError(f"{arch} is a Later fatbin slot: {later_note(arch)}")
    if arch not in KNOWN_ARCHES:
        raise ValueError(f"unknown fatbin slot {arch!r}")
    return arch


def supported_arches(dot: bool) -> tuple[str, ...]:
    return DOT_ARCHES if dot else KNOWN_ARCHES
