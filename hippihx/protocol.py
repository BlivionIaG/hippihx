"""plan / bind / run protocol (HIP/RDNA).

Layering only — same verbs as the b12x → serve split, no CUDA/CuTe types.

- ``plan`` is host-side and may allocate metadata.
- ``bind`` builds views; it must not allocate tensors.
- ``run`` is graph-capture safe: no device-to-host under capture.

Scratch is sized from the plan and zeroed by the *caller* (serve) so RDNA2
cudagraph page-commit sees committed pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

KNOWN_ARCHES: tuple[str, ...] = ("gfx1030", "gfx1100", "gfx900")
DOT_ARCHES: tuple[str, ...] = ("gfx1030", "gfx1100")
LATER_ARCHES: tuple[str, ...] = ("gfx906",)
DEFAULT_ARCH = "gfx1030"
ROCM_PIN = "7.14"
GFX1030_WAVE = 32
DOT_WAVE = 32
NO_FDOT2_BF16 = True
NO_WMMA_ON_SHARED_DOT = True

# Engine bind rules (room-locked). Serve wiring in rdna_extras must honor these.
ONE_TORCH_OP_PER_KERNEL = True
NO_TRITON_DOUBLE_FIRE = True
SCRATCH_FROM_PLAN = True
SCRATCH_ZEROED_FOR_PAGE_COMMIT = True
NO_D2H_UNDER_CAPTURE = True


@dataclass(frozen=True, slots=True)
class ScratchSpec:
    """Caller-owned buffer the serve layer allocates and zeros."""

    name: str
    nbytes: int
    zeroed: bool = True
    note: str = "zeroed for cudagraph page-commit"


@dataclass(frozen=True, slots=True)
class Caps:
    """Host-side capabilities handed to ``plan``. No device work."""

    arch: str = DEFAULT_ARCH
    wave: int | None = None
    device: str = "hip"

    def __post_init__(self) -> None:
        if self.arch in LATER_ARCHES:
            raise ValueError(
                f"{self.arch} is a Later fatbin slot (Vega variant; "
                "mad_mix / pk_fma — not DOT). Documented, not built yet"
            )
        if self.arch not in KNOWN_ARCHES:
            raise ValueError(
                f"unknown arch {self.arch!r}; built slots {KNOWN_ARCHES}; "
                f"Later slots {LATER_ARCHES}"
            )
        if self.wave is None:
            object.__setattr__(self, "wave", 64 if self.arch == "gfx900" else DOT_WAVE)


@dataclass(frozen=True, slots=True)
class Plan:
    """Launch + scratch policy. Must not own a serve workspace."""

    qualname: str
    arch: str
    specs: tuple[ScratchSpec, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def scratch_specs(self) -> tuple[ScratchSpec, ...]:
        return self.specs


@dataclass(frozen=True, slots=True)
class Binding:
    """Views container. Fresh every call; never allocates."""

    plan: Plan
    scratch: Any = None
    tensors: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OpMeta:
    qualname: str
    group: str
    name: str
    summary: str
    planned: bool = True
    dot: bool = False


@runtime_checkable
class PlannedOp(Protocol):
    META: OpMeta

    def plan(self, caps: Caps) -> Plan: ...

    def bind(self, plan: Plan, scratch: Any = None, **tensors: Any) -> Binding: ...

    def run(self, binding: Binding) -> Any: ...

    def is_supported(self, caps: Caps | None = None) -> bool: ...


def stub_plan(qualname: str, caps: Caps, nbytes: int = 0) -> Plan:
    """Placeholder plan: one scratch slab (may be 0 bytes), always zeroed."""
    specs = (
        ScratchSpec(
            name="workspace",
            nbytes=nbytes,
            zeroed=True,
            note="zeroed for cudagraph page-commit",
        ),
    )
    return Plan(qualname=qualname, arch=caps.arch, specs=specs)


def stub_bind(plan: Plan, scratch: Any = None, **tensors: Any) -> Binding:
    """View-only bind. Refuses to look like an allocator."""
    return Binding(plan=plan, scratch=scratch, tensors=dict(tensors))


def stub_run(binding: Binding) -> None:
    """Skeleton run: no device launch, no D2H."""
    if binding.plan is None:
        raise RuntimeError("run requires a planned binding")
    return None


def require_single_arch(arch: str) -> str:
    if "," in arch or " " in arch:
        raise ValueError("one fatbin slot per artifact; no multi-arch objects")
    if arch in LATER_ARCHES:
        raise ValueError(f"{arch} is a Later fatbin slot — not built yet")
    if arch not in KNOWN_ARCHES:
        raise ValueError(f"unknown fatbin slot {arch!r}")
    return arch


def supported_arches(dot: bool) -> tuple[str, ...]:
    """DOT tiles: gfx1030 + gfx1100 only. Vega slots do not load DOT objects."""
    return DOT_ARCHES if dot else KNOWN_ARCHES
