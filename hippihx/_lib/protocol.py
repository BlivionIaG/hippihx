"""plan / bind / run protocol (HIP/RDNA).

Same verbs as b12x → serve. No CUDA/CuTe types. Backends: HIP fatbins and
the FlyDSL compiler.

- ``plan`` is host-side and may allocate metadata. With params it is
  sized: it mirrors ``hippihx_v1_plan`` (V1 rev 4) and refuses the same
  inputs. Without params it is a contract plan (no scratch, not sized).
- ``bind`` builds views; it must not allocate tensors.
- ``run`` is graph-capture safe: no device-to-host under capture.

Scratch is sized from the plan and zeroed by the *caller* (serve).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .backend import Backend, is_zoo_backend
from .fabric import Fabric
from .fatbin import (
    DEFAULT_ARCH,
    KNOWN_ARCHES,
    LATER_ARCH_NOTES,
    LATER_ARCHES,
    default_wave,
)

KNOWN_DTYPES: tuple[str, ...] = ("fp16", "bf16", "fp32")
DTYPE_UNSET = 0
DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_FP32 = 3


def dtype_to_v1(dtype: str | None) -> int:
    if dtype is None:
        return DTYPE_UNSET
    mapping = {"fp16": DTYPE_FP16, "bf16": DTYPE_BF16, "fp32": DTYPE_FP32}
    if dtype not in mapping:
        raise ValueError(f"unknown dtype {dtype!r}; known {KNOWN_DTYPES}")
    return mapping[dtype]


@dataclass(frozen=True, slots=True)
class ScratchSpec:
    """Caller-owned buffer the serve layer allocates and zeros.

    ``offset`` is from the plan's scratch base (V1 rev 4 packs every spec
    of one plan into one aligned block).
    """

    name: str
    nbytes: int
    zeroed: bool = True
    note: str = "zeroed for cudagraph page-commit"
    offset: int = 0


@dataclass(frozen=True, slots=True)
class Caps:
    """Host-side capabilities handed to ``plan``. No device work.

    Serve bind keys on ``arch`` **and** ``wave``. Comm also keys on
    ``fabric.hop`` + ``fabric.switch`` (PIX vs PHB/PXB). ``backend`` is
    ``hip`` (fatbin / extras V1), ``flydsl`` (compiler), or ``mojo``
    (MAX authoring; not produce). Default is HIP.
    """

    arch: str = DEFAULT_ARCH
    wave: int | None = None
    device: str = "hip"
    dtype: str | None = None
    backend: str = "hip"
    fabric: Fabric | None = None

    def __post_init__(self) -> None:
        if self.arch in LATER_ARCHES:
            note = LATER_ARCH_NOTES.get(self.arch, "Later fatbin slot — not built")
            raise ValueError(f"{self.arch} is a Later fatbin slot: {note}")
        if self.arch not in KNOWN_ARCHES:
            raise ValueError(
                f"unknown arch {self.arch!r}; built slots {KNOWN_ARCHES}; "
                f"Later slots {LATER_ARCHES}"
            )
        if self.dtype is not None and self.dtype not in KNOWN_DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}; known {KNOWN_DTYPES}")
        try:
            backend = Backend(self.backend)
        except ValueError as exc:
            raise ValueError(
                f"unknown backend {self.backend!r}; known hip, flydsl, mojo"
            ) from exc
        if not is_zoo_backend(backend):
            raise ValueError(f"backend {self.backend!r} is not a hippihx zoo backend")
        if self.fabric is not None and not isinstance(self.fabric, Fabric):
            raise ValueError("caps.fabric must be hippihx.comm.fabric.Fabric or None")
        if self.wave is None:
            object.__setattr__(self, "wave", default_wave(self.arch))


@dataclass(frozen=True, slots=True)
class Plan:
    """Launch + scratch policy. Must not own a serve workspace.

    ``params`` holds the checked plan params (empty on a contract plan).
    ``ready`` is the V1 ready bit: False means serve keeps its fallback,
    decided before capture. ``variant`` is the explore rank plan picked.
    ``scratch_nbytes`` is the block to allocate for ``specs``.
    """

    qualname: str
    arch: str
    specs: tuple[ScratchSpec, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    ready: bool = False
    variant: int = 0
    scratch_nbytes: int = 0

    def scratch_specs(self) -> tuple[ScratchSpec, ...]:
        return self.specs


@dataclass(frozen=True, slots=True)
class Binding:
    """Views container. Fresh every call; never allocates.

    ``family`` is an optional bind hook (``qwen.qsa``, ``qwen.gdn``,
    ``qwen.ple``, ``moe.routed``, ``moe.leftover_bf16``, ``hybrid.heap``).
    It names a contract. It does not allocate a heap.
    """

    plan: Plan
    scratch: Any = None
    tensors: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None


@dataclass(frozen=True, slots=True)
class OpMeta:
    qualname: str
    group: str
    name: str
    summary: str
    planned: bool = True
    dot: bool = False
    fp16_act: bool = False
    tile: str = ""
    b12x_analogue: str = ""


@runtime_checkable
class PlannedOp(Protocol):
    META: OpMeta

    def plan(self, caps: Caps | None = None, **params: Any) -> Plan: ...

    def bind(
        self,
        plan: Plan,
        scratch: Any = None,
        *,
        family: str | None = None,
        **tensors: Any,
    ) -> Binding: ...

    def run(self, binding: Binding) -> Any: ...

    def is_supported(self, caps: Caps | None = None) -> bool: ...


def stub_plan(qualname: str, caps: Caps, nbytes: int = 0) -> Plan:
    """Contract plan. A ``nbytes`` workspace when > 0, else no scratch."""

    specs: tuple[ScratchSpec, ...] = ()
    if nbytes > 0:
        specs = (ScratchSpec(name="workspace", nbytes=nbytes, zeroed=True),)
    return Plan(qualname=qualname, arch=caps.arch, specs=specs, scratch_nbytes=nbytes)


def stub_bind(
    plan: Plan,
    scratch: Any = None,
    *,
    family: str | None = None,
    **tensors: Any,
) -> Binding:
    return Binding(plan=plan, scratch=scratch, tensors=dict(tensors), family=family)


def stub_run(binding: Binding) -> None:
    if binding.plan is None:
        raise RuntimeError("run requires a planned binding")
    return None


def dtype_supported(*, dot: bool, fp16_act: bool, dtype: str | None) -> bool:
    if dtype is None or dtype == "fp16":
        return True
    if dtype not in KNOWN_DTYPES:
        return False
    if fp16_act or dot:
        return False
    return True
