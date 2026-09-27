"""Tiny planned-op helper. Tile ``api.py`` files call ``export(globals())``."""

from __future__ import annotations

from typing import Any

from .backend import MOJO_NOT_PRODUCE_MSG, Backend, MojoNotProduce, is_zoo_backend
from .catalog import OpSpec, find_spec
from .explore import ranked_explore
from .families import hooks_for, require_family
from .fatbin import supported_arches
from .isa import arch_switch
from .protocol import (
    Binding,
    Caps,
    OpMeta,
    Plan,
    dtype_supported,
    stub_bind,
    stub_plan,
    stub_run,
)


class StubOp:
    """Host-side contract stub. Replace ``run`` when the HIP tile lands."""

    META: OpMeta

    def __init__(self, spec: OpSpec) -> None:
        group, name = spec.qualname.split(".", 1)
        self.META = OpMeta(
            qualname=spec.qualname,
            group=group,
            name=name,
            summary=spec.summary,
            dot=spec.dot,
            fp16_act=spec.fp16_act,
            tile=spec.tile,
            b12x_analogue=spec.b12x_analogue,
        )
        self.Caps = Caps

    def plan(self, caps: Caps | None = None) -> Plan:
        caps = caps or Caps()
        if not self.is_supported(caps):
            raise ValueError(
                f"{self.META.qualname} unsupported for "
                f"arch={caps.arch!r} dtype={caps.dtype!r} backend={caps.backend!r}"
            )
        base = stub_plan(self.META.qualname, caps)
        shapes = ranked_explore(self.META.qualname, caps.arch)
        families = hooks_for(self.META.qualname)
        return Plan(
            qualname=base.qualname,
            arch=base.arch,
            specs=base.specs,
            meta={
                "backend": caps.backend,
                "wave": caps.wave,
                "arch_switch": arch_switch(caps.arch).value,
                "explore": tuple(shape.name for shape in shapes),
                "families": tuple(hook.name for hook in families),
                "produce": caps.backend != Backend.MOJO.value,
            },
        )

    def bind(
        self,
        plan: Plan,
        scratch: Any = None,
        *,
        family: str | None = None,
        **tensors: Any,
    ) -> Binding:
        if plan.qualname != self.META.qualname:
            raise ValueError(
                f"plan {plan.qualname!r} does not match {self.META.qualname!r}"
            )
        if family is not None:
            require_family(self.META.qualname, family)
        return stub_bind(plan, scratch=scratch, family=family, **tensors)

    def run(self, binding: Binding) -> None:
        if binding.plan.qualname != self.META.qualname:
            raise ValueError("binding plan does not match this op")
        if binding.plan.meta.get("backend") == Backend.MOJO.value:
            raise MojoNotProduce(MOJO_NOT_PRODUCE_MSG)
        return stub_run(binding)

    def is_supported(self, caps: Caps | None = None) -> bool:
        caps = caps or Caps()
        if not is_zoo_backend(caps.backend):
            return False
        if caps.arch not in supported_arches(self.META.dot):
            return False
        return dtype_supported(
            dot=self.META.dot, fp16_act=self.META.fp16_act, dtype=caps.dtype
        )


def make_op(qualname: str) -> StubOp:
    return StubOp(find_spec(qualname))


def export(ns: dict[str, Any]) -> None:
    """Fill an op module namespace from the catalog (``api.py`` one-liner)."""
    name = ns["__name__"]
    if name.endswith(".api"):
        name = name[: -len(".api")]
    qualname = name.removeprefix("hippihx.")
    op = make_op(qualname)
    ns["META"] = op.META
    ns["Caps"] = Caps
    ns["plan"] = op.plan
    ns["bind"] = op.bind
    ns["run"] = op.run
    ns["is_supported"] = op.is_supported
    ns["FAMILY_HOOKS"] = hooks_for(qualname)
    ns.setdefault(
        "__all__",
        ["META", "Caps", "FAMILY_HOOKS", "plan", "bind", "run", "is_supported"],
    )
