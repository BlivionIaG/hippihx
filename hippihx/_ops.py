"""Tiny planned-op helper so tile modules stay one-liners."""

from __future__ import annotations

from typing import Any

from .protocol import (
    Binding,
    Caps,
    OpMeta,
    Plan,
    dtype_supported,
    stub_bind,
    stub_plan,
    stub_run,
    supported_arches,
)


class StubOp:
    """Host-side contract stub. Replace ``run`` when the HIP tile lands."""

    META: OpMeta

    def __init__(self, meta: OpMeta) -> None:
        self.META = meta
        self.Caps = Caps

    def plan(self, caps: Caps | None = None) -> Plan:
        caps = caps or Caps()
        if not self.is_supported(caps):
            raise ValueError(
                f"{self.META.qualname} unsupported for "
                f"arch={caps.arch!r} dtype={caps.dtype!r}"
            )
        return stub_plan(self.META.qualname, caps)

    def bind(self, plan: Plan, scratch: Any = None, **tensors: Any) -> Binding:
        if plan.qualname != self.META.qualname:
            raise ValueError(
                f"plan {plan.qualname!r} does not match {self.META.qualname!r}"
            )
        return stub_bind(plan, scratch=scratch, **tensors)

    def run(self, binding: Binding) -> None:
        if binding.plan.qualname != self.META.qualname:
            raise ValueError("binding plan does not match this op")
        return stub_run(binding)

    def is_supported(self, caps: Caps | None = None) -> bool:
        caps = caps or Caps()
        if caps.arch not in supported_arches(self.META.dot):
            return False
        return dtype_supported(
            dot=self.META.dot, fp16_act=self.META.fp16_act, dtype=caps.dtype
        )


def make_op(
    qualname: str,
    summary: str,
    *,
    dot: bool = False,
    fp16_act: bool | None = None,
) -> StubOp:
    group, name = qualname.split(".", 1)
    if fp16_act is None:
        fp16_act = dot
    return StubOp(
        OpMeta(
            qualname=qualname,
            group=group,
            name=name,
            summary=summary,
            dot=dot,
            fp16_act=fp16_act,
        )
    )
