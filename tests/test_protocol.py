from __future__ import annotations

import pytest

from hippihx.attn import fa_fdot2
from hippihx.protocol import (
    DEFAULT_ARCH,
    GFX1030_WAVE,
    NO_D2H_UNDER_CAPTURE,
    NO_TRITON_DOUBLE_FIRE,
    ONE_TORCH_OP_PER_KERNEL,
    ROCM_PIN,
    SCRATCH_FROM_PLAN,
    SCRATCH_ZEROED_FOR_PAGE_COMMIT,
    Caps,
    require_single_arch,
)


def test_engine_bind_rules_are_on() -> None:
    assert ONE_TORCH_OP_PER_KERNEL
    assert NO_TRITON_DOUBLE_FIRE
    assert SCRATCH_FROM_PLAN
    assert SCRATCH_ZEROED_FOR_PAGE_COMMIT
    assert NO_D2H_UNDER_CAPTURE
    assert DEFAULT_ARCH == "gfx1030"
    assert GFX1030_WAVE == 32
    assert ROCM_PIN == "7.14"


def test_plan_bind_run_stub() -> None:
    caps = fa_fdot2.Caps(arch="gfx1030")
    plan = fa_fdot2.plan(caps)
    assert plan.arch == "gfx1030"
    assert plan.scratch_specs()[0].zeroed is True
    binding = fa_fdot2.bind(plan, scratch=None)
    assert fa_fdot2.run(binding) is None
    assert fa_fdot2.is_supported(caps)


def test_bind_rejects_foreign_plan() -> None:
    from hippihx.gemm import w4a16_fdot2

    plan = w4a16_fdot2.plan(Caps())
    with pytest.raises(ValueError, match="does not match"):
        fa_fdot2.bind(plan, scratch=None)


def test_caps_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError, match="unknown arch"):
        Caps(arch="gfx1200")


def test_require_single_arch() -> None:
    assert require_single_arch("gfx1100") == "gfx1100"
    with pytest.raises(ValueError, match="one fatbin"):
        require_single_arch("gfx1030,gfx1100")
