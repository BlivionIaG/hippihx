from __future__ import annotations

import pytest

from hippihx.attn import fa_fdot2
from hippihx.protocol import (
    BIND_KEYS_ON_ARCH_AND_WAVE,
    DEFAULT_ARCH,
    DOT_ARCHES,
    DOT_WAVE,
    GFX1030_WAVE,
    LATER_ARCHES,
    LATER_NONDOT_ARCHES,
    LATER_VERIFY_DOT_ARCHES,
    NO_D2H_UNDER_CAPTURE,
    NO_FDOT2_BF16,
    NO_FOREIGN_ISA_LOAD,
    NO_HSA_OVERRIDE,
    NO_TRITON_DOUBLE_FIRE,
    NO_WMMA_ON_SHARED_DOT,
    ONE_TORCH_OP_PER_KERNEL,
    ROCM_PIN,
    SCRATCH_FROM_PLAN,
    SCRATCH_ZEROED_FOR_PAGE_COMMIT,
    UNOPTIMIZED_DOT_ARCHES,
    Caps,
    require_single_arch,
)
from hippihx.sequence import causal_conv


def test_engine_bind_rules_are_on() -> None:
    assert ONE_TORCH_OP_PER_KERNEL
    assert NO_TRITON_DOUBLE_FIRE
    assert SCRATCH_FROM_PLAN
    assert SCRATCH_ZEROED_FOR_PAGE_COMMIT
    assert NO_D2H_UNDER_CAPTURE
    assert NO_FDOT2_BF16
    assert NO_WMMA_ON_SHARED_DOT
    assert NO_HSA_OVERRIDE
    assert NO_FOREIGN_ISA_LOAD
    assert BIND_KEYS_ON_ARCH_AND_WAVE
    assert DEFAULT_ARCH == "gfx1030"
    assert GFX1030_WAVE == 32
    assert DOT_WAVE == 32
    assert "gfx1151" in DOT_ARCHES
    assert "gfx1013" not in DOT_ARCHES
    assert "gfx1035" in DOT_ARCHES
    assert UNOPTIMIZED_DOT_ARCHES == (
        "gfx1151",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1035",
        "gfx1036",
    )
    assert LATER_VERIFY_DOT_ARCHES == ("gfx1013",)
    assert LATER_NONDOT_ARCHES == ("gfx906",)
    assert LATER_ARCHES == ("gfx1013", "gfx906")
    assert ROCM_PIN == "7.14"


def test_plan_bind_run_stub() -> None:
    caps = fa_fdot2.Caps(arch="gfx1100")
    plan = fa_fdot2.plan(caps)
    assert plan.arch == "gfx1100"
    assert plan.scratch_specs()[0].zeroed is True
    binding = fa_fdot2.bind(plan, scratch=None)
    assert fa_fdot2.run(binding) is None
    assert fa_fdot2.is_supported(caps)
    assert fa_fdot2.is_supported(Caps(arch="gfx1030"))
    assert fa_fdot2.is_supported(Caps(arch="gfx1102"))
    assert fa_fdot2.is_supported(Caps(arch="gfx1151"))
    assert fa_fdot2.is_supported(Caps(arch="gfx1035"))
    assert not fa_fdot2.is_supported(Caps(arch="gfx900"))


def test_causal_conv_protocol() -> None:
    caps = causal_conv.Caps(arch="gfx1030")
    plan = causal_conv.plan(caps)
    assert plan.qualname == "sequence.causal_conv"
    assert plan.scratch_specs()[0].zeroed is True
    assert causal_conv.run(causal_conv.bind(plan)) is None
    assert causal_conv.is_supported(Caps(arch="gfx900"))


def test_bind_rejects_foreign_plan() -> None:
    from hippihx.gemm import w4a16_fdot2

    plan = w4a16_fdot2.plan(Caps())
    with pytest.raises(ValueError, match="does not match"):
        fa_fdot2.bind(plan, scratch=None)


def test_caps_rejects_unknown_and_later() -> None:
    with pytest.raises(ValueError, match="unknown arch"):
        Caps(arch="gfx1200")
    with pytest.raises(ValueError, match="Vega20/MI50") as gfx906:
        Caps(arch="gfx906")
    assert "not BC-250" in str(gfx906.value).lower() or "Not BC-250" in str(
        gfx906.value
    )
    with pytest.raises(ValueError, match="Later VERIFY") as gfx1013:
        Caps(arch="gfx1013")
    note = str(gfx1013.value).lower()
    assert "warp size 64" in note
    assert "arch + wave" in note or "arch + wave size" in note


def test_require_single_arch() -> None:
    assert require_single_arch("gfx1101") == "gfx1101"
    assert require_single_arch("gfx1151") == "gfx1151"
    with pytest.raises(ValueError, match="one fatbin"):
        require_single_arch("gfx1030,gfx1100")
    with pytest.raises(ValueError, match="Later"):
        require_single_arch("gfx906")
    with pytest.raises(ValueError, match="Later VERIFY"):
        require_single_arch("gfx1013")
