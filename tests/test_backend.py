from __future__ import annotations

from pathlib import Path

import pytest

from hippihx._lib.backend import (
    DEST_BACKEND,
    FATBIN_BACKEND,
    FLYDSL_DEST,
    FLYDSL_GATE0_OBJECT,
    FLYDSL_V1_CONSUME,
    FORBIDDEN_FLYDSL_KERNELS,
    Backend,
    flydsl_ready,
    is_zoo_backend,
)
from hippihx.flydsl import FDOT2, SDOT4, VEC_ADD, available, refuse_shipped_kernel
from hippihx.gemm.w4a16_fdot2.pack import (
    AWQ_ZERO_OFFSET,
    CONFIG_A_K_STEP,
    GPTQ_ZERO_OFFSET,
    K_STEP,
    dequant_nibble,
    k_per_split,
    refuse_config_h,
    zero_offset,
)
from hippihx.protocol import Caps


def test_hip_and_flydsl_are_zoo_backends() -> None:
    assert DEST_BACKEND is Backend.HIP
    assert FATBIN_BACKEND is Backend.HIP
    assert FLYDSL_DEST is True
    assert FLYDSL_V1_CONSUME is False
    assert FLYDSL_GATE0_OBJECT is False
    assert flydsl_ready() is False
    assert is_zoo_backend("hip")
    assert is_zoo_backend("flydsl")
    Caps(arch="gfx1030", backend="hip")
    Caps(arch="gfx1030", backend="flydsl")
    with pytest.raises(ValueError, match="unknown backend"):
        Caps(arch="gfx1030", backend="cute")


def test_flydsl_plan_bind_run_stub() -> None:
    from hippihx.attention import fa_fdot2

    caps = Caps(arch="gfx1030", backend="flydsl")
    assert fa_fdot2.is_supported(caps)
    plan = fa_fdot2.plan(caps)
    assert plan.qualname == "attention.fa_fdot2"
    assert fa_fdot2.run(fa_fdot2.bind(plan)) is None


def test_flydsl_optional_and_atoms() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "dependencies = []" in pyproject
    assert 'flydsl = ["flydsl"]' in pyproject
    assert available() is False
    assert FDOT2.llvm == "llvm.amdgcn.fdot2"
    assert "v_dot2_f32_f16" in FDOT2.isa
    assert SDOT4.llvm == "llvm.amdgcn.sdot4"
    assert VEC_ADD.arch == "gfx1030"
    assert VEC_ADD.wave == 32
    fly = (Path(__file__).resolve().parents[1] / "docs" / "FLYDSL.md").read_text(
        encoding="utf-8"
    )
    assert "dest compiler backend" in fly.lower()
    assert "research" not in fly.lower()
    assert "MFMA" in fly
    for name in FORBIDDEN_FLYDSL_KERNELS:
        with pytest.raises(ValueError, match="MFMA/WMMA"):
            refuse_shipped_kernel(name)


def test_w4_integer_zp_and_k_step() -> None:
    assert zero_offset(use_v2_format=False) == GPTQ_ZERO_OFFSET == 1
    assert zero_offset(use_v2_format=True) == AWQ_ZERO_OFFSET == 0
    assert dequant_nibble(0, 0, 1.5) == 0.0
    assert dequant_nibble(7, 1, 0.5) == 3.0
    assert k_per_split(4096, 8) == 512
    assert k_per_split(4096, 8) % K_STEP == 0
    with pytest.raises(ValueError, match="K_STEP"):
        k_per_split(640, 16)  # dest 40-wide split
    refuse_config_h(CONFIG_A_K_STEP)
    with pytest.raises(ValueError, match="ConfigH"):
        refuse_config_h(64)
