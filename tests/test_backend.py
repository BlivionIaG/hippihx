from __future__ import annotations

from pathlib import Path

import pytest

from hippihx._lib.backend import (
    DEST_BACKEND,
    FLYDSL_DEST,
    FLYDSL_GATE0_OBJECT,
    Backend,
    flydsl_ready,
)
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


def test_hip_is_dest_flydsl_is_not() -> None:
    assert DEST_BACKEND is Backend.HIP
    assert FLYDSL_DEST is False
    assert FLYDSL_GATE0_OBJECT is False
    assert flydsl_ready() is False
    Caps(arch="gfx1030", backend="hip")
    with pytest.raises(ValueError, match="FlyDSL is research"):
        Caps(arch="gfx1030", backend="flydsl")
    with pytest.raises(ValueError, match="unknown backend"):
        Caps(arch="gfx1030", backend="cute")


def test_no_flydsl_dependency() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "dependencies = []" in pyproject
    fly = (Path(__file__).resolve().parents[1] / "docs" / "FLYDSL.md").read_text(
        encoding="utf-8"
    )
    assert "not dest" in fly.lower()
    assert "MFMA" in fly


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
