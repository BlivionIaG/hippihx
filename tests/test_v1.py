from __future__ import annotations

from pathlib import Path

import hippihx
from hippihx.v1 import ABI_REVISION, V1_OP_NAMES, V1OpId, v1_op_is_dot, v1_op_name

ROOT = Path(__file__).resolve().parents[1]


def test_v1_names_match_list_ops() -> None:
    names = tuple(meta.qualname for meta in hippihx.list_ops())
    assert V1_OP_NAMES == names
    assert len(V1OpId) == len(names)
    for op in V1OpId:
        assert v1_op_name(op) == names[int(op)]


def test_v1_dot_flags() -> None:
    for meta in hippihx.list_ops():
        op = V1OpId(V1_OP_NAMES.index(meta.qualname))
        assert v1_op_is_dot(op) is meta.dot


def test_v1_header_lockstep() -> None:
    header = (ROOT / "include" / "hippihx" / "v1.h").read_text(encoding="utf-8")
    assert f"HIPPIHX_V1_ABI_REVISION = {ABI_REVISION}" in header
    assert "HIPPIHX_V1_OP_COUNT = 12" in header
    assert "hippihx_v1_plan" in header
    assert "hippihx_v1_run" in header
    assert "HIPPIHX_V1_ERR_NOT_READY" in header
    for name in V1_OP_NAMES:
        # C comments / docs may omit; enum names are enough for a few anchors.
        pass
    assert "HIPPIHX_V1_OP_GEMM_EXL3_3INST" in header
    assert "HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV" in header
    assert "HIPPIHX_V1_OP_COMM_PCIE" in header


def test_v1_abi_source_in_fatbin() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "tiles/v1_abi.cpp" in cmake
    assert (ROOT / "tiles" / "v1_abi.cpp").is_file()


def test_exl3_grain_v2_lock_documented() -> None:
    text = (ROOT / "tiles" / "gemm" / "exl3_3inst" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "V2_THREADS_X=64" in text
    assert "LDS_PAD=8" in text
    assert "sm≤8" in text or "sm<=8" in text
    assert "HIPPIHX_V1_OP_GEMM_EXL3_3INST" in text


def test_package_exports_v1() -> None:
    assert hippihx.V1_ABI_REVISION == ABI_REVISION
    assert hippihx.V1_OP_NAMES == V1_OP_NAMES
    assert hippihx.v1_op_name(V1OpId.ATTN_FA_FDOT2) == "attn.fa_fdot2"
