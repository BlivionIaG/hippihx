from __future__ import annotations

from pathlib import Path

import hippihx


EXPECTED = (
    "attn.fa_fdot2",
    "attn.gdn_scan",
    "attn.kda_scan",
    "attn.qsa_indexer",
    "attn.dsa_nope",
    "gemm.w4a16_fdot2",
    "gemm.exl3_3inst",
    "moe.routed",
    "moe.shared",
    "moe.leftover_bf16",
    "sequence.causal_conv",
    "comm.pcie",
)

DOT_OPS = {
    "attn.fa_fdot2",
    "gemm.w4a16_fdot2",
    "gemm.exl3_3inst",
    "moe.shared",
}


def test_list_ops_matches_contracts() -> None:
    names = tuple(meta.qualname for meta in hippihx.list_ops())
    assert names == EXPECTED


def test_find_op_roundtrip() -> None:
    meta = hippihx.find_op("sequence.causal_conv")
    assert meta.group == "sequence"
    assert meta.name == "causal_conv"
    assert meta.dot is False


def test_dot_flags() -> None:
    for meta in hippihx.list_ops():
        assert meta.dot is (meta.qualname in DOT_OPS)


def test_tiles_dirs_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "tiles"
    mapping = {
        "attn.fa_fdot2": root / "attn" / "fa_fdot2",
        "attn.gdn_scan": root / "attn" / "gdn_scan",
        "attn.kda_scan": root / "attn" / "kda_scan",
        "attn.qsa_indexer": root / "attn" / "qsa_indexer",
        "attn.dsa_nope": root / "attn" / "dsa_nope",
        "gemm.w4a16_fdot2": root / "gemm" / "w4a16_fdot2",
        "gemm.exl3_3inst": root / "gemm" / "exl3_3inst",
        "moe.routed": root / "moe" / "routed",
        "moe.shared": root / "moe" / "shared",
        "moe.leftover_bf16": root / "moe" / "leftover_bf16",
        "sequence.causal_conv": root / "sequence" / "causal_conv",
        "comm.pcie": root / "comm" / "pcie",
    }
    for qualname, path in mapping.items():
        assert (path / "README.md").is_file(), qualname
        assert (path / "kernel.hip").is_file(), qualname


def test_dot_sources_include_lock_header() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "tiles/attn/fa_fdot2/kernel.hip",
        "tiles/gemm/w4a16_fdot2/kernel.hip",
        "tiles/gemm/exl3_3inst/kernel.hip",
        "tiles/moe/shared/kernel.hip",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert '#include "hippihx/dot.hpp"' in text
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") and "WMMA" in stripped:
                raise AssertionError(f"{rel} has a WMMA preprocessor gate: {line}")


def test_causal_conv_not_under_gdn() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "tiles" / "attn" / "gdn_scan" / "causal_conv").exists()
    assert (root / "tiles" / "sequence" / "causal_conv" / "README.md").is_file()


def test_no_produce_dirs() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ["produce", "awq", "3inst"]
    top = {p.name for p in root.iterdir()}
    for name in forbidden:
        assert name not in top
