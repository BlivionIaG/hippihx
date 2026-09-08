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
    "comm.pcie",
)


def test_list_ops_matches_contracts() -> None:
    names = tuple(meta.qualname for meta in hippihx.list_ops())
    assert names == EXPECTED


def test_find_op_roundtrip() -> None:
    meta = hippihx.find_op("attn.fa_fdot2")
    assert meta.group == "attn"
    assert meta.name == "fa_fdot2"


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
        "comm.pcie": root / "comm" / "pcie",
    }
    for qualname, path in mapping.items():
        assert (path / "README.md").is_file(), qualname
        assert (path / "kernel.hip").is_file(), qualname


def test_no_produce_dirs() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ["produce", "awq", "3inst"]
    top = {p.name for p in root.iterdir()}
    for name in forbidden:
        assert name not in top
