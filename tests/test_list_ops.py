from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import hippihx
from hippihx._lib.catalog import OPS


EXPECTED = tuple(spec.qualname for spec in OPS)

DOT_OPS = {spec.qualname for spec in OPS if spec.dot}
FP16_ACT_OPS = {spec.qualname for spec in OPS if spec.fp16_act}


def test_catalog_ids_are_dense() -> None:
    ids = [spec.v1_id for spec in OPS]
    assert ids == list(range(len(OPS)))
    enums = [spec.enum for spec in OPS]
    assert len(enums) == len(set(enums))


def test_no_legacy_attn_package() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "hippihx" / "attn").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hippihx.attn")


def test_list_ops_matches_catalog() -> None:
    names = tuple(meta.qualname for meta in hippihx.list_ops())
    assert names == EXPECTED
    assert "attn.fa_fdot2" not in names
    assert "attention.fa_fdot2" in names


def test_find_op_roundtrip() -> None:
    meta = hippihx.find_op("sequence.causal_conv")
    assert meta.group == "sequence"
    assert meta.name == "causal_conv"
    assert meta.dot is False


def test_dot_flags() -> None:
    for meta in hippihx.list_ops():
        assert meta.dot is (meta.qualname in DOT_OPS)
        assert meta.fp16_act is (meta.qualname in FP16_ACT_OPS)


def test_tiles_dirs_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "tiles"
    assert not (root / "attn").exists()
    for spec in OPS:
        path = root / spec.tile
        assert (path / "README.md").is_file(), spec.qualname
        assert (path / "kernel.hip").is_file(), spec.qualname
        text = (path / "kernel.hip").read_text(encoding="utf-8")
        assert spec.stub_symbol in text, spec.qualname


def test_dot_sources_include_lock_header() -> None:
    root = Path(__file__).resolve().parents[1]
    for spec in OPS:
        if not spec.dot:
            continue
        rel = f"tiles/{spec.tile}/kernel.hip"
        text = (root / rel).read_text(encoding="utf-8")
        assert '#include "hippihx/dot.hpp"' in text
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") and "WMMA" in stripped:
                raise AssertionError(f"{rel} has a WMMA preprocessor gate: {line}")


def test_causal_conv_not_under_gdn() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "tiles" / "attention" / "gdn_scan" / "causal_conv").exists()
    assert (root / "tiles" / "sequence" / "causal_conv" / "README.md").is_file()


def test_no_produce_dirs() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ["produce", "awq", "3inst"]
    top = {p.name for p in root.iterdir()}
    for name in forbidden:
        assert name not in top


def test_b12x_analogues_are_maps_not_aliases() -> None:
    names = {meta.qualname for meta in hippihx.list_ops()}
    analogues = {meta.b12x_analogue for meta in hippihx.list_ops()}
    assert "attention.paged" in analogues
    assert "attention.paged" not in names
