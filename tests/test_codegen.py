"""Generated blocks follow the catalog. Host-only, no compiler."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hippihx._lib import codegen
from hippihx._lib.catalog import OPS
from hippihx._lib.fatbin import KNOWN_ARCHES, LATER_ARCHES

ROOT = Path(__file__).resolve().parents[1]


def test_generated_blocks_are_current() -> None:
    # Fix: python -m hippihx._lib.codegen
    assert codegen.stale() == []


def test_each_block_has_one_marker_pair() -> None:
    for block in codegen.BLOCKS:
        text = (ROOT / block.path).read_text(encoding="utf-8")
        assert text.count(codegen.marker(block, "begin")) == 1, block.name
        assert text.count(codegen.marker(block, "end")) == 1, block.name


def test_splice_rewrites_only_the_marked_lines() -> None:
    text = (
        "keep\n"
        "// hippihx:gen begin demo -- hint\n"
        "old\n"
        "// hippihx:gen end demo\n"
        "tail\n"
    )
    out = codegen.splice(text, "demo", "new 1\nnew 2\n")
    assert out == (
        "keep\n"
        "// hippihx:gen begin demo -- hint\n"
        "new 1\nnew 2\n"
        "// hippihx:gen end demo\n"
        "tail\n"
    )
    assert codegen.splice(out, "demo", "new 1\nnew 2\n") == out
    with pytest.raises(ValueError, match="exactly one"):
        codegen.splice("nothing here\n", "demo", "x\n")
    with pytest.raises(ValueError, match="exactly one"):
        codegen.splice(text + text, "demo", "x\n")
    with pytest.raises(ValueError, match="before"):
        codegen.splice(
            "// hippihx:gen end demo\n// hippihx:gen begin demo -- hint\n",
            "demo",
            "x\n",
        )


def test_v1_rows_and_slots_follow_the_catalog() -> None:
    abi = (ROOT / "tiles" / "v1_abi.cpp").read_text(encoding="utf-8")
    names = [line.split('"')[1] for line in abi.splitlines() if line.startswith('    {"')]
    assert names[: len(OPS)] == [spec.qualname for spec in OPS]
    assert names[len(OPS) :] == list(KNOWN_ARCHES)
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for spec in OPS:
        assert f"  tiles/{spec.tile}/kernel.hip\n" in cmake, spec.qualname


@pytest.mark.parametrize("arch", LATER_ARCHES)
def test_build_script_refuses_later_slots(arch: str, tmp_path: Path) -> None:
    env = dict(os.environ, HIPPIHX_ARCH=arch, BUILD_DIR=str(tmp_path / "build"))
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "build_fatbin.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert f"{arch}: Later" in proc.stderr
    assert not (tmp_path / "build").exists()  # refused before cmake runs


def test_build_script_refuses_unknown_arch(tmp_path: Path) -> None:
    env = dict(os.environ, HIPPIHX_ARCH="gfx1200", BUILD_DIR=str(tmp_path / "build"))
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "build_fatbin.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "unknown HIPPIHX_ARCH='gfx1200'" in proc.stderr
