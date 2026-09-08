from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = {".py", ".hip", ".cu", ".cuh", ".hpp", ".h", ".cpp", ".cc"}

IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(cute|cutlass|b12x)\b"
    r"|^\s*#include\s*[<\"](?:cute/|cutlass/|cuda_runtime|mma\.h|hip_wmma)",
    re.MULTILINE,
)


def test_no_b12x_or_cuda_imports() -> None:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if path.suffix not in CODE or not path.is_file():
            continue
        if any(part in {".git", "build", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if IMPORT_RE.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_tiles_are_torch_aten_free() -> None:
    """extras HIP is ATen-wrapped. A body dump is not a migrate."""
    tiles = ROOT / "tiles"
    hits: list[str] = []
    forbidden = ("torch/all.h", "ATen/", "c10/cuda/")
    for path in tiles.rglob("*"):
        if path.suffix not in {".hip", ".cu", ".cuh", ".hpp", ".h", ".cpp"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in forbidden):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_foreign_hip_attribution_policy() -> None:
    """leapdragon / a17t keep Author+Committer. Never steal via dest rewrite."""
    contrib = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    bp = (ROOT / "docs" / "BACKPORT.md").read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pcie = (ROOT / "tiles" / "comm" / "pcie" / "README.md").read_text(
        encoding="utf-8"
    )
    w4 = (ROOT / "tiles" / "gemm" / "w4a16_fdot2" / "README.md").read_text(
        encoding="utf-8"
    )

    sneaky = "even if the source Author field is wrong"
    for text in (contrib, bp, arch, readme, pcie, w4):
        assert sneaky not in text
    assert "cherry-pick -x" in contrib
    assert "GIT_COMMITTER_NAME" in contrib

    # Foreign recipe (classify / keep their Author) before this-lab rewrite.
    foreign_idx = contrib.find("Foreign FIRST")
    blivion_force = contrib.find("GIT_AUTHOR_NAME='BlivionIaG'")
    assert foreign_idx != -1
    assert blivion_force != -1
    assert foreign_idx < blivion_force
    assert "leapdragon@gmail.com" in contrib
    assert "Mail@simonsiebert.de" in contrib
    assert "NEVER GIT_AUTHOR_NAME=BlivionIaG" in contrib
    assert "foreign=1" in contrib
    assert "kev29lt@gmail.com" in contrib

    assert "BlivionIaG" in bp
    assert "leapdragon@gmail.com" in bp
    assert "Never `GIT_AUTHOR_NAME=BlivionIaG`" in bp
    assert "leapdragon and a17t keep their Author and Committer" in bp

    assert "even if the commit lives on dest extras" in arch
    assert "leapdragon@gmail.com" in pcie
    assert "GIT_AUTHOR_NAME=BlivionIaG" in pcie
    assert "GIT_AUTHOR_NAME=BlivionIaG" in w4
    assert "leapdragon@gmail.com" in readme
    assert "GIT_AUTHOR_NAME=BlivionIaG" in readme


def test_readme_unvalidated_inventory() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Unvalidated extras inventory" in text
    assert "**Unvalidated.**" in text
    assert "VLLM_RDNA_AR" in text
    assert "fa_rdna2" in text


def test_no_mistaken_dest_author_identities() -> None:
    forbidden = ("kletorch", "opencode.local")
    hits: list[str] = []
    scan = [ROOT / "CONTRIBUTING.md", ROOT / "README.md"]
    scan.extend((ROOT / "docs").rglob("*.md"))
    scan.extend((ROOT / "tiles").rglob("*.md"))
    for path in scan:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in forbidden):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []
