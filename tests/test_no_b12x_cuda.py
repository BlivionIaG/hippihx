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
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "cherry-pick -x" in text
    assert "GIT_COMMITTER_NAME" in text
    assert "--reset-author" in text
    bp = (ROOT / "docs" / "BACKPORT.md").read_text(encoding="utf-8")
    assert "kletorch" in bp
    assert "leapdragon@gmail.com" in bp
    assert "agent@opencode.local" in bp
    assert "cursoragent@cursor.com" in bp
