from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = {".py", ".hip", ".cu", ".cuh", ".hpp", ".h", ".cpp", ".cc"}

IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(cute|cutlass|b12x)\b"
    r"|^\s*#include\s*[<\"](?:cute/|cutlass/|cuda_runtime)",
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
