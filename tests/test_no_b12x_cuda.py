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


def test_no_fdot2_bf16_in_source() -> None:
    """gfx1030 LLVM aborts on fdot2.bf16; dest rolled those paths back."""
    forbidden = (
        "fdot2.bf16",
        "llvm.amdgcn.fdot2.bf16",
        "__builtin_amdgcn_fdot2_bf16",
    )
    hits: list[str] = []
    scan_dirs = [ROOT / "tiles", ROOT / "include"]
    for base in scan_dirs:
        for path in base.rglob("*"):
            if path.suffix not in {".hip", ".hpp", ".h", ".cu", ".cuh", ".cpp"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            # Mentions in comments/docs that *forbid* the ISA are fine.
            for token in forbidden:
                if token not in text:
                    continue
                for line in text.splitlines():
                    if token not in line:
                        continue
                    low = line.lower()
                    if any(
                        w in low
                        for w in (
                            "never",
                            "not ",
                            "not`",
                            "abort",
                            "error",
                            "no ",
                            "refus",
                        )
                    ):
                        continue
                    hits.append(f"{path.relative_to(ROOT)}:{line.strip()}")
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
    assert "kev29lt@gmail.com" in text
    bp = (ROOT / "docs" / "BACKPORT.md").read_text(encoding="utf-8")
    assert "BlivionIaG" in bp
    assert "leapdragon@gmail.com" in bp


def test_readme_unvalidated_inventory() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Unvalidated extras inventory" in text
    assert "**Unvalidated.**" in text
    assert "VLLM_RDNA_AR" in text
    assert "VLLM_RDNA_AR_MAX_KB" in text
    assert "fa_rdna2" in text
    assert "a4060647" in text
    assert "1046782" in text
    bp = (ROOT / "docs" / "BACKPORT.md").read_text(encoding="utf-8")
    assert "a4060647" in bp
    assert "1046782" in bp
    assert "i_t_local" in bp
    assert "rdna2_graph_keepalive" in bp
    assert "**not** squash" in bp
    assert "leapdragon@gmail.com" in bp
    assert "Dest extras defects" in bp
    assert "prep_zero_scale_fp16" in bp
    assert "k_per_split" in bp
    assert "fdot2.bf16" in bp
    assert "q_gemm_rdna2_awq_prefill" in bp
    w4 = (ROOT / "tiles" / "gemm" / "w4a16_fdot2" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "K_STEP=32" in w4
    assert "ConfigA" in w4
    assert "integer" in text.lower() or "Integer" in w4


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
