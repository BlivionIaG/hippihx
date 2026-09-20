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
    extras = (ROOT / "docs" / "EXTRAS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Unvalidated extras inventory" in extras
    assert "**Unvalidated.**" in extras
    assert "VLLM_RDNA_AR" in extras
    assert "VLLM_RDNA_AR_MAX_KB" in extras
    assert "fa_rdna2" in extras
    assert "a4060647" in extras
    assert "1ff735" in extras
    assert "4d25a0" in extras
    assert "d0d577" in extras
    assert "8960a3" in extras
    assert "388a61" in extras
    assert "50120e1" in extras
    assert "3bddd3c" in extras
    assert "5c4ab98" in extras
    assert "3b59ee1" in extras
    assert "609c9c0" in extras
    assert "f3dd65f" in extras
    assert "e45dd5c" in extras
    assert "d1b200" in extras
    assert "1046782" in extras
    assert "docs/EXTRAS.md" in readme
    assert "attention.fa_fdot2" in readme or "attention/fa_fdot2" in readme
    assert "attn.fa_fdot2" not in readme
    text = extras
    bp = (ROOT / "docs" / "BACKPORT.md").read_text(encoding="utf-8")
    assert "a4060647" in bp
    assert "1ff735" in bp
    assert "4d25a0" in bp
    assert "f5cbbd" in bp
    assert "7e70e2" in bp
    assert "d0d577" in bp
    assert "8960a3" in bp
    assert "388a61" in bp
    assert "50120e1" in bp
    assert "3bddd3c" in bp
    assert "5c4ab98" in bp
    assert "31003ff" in bp
    assert "741e5bc" in bp
    assert "3b59ee1" in bp
    assert "9c60294" in bp
    assert "609c9c0" in bp
    assert "4425834" in bp
    assert "3d6df9e" in bp
    assert "ed94e3f" in bp
    assert "b33f9b6" in bp
    assert "f3dd65f" in bp
    assert "dbb1e77" in bp
    assert "e45dd5c" in bp
    assert "d1b200" in bp
    assert "cd1231" in bp
    assert "9c9509" in bp
    assert "cb0d441" in bp
    assert "849292ec" in bp
    assert "820465" in bp
    assert "1046782" in bp
    assert "02adbfd4" in bp
    assert "new_zeros" in extras or "zeros_like" in bp
    assert "i_t_local" in bp
    assert "rdna2_graph_keepalive" in bp
    assert "**not** squash" in bp
    assert "leapdragon@gmail.com" in bp
    assert "Dest extras defects" in bp
    assert "prep_zero_scale_fp16" in bp
    assert "k_per_split" in bp
    assert "fdot2.bf16" in bp
    assert "q_gemm_rdna2_awq_prefill" in bp
    assert "_awq_prefill_available" in bp
    assert "ConfigH" in bp
    assert "VLLM_RDNA_QSA_HIP" in bp or "qsa_rdna2" in bp
    assert "torch.ops.hippihx" in extras or "torch.ops.hippihx" in bp
    assert "wvSplitK" in extras
    assert "PR **#12**" in extras or "PR **#12**" in bp
    assert "PR **#13**" in extras or "PR **#13**" in bp
    assert "PR **#14**" in extras or "PR **#14**" in bp
    assert "PR **#15**" in extras or "PR **#15**" in bp
    contrib = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "f3dd65fa" in contrib
    assert "a4060647" in contrib
    # CONTRIBUTING is tip-only. Historical dest SHAs live in BACKPORT.
    for token in (
        "b33f9b6",
        "3d6df9e",
        "ed94e3f",
        "4425834",
        "609c9c0",
        "3b59ee1",
        "741e5bc",
        "5c4ab98",
        "50120e1",
        "388a61",
        "dbb1e77",
        "9c60294",
        "31003ff",
    ):
        assert token not in contrib
    gdn = (ROOT / "tiles" / "attention" / "gdn_scan" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "02adbfd4" in gdn
    assert "fp16" in gdn
    qsa = (ROOT / "tiles" / "attention" / "qsa_indexer" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "1ff735" in qsa or "8cf0dedb" in qsa
    w4 = (ROOT / "tiles" / "gemm" / "w4a16_fdot2" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "K_STEP=32" in w4
    assert "ConfigA" in w4
    assert "ConfigH" in w4
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
