from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _docs() -> str:
    parts = [
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
        (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8"),
        (ROOT / "cmake" / "HippihxArch.cmake").read_text(encoding="utf-8"),
    ]
    return "\n".join(parts)


def test_bc250_is_gfx1013_not_gfx906() -> None:
    text = _docs()
    assert "gfx1013" in text
    assert "Cyan Skillfish" in text
    assert "not** BC-250" in text or "NOT BC-250" in text or "not BC-250" in text
    for line in text.splitlines():
        low = line.lower()
        if "bc-250" in low and "gfx906" in low:
            assert "not" in low, line


def test_docs_forbid_hsa_override() -> None:
    text = _docs()
    assert "HSA_OVERRIDE" in text


def test_fatbin_slots_named() -> None:
    text = _docs()
    for token in (
        "gfx1151",
        "gfx1031",
        "gfx1035",
        "gfx1013",
        "gfx906",
        "gfx1101",
        "gfx1102",
    ):
        assert token in text, token
    assert "portable" in text.lower()
    assert "unoptimized" in text.lower() or "not dest-tuned" in text.lower()


def test_gfx1013_is_later_not_true_rdna2() -> None:
    text = _docs()
    low = text.lower()
    assert "later" in low
    assert "not true rdna2" in low
    assert "cyan skillfish" in low
    assert "steam deck" in low
    assert "wave32" in low
    assert "gfx1033" in low
    cmake = (ROOT / "cmake" / "HippihxArch.cmake").read_text(encoding="utf-8")
    assert "HIPPIHX_LATER_ARCHES gfx906 gfx1013" in cmake
    assert "not a dest fatbin" in cmake.lower()
    assert "HSA_OVERRIDE" in text
