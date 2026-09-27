"""Generated contract blocks. The catalog is the one table.

Sources: ``catalog.OPS`` (ops, V1 ids, tiles), ``fatbin`` (arch slots,
ROCm pin), ``isa`` (packed DOT, LDS, wave32), ``protocol`` (V1 dtype
codes) and ``v1`` (ABI revision). Targets keep their hand-written prose.
Only the lines between a marker pair are rewritten::

    // hippihx:gen begin <name> -- python -m hippihx._lib.codegen; do not edit
    // hippihx:gen end <name>

(``#`` instead of ``//`` in CMake, shell and Mojo.)

    python -m hippihx._lib.codegen          # rewrite stale blocks
    python -m hippihx._lib.codegen --check  # exit 1 when a block is stale

Host-only: no compiler, no device, no torch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import fatbin, isa
from .catalog import OPS
from .protocol import DTYPE_UNSET, KNOWN_DTYPES, dtype_to_v1
from .v1 import ABI_REVISION

ROOT = Path(__file__).resolve().parents[2]
BEGIN = "hippihx:gen begin"
END = "hippihx:gen end"
HINT = "python -m hippihx._lib.codegen; do not edit"

# Sources every fatbin compiles that are not catalog tiles.
COMMON_EXTRA_SOURCES: tuple[str, ...] = ("tiles/smoke.hip", "tiles/v1_abi.cpp")


@dataclass(frozen=True, slots=True)
class Block:
    """One marked region in one file."""

    path: str
    name: str
    comment: str
    render: Callable[[], str]


def _lines(lines: Sequence[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def _nondot_arches() -> tuple[str, ...]:
    return tuple(a for a in fatbin.KNOWN_ARCHES if a not in fatbin.DOT_ARCHES)


def _safe_shell_text(text: str) -> str:
    """Text that sits inside a double-quoted bash string unchanged."""

    if any(ch in text for ch in '"$`\\'):
        raise ValueError(f"not safe inside a double-quoted shell string: {text!r}")
    return text


def _cmake_set(name: str, items: Sequence[str]) -> list[str]:
    one = f"set({name} {' '.join(items)})"
    if len(one) <= 78:
        return [one]
    out = [f"set({name}"]
    row = " "
    for item in items:
        if len(row) + 1 + len(item) > 78:
            out.append(row)
            row = " "
        row += f" {item}"
    out.append(row)
    out.append(")")
    return out


# --- C header: include/hippihx/v1.h -------------------------------------


def render_v1_dtype() -> str:
    lines = ["typedef enum hippihx_v1_dtype {"]
    lines.append(f"  HIPPIHX_V1_DTYPE_UNSET = {DTYPE_UNSET},")
    for dtype in KNOWN_DTYPES:
        lines.append(f"  HIPPIHX_V1_DTYPE_{dtype.upper()} = {dtype_to_v1(dtype)},")
    lines.append("} hippihx_v1_dtype;")
    return _lines(lines)


def render_v1_ops() -> str:
    lines = ["typedef enum hippihx_v1_op_id {"]
    lines += [f"  HIPPIHX_V1_OP_{spec.enum} = {spec.v1_id}," for spec in OPS]
    lines.append(f"  HIPPIHX_V1_OP_COUNT = {len(OPS)},")
    lines.append("} hippihx_v1_op_id;")
    return _lines(lines)


def render_v1_revision() -> str:
    return _lines([f"enum {{ HIPPIHX_V1_ABI_REVISION = {ABI_REVISION} }};"])


# --- C++ table: tiles/v1_abi.cpp ----------------------------------------


def render_v1_table() -> str:
    lines = ["constexpr OpRow kOps[HIPPIHX_V1_OP_COUNT] = {"]
    for spec in OPS:
        lines.append(
            f'    {{"{spec.qualname}", {int(spec.dot)}, 0, {int(spec.fp16_act)}}},'
        )
    lines.append("};")
    lines.append("")
    lines.append("constexpr ArchRow kArches[] = {")
    for arch in fatbin.KNOWN_ARCHES:
        lines.append(f'    {{"{arch}", {int(arch in fatbin.DOT_ARCHES)}}},')
    lines.append("};")
    return _lines(lines)


# --- ISA / arch headers --------------------------------------------------


def render_isa_macros() -> str:
    names = {dot.name for dot in isa.PACKED_DOT}
    macros = (
        ("HIPPIHX_ISA_WAVE32", isa.WAVE32),
        ("HIPPIHX_LDS_BANKS", isa.LDS_BANKS),
        ("HIPPIHX_LDS_BANK_BYTES", isa.LDS_BANK_BYTES),
        ("HIPPIHX_LDS_PAD", isa.LDS_PAD),
        ("HIPPIHX_ISA_FDOT2", int(isa.FDOT2.name in names)),
        ("HIPPIHX_ISA_SDOT4", int(isa.SDOT4.name in names)),
        ("HIPPIHX_ISA_WMMA_GATE", int(isa.WMMA_GATE_GFX1030)),
    )
    lines: list[str] = []
    for macro, value in macros:
        if lines:
            lines.append("")
        lines += [f"#ifndef {macro}", f"#define {macro} {value}", "#endif"]
    return _lines(lines)


def render_arch_macros() -> str:
    arches = fatbin.DOT_ARCHES + _nondot_arches() + fatbin.LATER_ARCHES
    lines: list[str] = []
    for arch in arches:
        lines.append(f"#if defined(__{arch}__) || defined(HIPPIHX_ARCH_{arch})")
        lines.append(f"#define HIPPIHX_{arch.upper()} 1")
        if arch in fatbin.DOT_ARCHES:
            lines.append("#define HIPPIHX_DOT_SLOT 1")
        lines.append("#endif")
    return _lines(lines)


# --- CMake / build script -------------------------------------------------


def render_cmake_arches() -> str:
    lines: list[str] = []
    lines += _cmake_set("HIPPIHX_KNOWN_ARCHES", fatbin.KNOWN_ARCHES)
    lines += _cmake_set("HIPPIHX_DOT_ARCHES", fatbin.DOT_ARCHES)
    lines += _cmake_set("HIPPIHX_DOT_UNOPTIMIZED_ARCHES", fatbin.UNOPTIMIZED_DOT_ARCHES)
    lines += _cmake_set("HIPPIHX_LATER_NONDOT_ARCHES", fatbin.LATER_NONDOT_ARCHES)
    lines += _cmake_set("HIPPIHX_LATER_ARCHES", fatbin.LATER_ARCHES)
    lines.append(f"set(HIPPIHX_DEFAULT_ARCH {fatbin.DEFAULT_ARCH})")
    lines.append(f'set(HIPPIHX_ROCM_PIN "{fatbin.ROCM_PIN}" CACHE STRING')
    lines.append(
        '    "Documented ROCm pin for V620 / gfx1030. '
        'Not auto-enforced when HIP is absent.")'
    )
    return _lines(lines)


def render_cmake_sources() -> str:
    dot = [f"tiles/{spec.tile}/kernel.hip" for spec in OPS if spec.dot]
    common = list(COMMON_EXTRA_SOURCES)
    common += [f"tiles/{spec.tile}/kernel.hip" for spec in OPS if not spec.dot]
    lines = ["set(HIPPIHX_DOT_SOURCES"]
    lines += [f"  {path}" for path in dot]
    lines.append(")")
    lines.append("set(HIPPIHX_COMMON_SOURCES")
    lines += [f"  {path}" for path in common]
    lines.append(")")
    return _lines(lines)


def render_build_case() -> str:
    lines = ['case "$ARCH" in', f"  {'|'.join(fatbin.KNOWN_ARCHES)}) ;;"]
    for arch in fatbin.LATER_ARCHES:
        note = _safe_shell_text(fatbin.later_note(arch))
        lines += [
            f"  {arch})",
            f'    echo "{arch}: {note}" >&2',
            "    exit 1",
            "    ;;",
        ]
    known = " ".join(fatbin.KNOWN_ARCHES)
    later = " ".join(fatbin.LATER_ARCHES)
    lines += [
        "  *)",
        f"    echo \"unknown HIPPIHX_ARCH='$ARCH' (built: {known}; Later: {later})\" >&2",
        "    exit 1",
        "    ;;",
        "esac",
    ]
    return _lines(lines)


# --- Mojo: mojo/isa/contracts.mojo ---------------------------------------


def render_mojo_isa() -> str:
    lines = [
        f"alias WAVE32 = {isa.WAVE32}",
        f"alias LDS_BANKS = {isa.LDS_BANKS}",
        f"alias LDS_BANK_BYTES = {isa.LDS_BANK_BYTES}",
        f"alias LDS_PAD = {isa.LDS_PAD}",
        f"alias WMMA_GATE_GFX1030 = {isa.WMMA_GATE_GFX1030}",
        f'alias PACKED_DOT_FDOT2 = "{isa.FDOT2.name}"',
        f'alias PACKED_DOT_SDOT4 = "{isa.SDOT4.name}"',
        f'alias LLVM_FDOT2 = "{isa.FDOT2.llvm}"',
        f'alias LLVM_SDOT4 = "{isa.SDOT4.llvm}"',
        f'alias ISA_FDOT2 = "{",".join(isa.FDOT2.isa)}"',
        f'alias ISA_SDOT4 = "{",".join(isa.SDOT4.isa)}"',
        "",
        "# Same order as hippihx._lib.fatbin.DOT_ARCHES / isa.MAD_MIX_ARCHES.",
        f'alias DOT_ARCHES = "{",".join(fatbin.DOT_ARCHES)}"',
        f'alias MAD_MIX_ARCHES = "{",".join(isa.MAD_MIX_ARCHES)}"',
    ]
    return _lines(lines)


def _mojo_any(arches: Sequence[str], result: str) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(arches), 4):
        chunk = arches[start : start + 4]
        test = " or ".join(f'arch == "{arch}"' for arch in chunk)
        lines += [f"    if {test}:", f'        return "{result}"']
    return lines


def render_mojo_arch_switch() -> str:
    lines = [
        "fn arch_switch(arch: StaticString) -> StaticString:",
        '    """``dot`` or ``mad_mix``. gfx1030 is the primary DOT slot.',
        "",
        "    The arch lists are ``DOT_ARCHES`` and ``MAD_MIX_ARCHES`` above.",
        '    """',
    ]
    lines += _mojo_any(fatbin.DOT_ARCHES, "dot")
    lines += _mojo_any(isa.MAD_MIX_ARCHES, "mad_mix")
    lines.append('    return "unknown"')
    return _lines(lines)


BLOCKS: tuple[Block, ...] = (
    Block("include/hippihx/v1.h", "v1_dtype", "//", render_v1_dtype),
    Block("include/hippihx/v1.h", "v1_ops", "//", render_v1_ops),
    Block("include/hippihx/v1.h", "v1_revision", "//", render_v1_revision),
    Block("tiles/v1_abi.cpp", "v1_table", "//", render_v1_table),
    Block("include/hippihx/isa.hpp", "isa_macros", "//", render_isa_macros),
    Block("include/hippihx/arch.hpp", "arch_macros", "//", render_arch_macros),
    Block("cmake/HippihxArch.cmake", "arch_lists", "#", render_cmake_arches),
    Block("CMakeLists.txt", "tile_sources", "#", render_cmake_sources),
    Block("scripts/build_fatbin.sh", "arch_case", "#", render_build_case),
    Block("mojo/isa/contracts.mojo", "isa_aliases", "#", render_mojo_isa),
    Block("mojo/isa/contracts.mojo", "arch_switch", "#", render_mojo_arch_switch),
)


def splice(text: str, name: str, body: str) -> str:
    """Replace the lines between the ``name`` marker pair with ``body``."""

    lines = text.splitlines(keepends=True)
    begin = [i for i, line in enumerate(lines) if f"{BEGIN} {name} " in line]
    end = [i for i, line in enumerate(lines) if line.rstrip().endswith(f"{END} {name}")]
    if len(begin) != 1 or len(end) != 1:
        raise ValueError(
            f"block {name!r}: need exactly one begin and one end marker, "
            f"found {len(begin)} begin / {len(end)} end"
        )
    if end[0] < begin[0]:
        raise ValueError(f"block {name!r}: end marker comes before begin marker")
    return "".join(lines[: begin[0] + 1]) + body + "".join(lines[end[0] :])


def marker(block: Block, which: str) -> str:
    if which == "begin":
        return f"{block.comment} {BEGIN} {block.name} -- {HINT}"
    return f"{block.comment} {END} {block.name}"


def rendered(root: Path = ROOT) -> dict[str, str]:
    """File path → full text with every block rendered."""

    texts: dict[str, str] = {}
    for block in BLOCKS:
        text = texts.get(block.path)
        if text is None:
            text = (root / block.path).read_text(encoding="utf-8")
        texts[block.path] = splice(text, block.name, block.render())
    return texts


def stale(root: Path = ROOT) -> list[str]:
    """Blocks whose file text differs from the rendered text."""

    out: list[str] = []
    for block in BLOCKS:
        text = (root / block.path).read_text(encoding="utf-8")
        if splice(text, block.name, block.render()) != text:
            out.append(f"{block.path}:{block.name}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hippihx._lib.codegen")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when a generated block is stale"
    )
    args = parser.parse_args(argv)
    if args.check:
        bad = stale()
        for item in bad:
            print(f"stale: {item}", file=sys.stderr)
        return 1 if bad else 0
    for path, text in rendered().items():
        target = ROOT / path
        if target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
