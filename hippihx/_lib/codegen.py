"""Generated contract blocks. The catalog is the one table.

Sources: ``catalog.OPS`` (ops, V1 ids, tiles, V1 rev 4 schemas),
``fatbin`` (arch slots, ROCm pin), ``isa`` (packed DOT, LDS, wave32),
``fabric`` (hop / switch classes, custom-AR set) and ``v1`` (revision,
status and dtype codes, limits). Targets keep their hand-written prose.
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

from . import fabric, fatbin, isa, v1
from .catalog import OPS, Cond, OpSpec, ParamSpec

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


def _c_ident(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).upper()


def render_v1_status() -> str:
    lines = ["enum {"]
    for status in v1.V1Status:
        name = "HIPPIHX_V1_OK" if status is v1.V1Status.OK else f"HIPPIHX_V1_{status.name}"
        note = v1.STATUS_NOTES.get(status)
        tail = f"  // {note}" if note else ""
        lines.append(f"  {name} = {int(status)},{tail}")
    lines.append("};")
    return _lines(lines)


def render_v1_limits() -> str:
    return _lines(
        [
            f"#define HIPPIHX_V1_MAX_RANK {v1.MAX_RANK}",
            f"#define HIPPIHX_V1_MAX_PARAMS {v1.MAX_PARAMS}",
            f"#define HIPPIHX_V1_MAX_SCRATCH {v1.MAX_SCRATCH}",
            f"#define HIPPIHX_V1_SCRATCH_ALIGN {v1.SCRATCH_ALIGN}",
        ]
    )


def render_v1_dtype() -> str:
    lines = ["typedef enum hippihx_v1_dtype {", "  HIPPIHX_V1_DTYPE_UNSET = 0,"]
    for dtype, code in v1.V1_DTYPES.items():
        lines.append(f"  HIPPIHX_V1_DTYPE_{dtype.upper()} = {code},")
    lines.append("} hippihx_v1_dtype;")
    return _lines(lines)


def render_v1_arch() -> str:
    lines = ["typedef enum hippihx_v1_arch {"]
    for index, arch in enumerate(fatbin.KNOWN_ARCHES):
        lines.append(f"  HIPPIHX_V1_ARCH_{arch.upper()} = {index},")
    lines.append(f"  HIPPIHX_V1_ARCH_COUNT = {len(fatbin.KNOWN_ARCHES)},")
    lines.append("} hippihx_v1_arch;")
    return _lines(lines)


def render_v1_fabric() -> str:
    lines = ["typedef enum hippihx_v1_hop {", "  HIPPIHX_V1_HOP_UNSET = 0,"]
    lines += [f"  HIPPIHX_V1_HOP_{_c_ident(h)} = {c}," for h, c in v1.V1_HOPS.items()]
    lines += ["} hippihx_v1_hop;", ""]
    lines += ["typedef enum hippihx_v1_switch {", "  HIPPIHX_V1_SWITCH_UNSET = 0,"]
    lines += [
        f"  HIPPIHX_V1_SWITCH_{_c_ident(s)} = {c}," for s, c in v1.V1_SWITCHES.items()
    ]
    lines.append("} hippihx_v1_switch;")
    return _lines(lines)


def _domain(param: ParamSpec) -> str:
    if param.kind == "float":
        return "float > 0"
    if param.kind == "bool":
        return "bool 0/1"
    if param.kind == "enum":
        if param.labels:
            pairs = " ".join(f"{lab}={val}" for lab, val in zip(param.labels, param.values))
            return f"enum {pairs}"
        return "enum " + "|".join(str(val) for val in param.values)
    if param.lo is not None and param.hi is not None:
        return f"int [{param.lo}, {param.hi}]"
    if param.lo is not None:
        return f"int >= {param.lo}"
    if param.hi is not None:
        return f"int <= {param.hi}"
    return "int"


def _cond_text(spec: OpSpec, cond: Cond) -> str:
    param = next(p for p in spec.params if p.name == cond.param)
    value = str(cond.value)
    if param.labels and cond.value in param.values:
        value = param.labels[param.values.index(cond.value)]
    return f"{cond.param} {cond.op} {value}"


def render_v1_schema() -> str:
    lines: list[str] = []
    for spec in OPS:
        if not spec.params and not spec.tensors:
            continue
        prefix = f"HIPPIHX_V1_{spec.enum}"
        if lines:
            lines.append("")
        if spec.params:
            lines.append(f"// {spec.qualname} params, in order.")
            lines.append("enum {")
            for index, param in enumerate(spec.params):
                lines.append(
                    f"  {prefix}_P_{_c_ident(param.name)} = {index},"
                    f"  // {_domain(param)}: {param.note}"
                )
            lines.append(f"  {prefix}_NPARAMS = {len(spec.params)},")
            lines.append("};")
            for param in spec.params:
                if not param.labels:
                    continue
                lines.append("enum {")
                for label, value in zip(param.labels, param.values):
                    lines.append(
                        f"  {prefix}_{_c_ident(param.name)}_{_c_ident(label)} = {value},"
                    )
                lines.append("};")
        if spec.tensors:
            lines.append(f"// {spec.qualname} tensor slots, in order.")
            lines.append("enum {")
            for index, slot in enumerate(spec.tensors):
                when = ""
                if slot.when:
                    when = " when " + " or ".join(_cond_text(spec, c) for c in slot.when)
                lines.append(
                    f"  {prefix}_T_{_c_ident(slot.name)} = {index},"
                    f"  // {slot.role} {slot.dtype} [{', '.join(slot.dims)}]"
                    f" {slot.layout}{when}"
                )
            lines.append(f"  {prefix}_NTENSORS = {len(spec.tensors)},")
            lines.append("};")
    return _lines(lines)


def render_v1_ops() -> str:
    lines = ["typedef enum hippihx_v1_op_id {"]
    lines += [f"  HIPPIHX_V1_OP_{spec.enum} = {spec.v1_id}," for spec in OPS]
    lines.append(f"  HIPPIHX_V1_OP_COUNT = {len(OPS)},")
    lines.append("} hippihx_v1_op_id;")
    return _lines(lines)


def render_v1_revision() -> str:
    return _lines([f"enum {{ HIPPIHX_V1_ABI_REVISION = {v1.ABI_REVISION} }};"])


# --- C++ table: tiles/v1_abi.cpp ----------------------------------------


_KIND = {"int": "kInt", "enum": "kEnum", "bool": "kBool", "float": "kFloat"}
_COND = {"==": "kEq", "!=": "kNe", ">": "kGt", ">=": "kGe", "<": "kLt", "<=": "kLe"}
_LAYOUT = {"contiguous": "kContiguous", "rows": "kRows", "strided": "kStrided"}


def _c_array(ctype: str, name: str, rows: Sequence[str], sentinel: str) -> list[str]:
    """``constexpr`` array. A zero-length array is ill-formed, so pad one row."""

    out = [f"constexpr {ctype} {name}[] = {{"]
    if rows:
        out += [f"    {row}," for row in rows]
    else:
        out.append(f"    {sentinel},  // unused sentinel")
    out.append("};")
    return out


def _int64(value: int | None, default: str) -> str:
    return default if value is None else str(value)


def render_v1_table() -> str:
    ops: list[str] = []
    params: list[str] = []
    enum_values: list[str] = []
    conds: list[str] = []
    checks: list[str] = []
    dims: list[str] = []
    slots: list[str] = []
    scratch: list[str] = []
    ranks: list[str] = []

    for spec in OPS:
        index = {param.name: i for i, param in enumerate(spec.params)}

        def cond_rows(items: Sequence[Cond]) -> tuple[int, int]:
            first = len(conds)
            conds.extend(
                f"{{{index[c.param]}, {_COND[c.op]}, {c.value}}}" for c in items
            )
            return first, len(items)

        def dim_rows(items: Sequence[str]) -> int:
            first = len(dims)
            for dim in items:
                if dim == "*":
                    dims.append("{kAny, 0}")
                elif dim.isdigit():
                    dims.append(f"{{kLiteral, {dim}}}")
                elif dim.startswith("<="):
                    dims.append(f"{{kLeParam, {index[dim[2:]]}}}")
                else:
                    dims.append(f"{{kParam, {index[dim]}}}")
            return first

        params_first = len(params)
        for param in spec.params:
            values_first = len(enum_values)
            enum_values.extend(str(value) for value in param.values)
            params.append(
                f'{{"{param.name}", {_KIND[param.kind]}, '
                f"{_int64(param.lo, 'INT64_MIN')}, {_int64(param.hi, 'INT64_MAX')}, "
                f"{values_first}, {len(param.values)}}}"
            )
        checks_first = len(checks)
        checks.extend(f"{{{index[c.num]}, {index[c.by]}}}" for c in spec.checks)
        slots_first = len(slots)
        for slot in spec.tensors:
            dims_first = dim_rows(slot.dims)
            when_first, when_count = cond_rows(slot.when)
            slots.append(
                f'{{"{slot.name}", HIPPIHX_V1_DTYPE_{slot.dtype.upper()}, '
                f"{len(slot.dims)}, {dims_first}, {_LAYOUT[slot.layout]}, "
                f"{when_first}, {when_count}}}"
            )
        scratch_first = len(scratch)
        for rule in spec.scratch:
            dims_first = dim_rows(rule.dims)
            when_first, when_count = cond_rows(rule.when)
            scratch.append(
                f'{{"{rule.name}", {rule.elem_bytes}, {dims_first}, {len(rule.dims)}, '
                f"{rule.max_elems}, {when_first}, {when_count}}}"
            )
        variant_param = index[spec.variant] if spec.variant else -1
        variant_first = len(ranks) if spec.variant else 0
        ranks.extend(str(rank) for rank in v1.variant_ranks(spec, fatbin.DEFAULT_ARCH))
        ops.append(
            f'{{"{spec.qualname}", {int(spec.dot)}, {int(spec.fp16_act)}, '
            f"{int(spec.ready)}, {int(spec.needs_fabric)}, "
            f"{params_first}, {len(spec.params)}, {checks_first}, {len(spec.checks)}, "
            f"{slots_first}, {len(spec.tensors)}, {scratch_first}, {len(spec.scratch)}, "
            f"{variant_param}, {variant_first}}}"
        )

    arches = [
        f'{{"{arch}", {int(arch in fatbin.DOT_ARCHES)}, {fatbin.default_wave(arch)}}}'
        for arch in fatbin.KNOWN_ARCHES
    ]
    ar_hops = [f"HIPPIHX_V1_HOP_{_c_ident(h)}" for h in fabric.CUSTOM_AR_HOPS]
    ar_switches = [f"HIPPIHX_V1_SWITCH_{_c_ident(s)}" for s in fabric.CUSTOM_AR_SWITCHES]

    lines = ["constexpr OpRow kOps[HIPPIHX_V1_OP_COUNT] = {"]
    lines += [f"    {row}," for row in ops]
    lines += ["};", "", "constexpr ArchRow kArches[HIPPIHX_V1_ARCH_COUNT] = {"]
    lines += [f"    {row}," for row in arches]
    lines += ["};", ""]
    lines += _c_array("ParamRow", "kParams", params, '{"", kInt, 0, 0, 0, 0}')
    lines += _c_array("int64_t", "kEnumValues", enum_values, "0")
    lines += _c_array("CondRow", "kConds", conds, "{0, kEq, 0}")
    lines += _c_array("CheckRow", "kChecks", checks, "{0, 0}")
    lines += _c_array("DimRow", "kDims", dims, "{kAny, 0}")
    lines += _c_array("SlotRow", "kSlots", slots, '{"", 0, 0, 0, kContiguous, 0, 0}')
    lines += _c_array("ScratchRow", "kScratch", scratch, '{"", 0, 0, 0, 0, 0, 0}')
    lines += _c_array("int", "kVariantRanks", ranks, "0")
    lines.append("")
    lines.append(f"constexpr int kNumHops = {len(v1.V1_HOPS)};")
    lines.append(f"constexpr int kNumSwitches = {len(v1.V1_SWITCHES)};")
    lines += _c_array("int", "kArHops", ar_hops, "0")
    lines += _c_array("int", "kArSwitches", ar_switches, "0")
    lines += _c_array("int", "kLinkWidths", [str(w) for w in fabric.LINK_WIDTHS], "0")
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
    Block("include/hippihx/v1.h", "v1_status", "//", render_v1_status),
    Block("include/hippihx/v1.h", "v1_limits", "//", render_v1_limits),
    Block("include/hippihx/v1.h", "v1_dtype", "//", render_v1_dtype),
    Block("include/hippihx/v1.h", "v1_arch", "//", render_v1_arch),
    Block("include/hippihx/v1.h", "v1_fabric", "//", render_v1_fabric),
    Block("include/hippihx/v1.h", "v1_ops", "//", render_v1_ops),
    Block("include/hippihx/v1.h", "v1_schema", "//", render_v1_schema),
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
