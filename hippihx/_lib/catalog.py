"""Single op registry. Python, C V1, tiles/, and tests stay in lockstep.

ISA class names stay (``fa_fdot2``, not b12x ``paged``). The group rename
``attn`` → ``attention`` matches b12x. ``b12x_analogue`` is a map, not an alias.

V1 rev 4 schema (``params``, ``checks``, ``tensors``, ``scratch``) is
pinned one op at a time. An empty schema is unpinned: ``plan`` takes no
params, ``run`` takes no tensors, and the op stays ``ready=False``. The C
tables and index enums are generated from these rows
(``python -m hippihx._lib.codegen``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .fabric import AR_MAX_KB

COND_OPS: tuple[str, ...] = ("==", "!=", ">", ">=", "<", "<=")
PARAM_KINDS: tuple[str, ...] = ("int", "enum", "bool", "float")
TENSOR_ROLES: tuple[str, ...] = ("in", "out", "inout")
TENSOR_LAYOUTS: tuple[str, ...] = ("contiguous", "rows", "strided")
TENSOR_DTYPES: tuple[str, ...] = ("fp16", "bf16", "fp32", "i32", "i64", "u8", "i8")


@dataclass(frozen=True, slots=True)
class Cond:
    """``param <op> value`` on a plan param. A tuple holds when any one holds."""

    param: str
    op: str
    value: int


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """A plan-time host scalar.

    A captured graph bakes params in. Anything that changes between
    replays (lengths, block tables, query offsets) is a tensor the kernel
    reads on device, never a param. ``int`` checks ``lo``/``hi``
    (inclusive, ``None`` is open). ``enum`` checks ``values`` (``labels``
    name them). ``bool`` is 0 or 1. ``float`` is finite and > 0.
    """

    name: str
    kind: str
    note: str
    lo: int | None = None
    hi: int | None = None
    values: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Divisible:
    """Cross-param check: ``num % by == 0``."""

    num: str
    by: str


@dataclass(frozen=True, slots=True)
class TensorSlot:
    """One V1 run view, in slot order.

    ``dims`` has one rule per axis: ``"*"`` any extent, ``"p"`` equal to
    param ``p``, ``"<=p"`` at most ``p``, or a literal. Every extent is
    >= 1. ``layout``: ``contiguous`` (row-major, no padding), ``rows``
    (unit inner stride, padded outer strides allowed) or ``strided`` (any
    non-negative strides; the kernel takes them). ``when`` empty means
    always required. Otherwise the slot is required iff a cond holds and
    ignored when none does.
    """

    name: str
    dtype: str
    role: str
    dims: tuple[str, ...]
    layout: str = "contiguous"
    when: tuple[Cond, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class ScratchRule:
    """One zeroed scratch spec of ``elem_bytes * prod(dims)`` bytes.

    ``dims`` are param names or literals. ``max_elems`` > 0 bounds
    ``prod(dims)`` (kernel index width). ``when`` empty means always.
    """

    name: str
    elem_bytes: int
    dims: tuple[str, ...]
    max_elems: int = 0
    when: tuple[Cond, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class OpSpec:
    qualname: str
    v1_id: int
    enum: str
    summary: str
    tile: str
    stub_symbol: str
    b12x_analogue: str
    dot: bool = False
    fp16_act: bool = False
    params: tuple[ParamSpec, ...] = ()
    checks: tuple[Divisible, ...] = ()
    tensors: tuple[TensorSlot, ...] = ()
    scratch: tuple[ScratchRule, ...] = ()
    # Enum param whose label names the explore shape plan picks.
    variant: str = ""
    # comm: plan needs caps.fabric on a custom-AR hop, else RCCL.
    needs_fabric: bool = False
    # A migrated HIP body enqueues. False for every op until extras binds V1.
    ready: bool = False


# --- attention.fa_fdot2 (pinned) ---------------------------------------
# Observed extras csrc/rocm/fa_rdna2.cu @ cd38a1d: fa_rdna2_decode_paged,
# fa_rdna2_prefill_paged_varlen[_splitk]. Contract only, not a body dump.
# extras allocates O and the split-K partials inside the op
# (rdna2_persist_zeros). Here the caller owns `out`, and plan sizes the
# partials as scratch.
FA_DECODE = 0
FA_PREFILL = 1
_FA_SPLIT_K = (Cond("mode", "==", FA_DECODE), Cond("kv_splits", ">", 1))
# extras indexes the fp32 partials with 32-bit int math in the decode
# per-head kernels, the decode combine, and the prefill split-K empty-split
# path (its main path widens to int64_t). Past this bound the index wraps.
INT32_INDEX_MAX = 2**31 - 1

_FA_FDOT2_PARAMS = (
    ParamSpec(
        "mode",
        "enum",
        "decode: split-K + combine; prefill: paged varlen",
        values=(FA_DECODE, FA_PREFILL),
        labels=("decode", "prefill"),
    ),
    ParamSpec("head_dim", "enum", "D; extras ships D=128 and D=256", values=(128, 256)),
    ParamSpec("num_q_heads", "int", "H_q", lo=1),
    ParamSpec("num_kv_heads", "int", "H_kv; GQA group is H_q / H_kv", lo=1),
    ParamSpec("block_size", "int", "KV page size in tokens", lo=1),
    ParamSpec("kv_splits", "int", "split-K partitions (extras MAX_SPLITS=16)", lo=1, hi=16),
    ParamSpec("sliding_window", "int", "0 = no window", lo=0),
    ParamSpec("causal", "bool", "prefill mask; decode ignores it"),
    ParamSpec(
        "max_tokens",
        "int",
        "bound on q rows for this plan (the capture bucket); sizes the partials",
        lo=1,
    ),
    ParamSpec("scale", "float", "softmax scale; extras passes 1/sqrt(D)"),
)

_FA_FDOT2_TENSORS = (
    TensorSlot(
        "q", "fp16", "in", ("<=max_tokens", "num_q_heads", "head_dim"), note="[tokens, H_q, D]"
    ),
    TensorSlot(
        "k_cache",
        "fp16",
        "in",
        ("*", "num_kv_heads", "*", "block_size", "*"),
        layout="strided",
        note="[num_blocks, H_kv, D/x, block_size, x]; strides go to the kernel",
    ),
    TensorSlot(
        "v_cache",
        "fp16",
        "in",
        ("*", "*", "*", "*", "*"),
        layout="strided",
        note="5-D paged V; strides go to the kernel",
    ),
    TensorSlot(
        "block_table",
        "i32",
        "in",
        ("*", "*"),
        layout="rows",
        note="[rows, max_blocks]; decode reads one row per query token",
    ),
    TensorSlot("seq_lens", "i32", "in", ("*",), note="[rows]; read on device at replay"),
    TensorSlot(
        "cu_query_lens",
        "i32",
        "in",
        ("*",),
        when=(Cond("mode", "==", FA_PREFILL),),
        note="[num_seqs + 1]; prefill only",
    ),
    TensorSlot(
        "out",
        "fp16",
        "out",
        ("<=max_tokens", "num_q_heads", "head_dim"),
        note="caller-owned [tokens, H_q, D]",
    ),
)

_FA_FDOT2_SCRATCH = (
    ScratchRule(
        "o_partial",
        4,
        ("max_tokens", "num_q_heads", "kv_splits", "head_dim"),
        max_elems=INT32_INDEX_MAX,
        when=_FA_SPLIT_K,
        note="fp32 [T, H_q, S, D]",
    ),
    ScratchRule(
        "m_partial",
        4,
        ("max_tokens", "num_q_heads", "kv_splits"),
        when=_FA_SPLIT_K,
        note="fp32 [T, H_q, S]",
    ),
    ScratchRule(
        "l_partial",
        4,
        ("max_tokens", "num_q_heads", "kv_splits"),
        when=_FA_SPLIT_K,
        note="fp32 [T, H_q, S]",
    ),
)

# --- comm.pcie (params pinned, tensors not) ------------------------------
_COMM_PCIE_PARAMS = (
    ParamSpec(
        "max_bytes",
        "int",
        "largest all-reduce payload on this plan; RCCL above AR_MAX_KB",
        lo=1,
        hi=AR_MAX_KB * 1024,
    ),
)


# Order is the V1 id table. Do not renumber. Qualnames may change with an
# ABI bump (rev 3: attn.* → attention.*).
OPS: tuple[OpSpec, ...] = (
    OpSpec(
        "attention.fa_fdot2",
        0,
        "ATTN_FA_FDOT2",
        "Paged / varlen flash-attn via fdot2 (shared gfx1030+gfx1100 DOT)",
        "attention/fa_fdot2",
        "hippihx_attention_fa_fdot2_stub",
        "attention.paged",
        dot=True,
        fp16_act=True,
        params=_FA_FDOT2_PARAMS,
        checks=(Divisible("num_q_heads", "num_kv_heads"),),
        tensors=_FA_FDOT2_TENSORS,
        scratch=_FA_FDOT2_SCRATCH,
        variant="mode",
    ),
    OpSpec(
        "attention.gdn_scan",
        1,
        "ATTN_GDN_SCAN",
        "GDN / hybrid-state scan (decode + prefill chain)",
        "attention/gdn_scan",
        "hippihx_attention_gdn_scan_stub",
        "sequence.gdn_decode",
        fp16_act=True,
    ),
    OpSpec(
        "attention.kda_scan",
        2,
        "ATTN_KDA_SCAN",
        "KDA linear-state scan (not GDN, not a fused 128x128 product kernel)",
        "attention/kda_scan",
        "hippihx_attention_kda_scan_stub",
        "sequence.kda_prefill",
    ),
    OpSpec(
        "attention.qsa_indexer",
        3,
        "ATTN_QSA_INDEXER",
        "QSA / group-select indexer (not V4 Lightning)",
        "attention/qsa_indexer",
        "hippihx_attention_qsa_indexer_stub",
        "attention.dsa_indexer",
    ),
    OpSpec(
        "attention.dsa_nope",
        4,
        "ATTN_DSA_NOPE",
        "DSA MLA-NoPE (not V4 compressed Lightning)",
        "attention/dsa_nope",
        "hippihx_attention_dsa_nope_stub",
        "attention.sparse_mla",
    ),
    OpSpec(
        "gemm.w4a16_fdot2",
        5,
        "GEMM_W4A16_FDOT2",
        "W4A16 nibble+ZP via fdot2 (GPTQ/AWQ pack modes; one GEMM family)",
        "gemm/w4a16_fdot2",
        "hippihx_gemm_w4a16_fdot2_stub",
        "moe.fused_moe",  # b12x folds W4 into MoE; zoo keeps a dense GEMM class
        dot=True,
        fp16_act=True,
    ),
    OpSpec(
        "gemm.exl3_3inst",
        6,
        "GEMM_EXL3_3INST",
        "EXL3 3inst consume hook (produce stays outside)",
        "gemm/exl3_3inst",
        "hippihx_gemm_exl3_3inst_stub",
        "gemm.trellis_linear",
        dot=True,
        fp16_act=True,
    ),
    OpSpec(
        "moe.routed",
        7,
        "MOE_ROUTED",
        "Routed expert gate / up / down",
        "moe/routed",
        "hippihx_moe_routed_stub",
        "moe.fused_moe",
    ),
    OpSpec(
        "moe.shared",
        8,
        "MOE_SHARED",
        "Shared expert path (shared gfx1030+gfx1100 DOT source)",
        "moe/shared",
        "hippihx_moe_shared_stub",
        "moe.fused_moe",
        dot=True,
        fp16_act=True,
    ),
    OpSpec(
        "moe.leftover_bf16",
        9,
        "MOE_LEFTOVER_BF16",
        "Leftover dense BF16 / cvt then fdot2 (scalar FMA, not fdot2.bf16)",
        "moe/leftover_bf16",
        "hippihx_moe_leftover_bf16_stub",
        "gemm.bf16_gemv",
    ),
    OpSpec(
        "sequence.causal_conv",
        10,
        "SEQUENCE_CAUSAL_CONV",
        "Short-window causal conv (scalar FMA, state_len≈4; not gdn_scan)",
        "sequence/causal_conv",
        "hippihx_sequence_causal_conv_stub",
        "sequence.ple",
    ),
    OpSpec(
        "comm.pcie",
        11,
        "COMM_PCIE",
        "PCIe Uncached+push AR (PIX on PEX88096; INT8/Q8 wire)",
        "comm/pcie",
        "hippihx_comm_pcie_stub",
        "comm.pcie",
        params=_COMM_PCIE_PARAMS,
        needs_fabric=True,
    ),
)


def list_qualnames() -> tuple[str, ...]:
    return tuple(spec.qualname for spec in OPS)


def find_spec(qualname: str) -> OpSpec:
    for spec in OPS:
        if spec.qualname == qualname:
            return spec
    known = list_qualnames()
    raise KeyError(f"unknown hippihx op {qualname!r}; known: {known}")


def spec_by_v1_id(v1_id: int) -> OpSpec:
    for spec in OPS:
        if spec.v1_id == v1_id:
            return spec
    raise KeyError(f"unknown hippihx V1 id {v1_id}")


def schema_errors(spec: OpSpec) -> list[str]:
    """Structural problems in one op's V1 schema. Empty when consistent."""

    errors: list[str] = []
    names = [param.name for param in spec.params]
    kinds = {param.name: param.kind for param in spec.params}

    def int_param(name: str, where: str) -> None:
        if kinds.get(name) not in ("int", "enum", "bool"):
            errors.append(f"{where}: {name!r} is not an int/enum/bool param")

    def conds(items: tuple[Cond, ...], where: str) -> None:
        for cond in items:
            int_param(cond.param, where)
            if cond.op not in COND_OPS:
                errors.append(f"{where}: unknown cond op {cond.op!r}")

    if len(set(names)) != len(names):
        errors.append("duplicate param names")
    for param in spec.params:
        if param.kind not in PARAM_KINDS:
            errors.append(f"param {param.name}: unknown kind {param.kind!r}")
        if param.kind == "enum":
            if not param.values or len(set(param.values)) != len(param.values):
                errors.append(f"param {param.name}: enum needs distinct values")
            if param.labels and len(param.labels) != len(param.values):
                errors.append(f"param {param.name}: labels do not match values")
        elif param.values or param.labels:
            errors.append(f"param {param.name}: values/labels on a non-enum")
        if param.kind != "int" and (param.lo is not None or param.hi is not None):
            errors.append(f"param {param.name}: lo/hi on a non-int")
        if param.lo is not None and param.hi is not None and param.lo > param.hi:
            errors.append(f"param {param.name}: lo > hi")
    for check in spec.checks:
        int_param(check.num, "check")
        int_param(check.by, "check")
    slots = [slot.name for slot in spec.tensors]
    if len(set(slots)) != len(slots):
        errors.append("duplicate tensor slot names")
    for slot in spec.tensors:
        where = f"tensor {slot.name}"
        if slot.dtype not in TENSOR_DTYPES:
            errors.append(f"{where}: unknown dtype {slot.dtype!r}")
        if slot.role not in TENSOR_ROLES:
            errors.append(f"{where}: unknown role {slot.role!r}")
        if slot.layout not in TENSOR_LAYOUTS:
            errors.append(f"{where}: unknown layout {slot.layout!r}")
        if not slot.dims:
            errors.append(f"{where}: rank 0")
        for dim in slot.dims:
            if dim == "*" or dim.isdigit():
                continue
            int_param(dim.removeprefix("<="), where)
        conds(slot.when, where)
    for rule in spec.scratch:
        where = f"scratch {rule.name}"
        if rule.elem_bytes < 1 or not rule.dims:
            errors.append(f"{where}: needs elem_bytes >= 1 and dims")
        for dim in rule.dims:
            if not dim.isdigit():
                int_param(dim, where)
        conds(rule.when, where)
    if spec.variant:
        match = [param for param in spec.params if param.name == spec.variant]
        if not match or match[0].kind != "enum" or not match[0].labels:
            errors.append(f"variant {spec.variant!r} is not a labelled enum param")
    if spec.ready and not spec.tensors:
        errors.append("ready op needs a pinned tensor schema")
    return errors
