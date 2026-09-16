"""Single op registry. Python, C V1, tiles/, and tests stay in lockstep.

ISA class names stay (``fa_fdot2``, not b12x ``paged``). The group rename
``attn`` → ``attention`` matches b12x. ``b12x_analogue`` is a map, not an alias.
"""

from __future__ import annotations

from dataclasses import dataclass


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
