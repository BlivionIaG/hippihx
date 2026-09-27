"""Mojo/MAX authoring scaffold. Host-only. No Mojo compiler, no tok/s."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hippihx._lib.backend import is_dest_backend, is_v1_consume_backend, is_zoo_backend
from hippihx._lib.fatbin import DOT_ARCHES
from hippihx.attention import dsa_nope, fa_fdot2, gdn_scan, kda_scan, qsa_indexer
from hippihx.isa import (
    FDOT2,
    LDS_BANK_BYTES,
    LDS_BANKS,
    LDS_PAD,
    MAD_MIX_ARCHES,
    QSA_OCCUPANCY_THREADS,
    QSA_SPILL_WARPS,
    SDOT4,
    WAVE32,
    WMMA_GATE_GFX1030,
    ArchSwitch,
    arch_switch,
    refuse_dot_on_mad_mix,
    wave_occupancy_ok,
)
from hippihx.moe import leftover_bf16, routed
from hippihx.mojo import (
    FA_FDOT2_MOJO,
    HOOKS,
    ISA_MOJO,
    LEFT_WITHOUT_NEW_BRIEF,
    MOJO_AUTHORING,
    MOJO_DEST_READY,
    MOJO_PRODUCE,
    MOJO_V1_CONSUME,
    SOURCE_ROOT,
    preferred_explore,
    ranked_explore,
    refuse_produce,
)
from hippihx.protocol import Caps, MojoNotProduce
from hippihx.sequence import causal_conv

ROOT = Path(__file__).resolve().parents[1]


def test_mojo_is_authoring_not_produce() -> None:
    assert MOJO_AUTHORING is True
    assert MOJO_PRODUCE is False
    assert MOJO_V1_CONSUME is False
    assert MOJO_DEST_READY is False
    assert is_zoo_backend("mojo")
    assert not is_dest_backend("mojo")
    assert is_dest_backend("hip")
    assert is_dest_backend("flydsl")
    assert is_v1_consume_backend("hip")
    assert not is_v1_consume_backend("mojo")
    assert not is_v1_consume_backend("flydsl")
    Caps(arch="gfx1030", backend="mojo")
    with pytest.raises(MojoNotProduce):
        refuse_produce()


def test_mojo_plan_bind_refuses_run() -> None:
    caps = Caps(arch="gfx1030", backend="mojo")
    assert fa_fdot2.is_supported(caps)
    plan = fa_fdot2.plan(caps)
    assert plan.meta["backend"] == "mojo"
    assert plan.meta["arch_switch"] == "dot"
    assert plan.meta["produce"] is False
    assert plan.meta["explore"] == ("decode", "prefill")
    assert plan.meta["wave"] == 32
    binding = fa_fdot2.bind(plan, scratch=None)
    with pytest.raises(MojoNotProduce, match="not dest-ready"):
        fa_fdot2.run(binding)
    with pytest.raises(MojoNotProduce, match="hipModuleLoad"):
        fa_fdot2.run(binding)
    with pytest.raises(MojoNotProduce, match="ABI gap"):
        fa_fdot2.run(binding)


def test_hip_plan_still_runs_stub() -> None:
    plan = fa_fdot2.plan(Caps(arch="gfx1030"))
    assert plan.meta["produce"] is True
    assert plan.meta["backend"] == "hip"
    assert fa_fdot2.run(fa_fdot2.bind(plan)) is None


def test_packed_dot_and_no_wmma_gate() -> None:
    assert WMMA_GATE_GFX1030 is False
    assert FDOT2.name == "fdot2"
    assert FDOT2.llvm == "llvm.amdgcn.fdot2"
    assert "v_dot2c_f32_f16" in FDOT2.isa
    assert SDOT4.name == "sdot4"
    assert SDOT4.llvm == "llvm.amdgcn.sdot4"
    assert SDOT4.k == 4
    assert LDS_BANKS == 32
    assert LDS_BANK_BYTES == 4
    assert LDS_PAD == 8
    assert WAVE32 == 32
    assert wave_occupancy_ok(dot=True, wave=32)
    assert not wave_occupancy_ok(dot=True, wave=64)
    from hippihx.flydsl import FDOT2 as FLY_FDOT2
    from hippihx.flydsl import SDOT4 as FLY_SDOT4

    assert FLY_FDOT2.llvm == FDOT2.llvm
    assert FLY_SDOT4.isa == SDOT4.isa


def test_arch_switch_keeps_dot_off_mad_mix() -> None:
    assert arch_switch("gfx1030") is ArchSwitch.DOT
    assert arch_switch("gfx1102") is ArchSwitch.DOT
    assert arch_switch("gfx1200") is ArchSwitch.DOT
    assert refuse_dot_on_mad_mix("gfx1200") == "gfx1200"
    assert arch_switch("gfx1033") is ArchSwitch.DOT
    assert MAD_MIX_ARCHES == ("gfx900", "gfx906", "gfx1013")
    for arch in MAD_MIX_ARCHES:
        assert arch_switch(arch) is ArchSwitch.MAD_MIX
        with pytest.raises(ValueError, match="mad_mix"):
            refuse_dot_on_mad_mix(arch)
    assert refuse_dot_on_mad_mix("gfx1030") == "gfx1030"
    assert ranked_explore("attention.fa_fdot2", "gfx900") == ()
    assert ranked_explore("gemm.w4a16_fdot2", "gfx1013") == ()
    assert ranked_explore("gemm.exl3_3inst", "gfx906") == ()


def test_ranked_explore_shapes() -> None:
    fa = ranked_explore("attention.fa_fdot2", "gfx1030")
    assert [shape.rank for shape in fa] == [0, 1]
    assert fa[0].name == "decode" and fa[0].threads == 128
    assert fa[1].name == "prefill"
    qsa = ranked_explore("attention.qsa_indexer", "gfx1030")
    assert qsa[0].threads == QSA_OCCUPANCY_THREADS == 128
    assert qsa[0].spill is False
    assert qsa[1].spill is True
    assert qsa[1].threads == QSA_SPILL_WARPS * WAVE32
    assert preferred_explore("attention.qsa_indexer", "gfx1030") == (qsa[0],)
    gdn = ranked_explore("attention.gdn_scan", "gfx1030")
    assert gdn[0].lds_bytes == 0 and gdn[0].threads == 256
    assert gdn[1].name == "prefill_o" and gdn[1].lds_bytes == 45312
    w4 = ranked_explore("gemm.w4a16_fdot2", "gfx1100")
    assert [shape.name for shape in w4] == ["config_c", "config_a"]
    assert "config_h" not in {shape.name for shape in w4}
    assert "K_STEP=32" in w4[0].note
    conv = ranked_explore("sequence.causal_conv", "gfx1030")
    assert conv[0].lds_bytes == 0 and "state_len 4" in conv[0].note
    assert "fdot2" in ranked_explore("moe.leftover_bf16", "gfx1030")[0].note


def test_family_hooks_on_bind() -> None:
    names = {hook.name for hook in HOOKS}
    assert names == {
        "qwen.qsa",
        "qwen.gdn",
        "qwen.ple",
        "moe.routed",
        "moe.leftover_bf16",
        "hybrid.heap",
    }
    assert fa_fdot2.FAMILY_HOOKS == ()
    gdn = gdn_scan.plan(Caps(arch="gfx1030"))
    assert gdn.meta["families"] == ("qwen.gdn", "hybrid.heap")
    bound = gdn_scan.bind(gdn, family="hybrid.heap")
    assert bound.family == "hybrid.heap"
    assert bound.scratch is None
    with pytest.raises(ValueError, match="not a bind hook"):
        gdn_scan.bind(gdn, family="qwen.qsa")
    ple = causal_conv.plan(Caps(arch="gfx1030"))
    assert "qwen.ple" in ple.meta["families"]
    assert "hybrid.heap" in ple.meta["families"]
    causal_conv.bind(ple, family="qwen.ple")
    assert {hook.name for hook in qsa_indexer.FAMILY_HOOKS} == {"qwen.qsa"}
    assert {hook.name for hook in routed.FAMILY_HOOKS} == {"moe.routed"}
    assert {hook.name for hook in leftover_bf16.FAMILY_HOOKS} == {"moe.leftover_bf16"}
    routed_plan = routed.plan(Caps())
    routed.bind(routed_plan, family="moe.routed")
    with pytest.raises(ValueError, match="not a bind hook"):
        routed.bind(routed_plan, family="moe.leftover_bf16")
    by_name = {hook.name: hook for hook in HOOKS}
    assert [t.name for t in by_name["qwen.qsa"].tensors] == ["groups"]
    assert [t.dtype for t in by_name["qwen.gdn"].tensors] == [
        "fp16",
        "fp16",
        "fp16",
        "fp16",
        "fp16|fp32",
    ]
    assert [t.name for t in by_name["qwen.gdn"].tensors] == [
        "mixed_qkv",
        "a",
        "b",
        "out",
        "state",
    ]
    assert [t.name for t in by_name["qwen.ple"].tensors] == ["input", "out", "state"]
    assert [t.name for t in by_name["moe.routed"].tensors] == ["gate", "up", "down"]
    assert by_name["moe.leftover_bf16"].tensors[0].name == "dense"
    assert by_name["moe.leftover_bf16"].tensors[0].dtype == "bf16"
    assert by_name["hybrid.heap"].tensors[0].name == "heap"
    assert LEFT_WITHOUT_NEW_BRIEF == ("attention.kda_scan", "attention.dsa_nope")
    assert kda_scan.FAMILY_HOOKS == ()
    assert dsa_nope.FAMILY_HOOKS == ()
    with pytest.raises(ValueError, match="not a bind hook"):
        kda_scan.bind(kda_scan.plan(Caps()), family="qwen.gdn")


def test_mojo_sources_match_isa_and_are_not_fatbin() -> None:
    assert SOURCE_ROOT.is_dir()
    assert (SOURCE_ROOT / "__init__.mojo").is_file()
    isa = ISA_MOJO.read_text(encoding="utf-8")
    header = (ROOT / "include" / "hippihx" / "isa.hpp").read_text(encoding="utf-8")
    fa = FA_FDOT2_MOJO.read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    def alias(name: str) -> str:
        match = re.search(rf'^alias {name} = "([^"]+)"', isa, re.M)
        assert match is not None, name
        return match.group(1)

    assert tuple(alias("DOT_ARCHES").split(",")) == DOT_ARCHES
    assert tuple(alias("MAD_MIX_ARCHES").split(",")) == MAD_MIX_ARCHES
    assert alias("PACKED_DOT_FDOT2") == FDOT2.name
    assert alias("PACKED_DOT_SDOT4") == SDOT4.name
    assert alias("LLVM_FDOT2") == FDOT2.llvm
    assert alias("LLVM_SDOT4") == SDOT4.llvm
    assert "hipModuleLoad" in isa
    assert "hipcc 7.14" in isa
    assert "WMMA_GATE_GFX1030 = False" in isa
    for macro, value in (
        ("HIPPIHX_LDS_BANKS", LDS_BANKS),
        ("HIPPIHX_LDS_BANK_BYTES", LDS_BANK_BYTES),
        ("HIPPIHX_LDS_PAD", LDS_PAD),
        ("HIPPIHX_ISA_WAVE32", WAVE32),
        ("HIPPIHX_ISA_WMMA_GATE", 0),
        ("HIPPIHX_ISA_FDOT2", 1),
        ("HIPPIHX_ISA_SDOT4", 1),
    ):
        assert f"#define {macro} {value}" in header
    assert "hipModuleLoad" in header
    assert '@extensibility.register' in fa
    assert 'arch="gfx1030"' in fa
    assert 'arch="gfx900"' in fa
    assert "hipModuleLoad" in fa
    assert "enqueue_function" not in fa
    assert "alias WAVE = 32" in fa
    assert "alias PRODUCE = False" in fa
    assert ".mojo" not in cmake
    doc = (ROOT / "docs" / "MOJO.md").read_text(encoding="utf-8")
    assert "hipModuleLoad" in doc
    assert "not dest-ready" in doc.lower()
    assert "ABI gap" in doc
    assert "hippihx_v1_plan" in doc
    assert "MAX-serve soak" in doc
    assert "LEFT_WITHOUT_NEW_BRIEF" in doc
    assert "attention.kda_scan" in doc
    assert "DeepSeek" in doc
    assert "mixed_qkv" in doc
    assert "rdna_extras" in doc
