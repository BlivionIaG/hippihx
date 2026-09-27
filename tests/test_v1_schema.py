"""V1 rev 4 schema rows and the Python sized plan. Host-only, no compiler."""

from __future__ import annotations

import math

import pytest

from hippihx._lib.catalog import FA_DECODE, FA_PREFILL, OPS, find_spec, schema_errors
from hippihx._lib.fatbin import DOT_ARCHES
from hippihx._lib.v1 import (
    MAX_PARAMS,
    MAX_RANK,
    MAX_SCRATCH,
    SCRATCH_ALIGN,
    V1Error,
    V1Status,
    arena_layout,
    variant_ranks,
)
from hippihx.attention import fa_fdot2
from hippihx.comm import pcie
from hippihx.mojo import LEFT_WITHOUT_NEW_BRIEF
from hippihx.protocol import Caps, Fabric

DECODE = {
    "mode": "decode",
    "head_dim": 128,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "block_size": 16,
    "kv_splits": 4,
    "sliding_window": 0,
    "causal": True,
    "max_tokens": 8,
    "scale": 1 / math.sqrt(128),
}


@pytest.mark.parametrize("spec", OPS, ids=lambda spec: spec.qualname)
def test_schema_rows_are_consistent(spec) -> None:
    assert schema_errors(spec) == []
    assert len(spec.params) <= MAX_PARAMS
    assert len(spec.scratch) <= MAX_SCRATCH
    assert all(len(slot.dims) <= MAX_RANK for slot in spec.tensors)


def test_nothing_is_ready_before_a_body_migrates() -> None:
    # Tests before claims: ready flips with a migrated, silicon-checked body.
    assert [spec.qualname for spec in OPS if spec.ready] == []


def test_left_without_new_brief_stays_unpinned() -> None:
    for qualname in LEFT_WITHOUT_NEW_BRIEF:
        spec = find_spec(qualname)
        assert (spec.params, spec.tensors, spec.scratch) == ((), (), ())


def test_fa_variant_ranks_hold_on_every_dot_slot() -> None:
    spec = find_spec("attention.fa_fdot2")
    for arch in DOT_ARCHES:
        assert variant_ranks(spec, arch) == (0, 1)


def test_fa_sized_decode_plan() -> None:
    plan = fa_fdot2.plan(Caps(arch="gfx1030"), **DECODE)
    assert plan.meta["sized"] is True
    assert plan.ready is False
    assert plan.variant == 0
    assert plan.params["mode"] == FA_DECODE
    assert plan.params["causal"] == 1
    assert [(s.name, s.nbytes, s.offset) for s in plan.scratch_specs()] == [
        ("o_partial", 8 * 32 * 4 * 128 * 4, 0),
        ("m_partial", 8 * 32 * 4 * 4, 524288),
        ("l_partial", 8 * 32 * 4 * 4, 528384),
    ]
    assert all(spec.zeroed for spec in plan.scratch_specs())
    assert plan.scratch_nbytes == 532480
    assert plan.scratch_nbytes % SCRATCH_ALIGN == 0


def test_fa_prefill_without_split_k_has_no_scratch() -> None:
    plan = fa_fdot2.plan(Caps(), **{**DECODE, "mode": FA_PREFILL, "kv_splits": 1})
    assert plan.scratch_specs() == ()
    assert plan.scratch_nbytes == 0
    assert plan.variant == 1
    split = fa_fdot2.plan(Caps(), **{**DECODE, "mode": "prefill", "kv_splits": 2})
    assert [s.name for s in split.scratch_specs()] == ["o_partial", "m_partial", "l_partial"]


def test_fa_param_refusals_carry_the_c_status() -> None:
    with pytest.raises(V1Error, match="missing") as missing:
        fa_fdot2.plan(Caps(), head_dim=128)
    assert missing.value.status is V1Status.ERR_PARAM
    with pytest.raises(V1Error, match="unknown"):
        fa_fdot2.plan(Caps(), **DECODE, gqa_mode="subgroup")
    with pytest.raises(V1Error, match="not in"):
        fa_fdot2.plan(Caps(), **{**DECODE, "mode": "persist"})
    with pytest.raises(V1Error, match="multiple"):
        fa_fdot2.plan(Caps(), **{**DECODE, "num_q_heads": 12})
    with pytest.raises(V1Error, match="got a bool"):
        fa_fdot2.plan(Caps(), **{**DECODE, "kv_splits": True})
    with pytest.raises(ValueError, match="unsupported"):
        fa_fdot2.plan(Caps(dtype="bf16"), **DECODE)  # no fdot2.bf16
    from hippihx.sequence import causal_conv

    with pytest.raises(V1Error, match="unpinned") as unpinned:
        causal_conv.plan(Caps(), state_len=4)
    assert unpinned.value.status is V1Status.ERR_PARAM


def test_bind_names_follow_the_slots() -> None:
    plan = fa_fdot2.plan(Caps(), **DECODE)
    bound = fa_fdot2.bind(plan, scratch=None, q="q", k_cache="k", v_cache="v", out="o")
    assert set(bound.tensors) == {"q", "k_cache", "v_cache", "out"}
    with pytest.raises(ValueError, match="no tensor slot"):
        fa_fdot2.bind(plan, scratch=None, q="q", k="k", v="v")


def test_dot_refuses_wave64() -> None:
    assert not fa_fdot2.is_supported(Caps(arch="gfx1030", wave=64))
    from hippihx.sequence import causal_conv

    assert causal_conv.is_supported(Caps(arch="gfx1030", wave=64))
    assert causal_conv.is_supported(Caps(arch="gfx900"))


def test_comm_sized_plan_keys_on_the_fabric() -> None:
    pix = Fabric(hop="pix", switch="88096")
    plan = pcie.plan(Caps(fabric=pix), max_bytes=65536)
    assert plan.meta["custom_ar"] is True
    assert plan.params == {"max_bytes": 65536}
    for fabric in (None, Fabric(hop="phb"), Fabric(hop="pxb"), Fabric(hop="pix", switch="8749")):
        with pytest.raises(V1Error) as refused:
            pcie.plan(Caps(fabric=fabric), max_bytes=65536)
        assert refused.value.status is V1Status.ERR_UNSUPPORTED_FABRIC
        # The contract plan (no params) stays descriptive: RCCL fallback.
        assert pcie.plan(Caps(fabric=fabric)).meta["fallback"] == "rccl"


def test_arena_layout_packs_aligned_blocks() -> None:
    offsets, total = arena_layout([532480, 0, 100, 4096])
    assert offsets == (0, 532480, 532480, 532736)
    assert total == 536832
    assert all(offset % SCRATCH_ALIGN == 0 for offset in offsets)
    assert arena_layout([]) == ((), 0)
    with pytest.raises(ValueError):
        arena_layout([-1])
