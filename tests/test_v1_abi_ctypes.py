"""The compiled C V1 table (tiles/v1_abi.cpp) against the Python mirror.

Builds the torch-free V1 host source as a shared object with the host C++
compiler and calls it through ctypes. Skipped when no compiler is found.
No device, no HIP, no tok/s.
"""

from __future__ import annotations

import ctypes
import itertools
import math
from pathlib import Path

import pytest

import hippihx
from hippihx._lib import v1_ctypes as cv
from hippihx._lib.catalog import FA_DECODE, FA_PREFILL, OPS, find_spec
from hippihx._lib.fabric import AR_MAX_KB, HOPS, SWITCHES, Fabric
from hippihx._lib.fatbin import KNOWN_ARCHES, supported_arches
from hippihx._lib.isa import wave_occupancy_ok
from hippihx._lib.protocol import Caps, dtype_supported, dtype_to_v1
from hippihx._lib.v1 import (
    ABI_REVISION,
    SCRATCH_ALIGN,
    V1_DTYPES,
    V1Error,
    V1Status,
)
from hippihx.comm import pcie
from hippihx.comm.pcie.policy import custom_ar_ok

ROOT = Path(__file__).resolve().parents[1]
FA = find_spec("attention.fa_fdot2")
COMM = find_spec("comm.pcie")
FA_ID = FA.v1_id

FA_DECODE_PARAMS = {
    "mode": FA_DECODE,
    "head_dim": 128,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "block_size": 16,
    "kv_splits": 4,
    "sliding_window": 0,
    "causal": 1,
    "max_tokens": 8,
    "scale": 1 / math.sqrt(128),
}
PIX = Fabric(hop="pix", switch="pex88096")


@pytest.fixture(scope="module")
def lib(v1_lib: ctypes.CDLL) -> ctypes.CDLL:
    return v1_lib


def c_plan(
    lib: ctypes.CDLL,
    op: int,
    arch: str = "gfx1030",
    *,
    wave: int = 0,
    dtype: int = 0,
    fabric: cv.Fabric | None = None,
    params: dict | None = None,
    nparams: int | None = None,
) -> tuple[int, cv.Plan]:
    spec = OPS[op]
    caps = cv.Caps(arch=arch.encode(), wave=wave, dtype=dtype)
    if fabric is not None:
        caps.fabric = ctypes.pointer(fabric)
    array = cv.params_array(spec, params) if params is not None else None
    count = len(spec.params) if params is not None else 0
    plan = cv.new_plan()
    rc = lib.hippihx_v1_plan(
        op, ctypes.byref(caps), array, count if nparams is None else nparams, ctypes.byref(plan)
    )
    return rc, plan


def c_specs(plan: cv.Plan) -> list[tuple[str, int, int]]:
    return [
        (plan.scratch[i].name.decode(), plan.scratch[i].nbytes, plan.scratch[i].offset)
        for i in range(plan.nscratch)
    ]


def py_specs(plan: hippihx.Plan) -> list[tuple[str, int, int]]:
    return [(spec.name, spec.nbytes, spec.offset) for spec in plan.specs]


def py_status(fn) -> V1Status:
    try:
        fn()
    except V1Error as exc:
        return exc.status
    return V1Status.OK


def test_c_tables_follow_the_catalog(lib: ctypes.CDLL) -> None:
    assert lib.hippihx_v1_abi_revision() == ABI_REVISION == 4
    assert lib.hippihx_v1_op_count() == len(OPS)
    for spec in OPS:
        op = spec.v1_id
        assert lib.hippihx_v1_op_name(op).decode() == spec.qualname
        assert lib.hippihx_v1_op_is_dot(op) == int(spec.dot)
        assert lib.hippihx_v1_op_fp16_act(op) == int(spec.fp16_act)
        assert lib.hippihx_v1_op_nparams(op) == len(spec.params)
        assert lib.hippihx_v1_op_ntensors(op) == len(spec.tensors)
        names = [lib.hippihx_v1_op_param_name(op, i).decode() for i in range(len(spec.params))]
        assert names == [param.name for param in spec.params]
        slots = [lib.hippihx_v1_op_tensor_name(op, i).decode() for i in range(len(spec.tensors))]
        assert slots == [slot.name for slot in spec.tensors]
        assert lib.hippihx_v1_op_param_name(op, len(spec.params)) is None
        assert lib.hippihx_v1_op_tensor_name(op, -1) is None
    assert lib.hippihx_v1_op_name(len(OPS)) is None
    assert lib.hippihx_v1_op_nparams(-1) == -1


@pytest.mark.parametrize("spec", OPS, ids=lambda spec: spec.qualname)
def test_c_plan_matches_python_on_every_slot(lib: ctypes.CDLL, spec) -> None:
    op = hippihx.find_op(spec.qualname)
    module = __import__(f"hippihx.{spec.qualname}", fromlist=["plan"])
    params = {FA.qualname: FA_DECODE_PARAMS, COMM.qualname: {"max_bytes": 4096}}.get(
        spec.qualname
    )
    fabric = cv.fabric_struct(PIX) if spec.needs_fabric else None
    for arch, wave, dtype in itertools.product(
        KNOWN_ARCHES, (0, 32, 64), (None, "fp16", "bf16", "fp32")
    ):
        caps = Caps(arch=arch, wave=wave or None, dtype=dtype, fabric=PIX)
        if arch not in supported_arches(op.dot) or not wave_occupancy_ok(dot=op.dot, wave=caps.wave):
            want = V1Status.ERR_UNSUPPORTED_ARCH
        elif not dtype_supported(dot=op.dot, fp16_act=op.fp16_act, dtype=dtype):
            want = V1Status.ERR_UNSUPPORTED_DTYPE
        else:
            want = V1Status.OK
        assert module.is_supported(caps) is (want is V1Status.OK)
        rc, plan = c_plan(
            lib, spec.v1_id, arch, wave=wave, dtype=dtype_to_v1(dtype), fabric=fabric, params=params
        )
        assert rc == want, (arch, wave, dtype)
        if want is not V1Status.OK:
            assert plan.abi_revision == 0  # failed plans are zeroed
            continue
        py = module.plan(caps, **(params or {}))
        assert plan.abi_revision == ABI_REVISION
        assert plan.op == spec.v1_id
        assert plan.arch == KNOWN_ARCHES.index(arch)
        assert plan.wave == caps.wave
        assert plan.dtype == dtype_to_v1(dtype)
        assert plan.ready == int(py.ready) == 0
        assert plan.variant == py.variant
        assert plan.nparams == len(spec.params)
        assert c_specs(plan) == py_specs(py)
        assert plan.scratch_nbytes == py.scratch_nbytes
        assert all(plan.scratch[i].zeroed == 1 for i in range(plan.nscratch))


def fa_params(**overrides) -> dict:
    return {**FA_DECODE_PARAMS, **overrides}


@pytest.mark.parametrize(
    ("mode", "head_dim", "kv_splits", "max_tokens", "heads"),
    list(
        itertools.product(
            (FA_DECODE, FA_PREFILL),
            (128, 256),
            (1, 2, 16),
            (1, 7, 256, 16384),
            ((32, 8), (64, 64), (12, 4)),
        )
    ),
)
def test_fa_scratch_and_variant_parity(
    lib: ctypes.CDLL, mode: int, head_dim: int, kv_splits: int, max_tokens: int, heads
) -> None:
    from hippihx.attention import fa_fdot2

    q_heads, kv_heads = heads
    params = fa_params(
        mode=mode,
        head_dim=head_dim,
        kv_splits=kv_splits,
        max_tokens=max_tokens,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        scale=head_dim**-0.5,
    )
    rc, plan = c_plan(lib, FA_ID, params=params)
    status = py_status(lambda: fa_fdot2.plan(Caps(), **params))
    assert rc == status
    if status is not V1Status.OK:
        assert status is V1Status.ERR_PARAM  # int32 partial index bound
        assert max_tokens * q_heads * kv_splits * head_dim > 2**31 - 1
        return
    py = fa_fdot2.plan(Caps(), **params)
    assert c_specs(plan) == py_specs(py)
    assert plan.scratch_nbytes == py.scratch_nbytes
    assert plan.variant == py.variant == (0 if mode == FA_DECODE else 1)
    split_k = mode == FA_DECODE or kv_splits > 1
    assert [name for name, _, _ in c_specs(plan)] == (
        ["o_partial", "m_partial", "l_partial"] if split_k else []
    )
    for _, _, offset in c_specs(plan):
        assert offset % SCRATCH_ALIGN == 0
    assert plan.scratch_nbytes % SCRATCH_ALIGN == 0


def test_fa_int32_partial_bound_is_exact(lib: ctypes.CDLL) -> None:
    from hippihx.attention import fa_fdot2

    edge = (2**31 - 1) // 128  # T * 1 * 1 * 128 <= 2^31 - 1
    base = fa_params(num_q_heads=1, num_kv_heads=1, kv_splits=1)
    assert c_plan(lib, FA_ID, params={**base, "max_tokens": edge})[0] == V1Status.OK
    fa_fdot2.plan(Caps(), **{**base, "max_tokens": edge})
    assert c_plan(lib, FA_ID, params={**base, "max_tokens": edge + 1})[0] == V1Status.ERR_PARAM
    with pytest.raises(V1Error, match="index width"):
        fa_fdot2.plan(Caps(), **{**base, "max_tokens": edge + 1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_dim": 64},
        {"kv_splits": 0},
        {"kv_splits": 17},
        {"num_q_heads": 12, "num_kv_heads": 8},
        {"scale": 0.0},
        {"scale": -1.0},
        {"scale": math.inf},
        {"scale": math.nan},
        {"causal": 2},
        {"mode": 2},
        {"sliding_window": -1},
        {"max_tokens": 0},
        {"num_kv_heads": 0},
    ],
)
def test_fa_param_refusals_match(lib: ctypes.CDLL, overrides: dict) -> None:
    from hippihx.attention import fa_fdot2

    params = fa_params(**overrides)
    assert c_plan(lib, FA_ID, params=params)[0] == V1Status.ERR_PARAM
    assert py_status(lambda: fa_fdot2.plan(Caps(), **params)) is V1Status.ERR_PARAM


def test_param_count_and_null_params(lib: ctypes.CDLL) -> None:
    assert c_plan(lib, FA_ID, params=FA_DECODE_PARAMS, nparams=9)[0] == V1Status.ERR_PARAM
    assert c_plan(lib, FA_ID)[0] == V1Status.ERR_PARAM  # pinned op needs its params
    caps = cv.Caps(arch=b"gfx1030")
    plan = cv.new_plan()
    rc = lib.hippihx_v1_plan(FA_ID, ctypes.byref(caps), None, 10, ctypes.byref(plan))
    assert rc == V1Status.ERR_BAD_ARG
    conv = find_spec("sequence.causal_conv").v1_id
    assert c_plan(lib, conv, params=None, nparams=0)[0] == V1Status.OK
    assert lib.hippihx_v1_plan(99, ctypes.byref(caps), None, 0, ctypes.byref(plan)) == (
        V1Status.ERR_UNKNOWN_OP
    )
    assert lib.hippihx_v1_plan(conv, None, None, 0, ctypes.byref(plan)) == V1Status.ERR_BAD_ARG
    assert lib.hippihx_v1_plan(conv, ctypes.byref(caps), None, 0, None) == V1Status.ERR_BAD_ARG
    for arch in (b"", b"gfx1030,gfx1100", b"gfx906", b"gfx1013", b"gfx1200"):
        caps = cv.Caps(arch=arch)
        assert lib.hippihx_v1_plan(conv, ctypes.byref(caps), None, 0, ctypes.byref(plan)) == (
            V1Status.ERR_UNSUPPORTED_ARCH
        )


def test_struct_size_mismatch_and_failed_plan(lib: ctypes.CDLL) -> None:
    caps = cv.Caps(arch=b"gfx1030")
    plan = cv.new_plan()
    plan.struct_size = 8
    plan.variant = 7
    conv = find_spec("sequence.causal_conv").v1_id
    assert lib.hippihx_v1_plan(conv, ctypes.byref(caps), None, 0, ctypes.byref(plan)) == (
        V1Status.ERR_ABI
    )
    assert plan.variant == 7  # header / library mismatch: plan untouched
    rc, failed = c_plan(lib, FA_ID, params=fa_params(head_dim=64))
    assert rc == V1Status.ERR_PARAM
    assert failed.abi_revision == 0 and failed.ready == 0 and failed.nscratch == 0
    assert lib.hippihx_v1_run(ctypes.byref(failed), None, 0, None, 0, None) == V1Status.ERR_ABI
    assert lib.hippihx_v1_run(None, None, 0, None, 0, None) == V1Status.ERR_BAD_ARG


def test_comm_fabric_parity(lib: ctypes.CDLL) -> None:
    comm = COMM.v1_id
    for hop, switch, acs, bar in itertools.product(HOPS, SWITCHES, (True, False), (True, False)):
        try:
            fabric = Fabric(hop=hop, switch=switch, acs_clear=acs, large_bar=bar)
        except ValueError:
            # PIX without ACS clear is not a hop. C: malformed fabric.
            bad = cv.Fabric(
                hop=cv.V1_HOPS[hop],
                pcie_switch=cv.V1_SWITCHES[switch],
                width=16,
                acs_clear=int(acs),
                large_bar=int(bar),
            )
            assert c_plan(lib, comm, fabric=bad, params={"max_bytes": 4096})[0] == (
                V1Status.ERR_BAD_ARG
            )
            continue
        rc, _ = c_plan(lib, comm, fabric=cv.fabric_struct(fabric), params={"max_bytes": 4096})
        caps = Caps(fabric=fabric)
        assert rc == py_status(lambda: pcie.plan(caps, max_bytes=4096))
        assert (rc == V1Status.OK) is custom_ar_ok(fabric, nbytes=4096)
        assert (rc == V1Status.OK) is pcie.plan(caps).meta["custom_ar"]
    assert c_plan(lib, comm, params={"max_bytes": 4096})[0] == V1Status.ERR_UNSUPPORTED_FABRIC
    over = {"max_bytes": AR_MAX_KB * 1024 + 1}
    assert c_plan(lib, comm, fabric=cv.fabric_struct(PIX), params=over)[0] == V1Status.ERR_PARAM
    assert py_status(lambda: pcie.plan(Caps(fabric=PIX), **over)) is V1Status.ERR_PARAM
    for field, value in (("hop", 0), ("hop", 9), ("pcie_switch", 0), ("width", 3), ("acs_clear", 2)):
        bad = cv.fabric_struct(PIX)
        setattr(bad, field, value)
        assert c_plan(lib, comm, fabric=bad, params={"max_bytes": 4096})[0] == (
            V1Status.ERR_BAD_ARG
        ), field


class Scratch:
    """Aligned host bytes. run only checks the pointer; it never reads it."""

    def __init__(self, nbytes: int) -> None:
        self.raw = ctypes.create_string_buffer(nbytes + 2 * SCRATCH_ALIGN)
        base = ctypes.addressof(self.raw)
        self.ptr = base + (-base) % SCRATCH_ALIGN


FP16, I32, BF16 = V1_DTYPES["fp16"], V1_DTYPES["i32"], V1_DTYPES["bf16"]
ADDR = 0x10000  # run never dereferences device pointers


def fa_tensors() -> list[cv.Tensor]:
    return [
        cv.tensor(ADDR, FP16, (8, 32, 128)),  # q
        cv.tensor(ADDR, FP16, (100, 8, 16, 16, 8)),  # k_cache [blocks, H_kv, D/x, bs, x]
        cv.tensor(ADDR, FP16, (100, 8, 16, 16, 8)),  # v_cache
        cv.tensor(ADDR, I32, (8, 64), (80, 1)),  # block_table, padded rows
        cv.tensor(ADDR, I32, (8,)),  # seq_lens
        cv.tensor(0, I32, (1,)),  # cu_query_lens: unused in decode
        cv.tensor(ADDR, FP16, (8, 32, 128)),  # out
    ]


def c_run(lib, plan, tensors, scratch_ptr, nbytes, ntensors=None) -> int:
    array = (cv.Tensor * max(1, len(tensors)))(*tensors)
    count = len(tensors) if ntensors is None else ntensors
    return lib.hippihx_v1_run(ctypes.byref(plan), array, count, scratch_ptr, nbytes, None)


def test_run_checks_fa_tensors_then_not_ready(lib: ctypes.CDLL) -> None:
    rc, plan = c_plan(lib, FA_ID, params=FA_DECODE_PARAMS)
    assert rc == V1Status.OK and plan.scratch_nbytes > 0
    scratch = Scratch(plan.scratch_nbytes)
    good = fa_tensors()
    assert c_run(lib, plan, good, scratch.ptr, plan.scratch_nbytes) == V1Status.ERR_NOT_READY

    def mutated(slot: int, **change) -> list[cv.Tensor]:
        tensors = fa_tensors()
        t = tensors[slot]
        shape = change.pop("shape", None)
        stride = change.pop("stride", None)
        if shape is not None or stride is not None:
            tensors[slot] = cv.tensor(
                t.data, t.dtype, shape or list(t.shape[: t.rank]), stride
            )
        for key, value in change.items():
            setattr(tensors[slot], key, value)
        return tensors

    bad_cases = {
        "q bf16": mutated(0, dtype=BF16),
        "q rank 2": mutated(0, rank=2),
        "q rows > max_tokens": mutated(0, shape=(9, 32, 128)),
        "q heads != H_q": mutated(0, shape=(8, 16, 128)),
        "q head_dim != D": mutated(0, shape=(8, 32, 256)),
        "q not contiguous": mutated(0, stride=(8192, 256, 1)),
        "q extent 0": mutated(0, shape=(0, 32, 128)),
        "k_cache H_kv": mutated(1, shape=(100, 4, 16, 16, 8)),
        "k_cache block_size": mutated(1, shape=(100, 8, 16, 32, 8)),
        "k_cache negative stride": mutated(1, stride=(-1, 1, 1, 1, 1)),
        "block_table inner stride": mutated(3, stride=(128, 2)),
        "block_table rows overlap": mutated(3, stride=(32, 1)),
        "seq_lens i64": mutated(4, dtype=V1_DTYPES["i64"]),
        "out NULL": mutated(6, data=0),
    }
    for name, tensors in bad_cases.items():
        assert c_run(lib, plan, tensors, scratch.ptr, plan.scratch_nbytes) == (
            V1Status.ERR_TENSOR
        ), name

    # A size-1 axis carries no stride contract (torch is_contiguous).
    one_row = mutated(0, shape=(1, 32, 128), stride=(123456, 128, 1))
    assert c_run(lib, plan, one_row, scratch.ptr, plan.scratch_nbytes) == V1Status.ERR_NOT_READY

    assert c_run(lib, plan, good, scratch.ptr, plan.scratch_nbytes, ntensors=6) == (
        V1Status.ERR_TENSOR
    )
    assert lib.hippihx_v1_run(
        ctypes.byref(plan), None, len(good), scratch.ptr, plan.scratch_nbytes, None
    ) == V1Status.ERR_BAD_ARG
    assert c_run(lib, plan, good, scratch.ptr, plan.scratch_nbytes - 1) == V1Status.ERR_SCRATCH
    assert c_run(lib, plan, good, None, plan.scratch_nbytes) == V1Status.ERR_SCRATCH
    assert c_run(lib, plan, good, scratch.ptr + 8, plan.scratch_nbytes) == V1Status.ERR_SCRATCH


def test_prefill_requires_cu_query_lens(lib: ctypes.CDLL) -> None:
    rc, plan = c_plan(lib, FA_ID, params=fa_params(mode=FA_PREFILL, kv_splits=1))
    assert rc == V1Status.OK and plan.scratch_nbytes == 0 and plan.variant == 1
    tensors = fa_tensors()
    assert c_run(lib, plan, tensors, None, 0) == V1Status.ERR_TENSOR  # NULL cu_query_lens
    tensors[5] = cv.tensor(ADDR, I32, (9,))
    assert c_run(lib, plan, tensors, None, 0) == V1Status.ERR_NOT_READY


def test_unpinned_op_takes_no_tensors(lib: ctypes.CDLL) -> None:
    conv = find_spec("sequence.causal_conv").v1_id
    rc, plan = c_plan(lib, conv, dtype=V1_DTYPES["bf16"])
    assert rc == V1Status.OK and plan.nscratch == 0 and plan.scratch_nbytes == 0
    assert lib.hippihx_v1_run(ctypes.byref(plan), None, 0, None, 0, None) == V1Status.ERR_NOT_READY
    one = (cv.Tensor * 1)(cv.tensor(ADDR, FP16, (4,)))
    assert lib.hippihx_v1_run(ctypes.byref(plan), one, 1, None, 0, None) == V1Status.ERR_TENSOR


def test_activation_dtype_codes_are_the_v1_codes() -> None:
    for name in ("fp16", "bf16", "fp32"):
        assert dtype_to_v1(name) == V1_DTYPES[name]
