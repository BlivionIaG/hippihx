# Consume hippihx from `rdna_extras`

hippihx is a library. Serve wiring stays in
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Do not edit that fork from this tree.

## One V1 entry per op

C ABI: `include/hippihx/v1.h` (rev **4**). Its enums and tables are
generated from `hippihx/_lib/catalog.py`.

| Call | When |
|---|---|
| `hippihx_v1_plan(op, &caps, params, nparams, &plan)` | Host, **before capture**, once per capture bucket and layer config. Checks caps and params, sizes scratch, picks the explore variant, reports `plan.ready`. |
| `hippihx_v1_run(&plan, tensors, ntensors, scratch, nbytes, stream)` | Capture-safe enqueue. Checks tensors against the op's slots and scratch against the plan. Never allocates, never D2H. `NOT_READY` until a body migrates. |

- `plan.ready == 0` means keep the serve path. Decide it at plan time,
  never under capture. Every op is `ready == 0` today.
- Params are host scalars, and a captured graph bakes them in. Lengths,
  block tables and query offsets are tensors the kernel reads on device.
- `plan.scratch_nbytes` is one block. Each spec has an aligned `offset`
  inside it. Zero it once for page-commit. Several plans can share one
  arena (`hippihx.v1.arena_layout`, `HIPPIHX_V1_SCRATCH_ALIGN`).
- `caps.fabric` carries hop + switch. A comm plan that is not PIX +
  ACS-clear + large-BAR on 88096 fails with `UNSUPPORTED_FABRIC`, and
  serve keeps RCCL.
- `hippihx_v1_plan_t` is plain data: cache it per bucket, never edit it.
  A failed plan is zeroed and `run` refuses it (`ERR_ABI`).

Pinned schemas. Every other op is unpinned (`nparams == 0`,
`ntensors == 0`) until its body migrates:

| Op | Params | Tensor slots | Scratch |
|---|---|---|---|
| `attention.fa_fdot2` | `mode` (decode / prefill), `head_dim` (128 / 256), `num_q_heads`, `num_kv_heads`, `block_size`, `kv_splits` (1–16), `sliding_window`, `causal`, `max_tokens`, `scale` | `q`, `k_cache`, `v_cache`, `block_table`, `seq_lens`, `cu_query_lens` (prefill), `out` | fp32 `o/m/l_partial` for decode and prefill split-K |
| `comm.pcie` | `max_bytes` (≤ `AR_MAX_KB`) | unpinned | none |

C++ (extras side, sketch):

```cpp
hippihx_v1_caps caps{"gfx1030", 32, HIPPIHX_V1_DTYPE_FP16, nullptr};
hippihx_v1_param p[HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS] = {};
p[HIPPIHX_V1_ATTN_FA_FDOT2_P_MODE].i = HIPPIHX_V1_ATTN_FA_FDOT2_MODE_DECODE;
p[HIPPIHX_V1_ATTN_FA_FDOT2_P_HEAD_DIM].i = 128;
// ... the other _P_* slots ...
hippihx_v1_plan_t plan{};
plan.struct_size = sizeof(plan);
const int rc = hippihx_v1_plan(HIPPIHX_V1_OP_ATTN_FA_FDOT2, &caps, p,
                               HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS, &plan);
if (rc != HIPPIHX_V1_OK || !plan.ready) {
  // serve fallback, chosen before capture
}
// Per call, eager or under capture:
hippihx_v1_tensor t[HIPPIHX_V1_ATTN_FA_FDOT2_NTENSORS] = {};  // _T_* order
hippihx_v1_run(&plan, t, HIPPIHX_V1_ATTN_FA_FDOT2_NTENSORS, scratch,
               plan.scratch_nbytes, stream);
```

Python (torch-free, same contracts). A sized plan refuses what the C plan
refuses (`V1Error.status` is the C code). `plan(caps)` without params is
a contract plan: no scratch, `meta["sized"]` false.

```python
from hippihx.attention import fa_fdot2

caps = fa_fdot2.Caps(arch="gfx1030", wave=32, dtype="fp16")
plan = fa_fdot2.plan(
    caps, mode="decode", head_dim=128, num_q_heads=32, num_kv_heads=8,
    block_size=16, kv_splits=4, sliding_window=0, causal=True,
    max_tokens=8, scale=128 ** -0.5,
)
# extras: scratch = torch.zeros(plan.scratch_nbytes, dtype=torch.uint8, device="cuda")
binding = fa_fdot2.bind(plan, scratch=scratch, q=q, k_cache=k_cache,
                        v_cache=v_cache, block_table=block_table,
                        seq_lens=seq_lens, out=out)
fa_fdot2.run(binding)
```

Register **one** `torch.ops.hippihx.<op>` that calls `hippihx_v1_run`.
No Triton→HIP double-fire. Bind keys on **arch + wave**, not GFX name
alone. extras V1 consume is the HIP fatbin. Dest extras tip
`330b42abb5b0` still has no `torch.ops.hippihx` bind. Dest extras **#27**
(`a3f7e5da`) is opt-in `VLLM_HIPPIHX` ctypes consume via
`hippihx.v1_ctypes` + `hippihx_v1_load` (default **off**; every plan
still not ready). FlyDSL extras consume waits on `FLYDSL_V1_CONSUME`
(graph-safe JIT) — see [`FLYDSL.md`](FLYDSL.md).

## Host contracts already here

Import these instead of reimplementing extras bugs:

- `hippihx.gemm.w4a16_fdot2.pack` — integer ZP, `K_STEP=32`, refuse ConfigH
  and 40-wide K-splits.
- `hippihx.comm.fabric.Fabric` + `hippihx.comm.pcie.policy` — PIX vs
  PHB/PXB, 88096 vs 8749, Uncached+push, `AR_MAX_KB=512`. extras PIX
  `lspci` helpers stay serve.
- `Caps.dtype` — DOT and GDN HIP refuse bf16 (`HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`).

## Artifacts

One configure tree per slot writes to `build/fatbin/<arch>/`:

| File | What | Serve does |
|---|---|---|
| `libhippihx_v1.so` | V1 symbols + generated tables. No device code, so the same for every slot. Links `libamdhip64` when built with HIP | `dlopen` once |
| `hippihx_<arch>.hsaco` | this slot's device code: one raw AMDGPU ELF (one `--offload-arch`, `--cuda-device-only`, `--no-gpu-bundle-output`), from `tiles/code_object.hip` | `hippihx_v1_load(arch, path)` at init, before any plan or capture |
| `libhippihx_<arch>.a` | link smoke (`hippihx_smoke_host`) | nothing |

`hippihx_v1_load` checks the code object before `hipModuleLoadData` sees
it. It refuses a slot that is not built (`UNSUPPORTED_ARCH`, so gfx1013
and gfx906 never load), `HSA_OVERRIDE_GFX_VERSION` in the environment, or
an ELF whose `EF_AMDGPU_MACH` is not the slot's (`FOREIGN_ISA`). It also
refuses anything that is not one raw AMDGPU ELF, such as offload bundles
and `.o` files (`CODE_OBJECT`). `plan.ready` also needs the slot loaded.
Tile entries are `extern "C"`, so a migrated body resolves its kernels by
plain name. A host-stub build validates and returns `NO_HIP`.

Python mirror of the check: `hippihx._lib.codeobject.check_code_object`.

## After extras rewires

Delete the extras `.cu` copy. ISA lives here. Capture arenas,
keepalive, env flags, FULL→PIECEWISE stay extras. extras already
page-commits with `new_zeros` / `zeros_like` — that is the serve half
of `plan` scratch.
