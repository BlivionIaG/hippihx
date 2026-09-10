# extras → hippihx backport review

Last checked `opengfx1030/vllm-rdna` `rdna_extras` @ `a4060647cfbb`
(2026-09-08 21:57 UTC; re-checked **2026-09-10**). Tip **unchanged**.
Open extras PRs **#2** / **#3** did not move. **Do not copy kernel bodies
into this tree yet.**

Dest extras **rolled back** several serve/kernel defaults (GDN HIP
prefill default-off, spec/MTP gate churn, capture experiments). A
Flash-Next V620 integration fork of that tip then had to roll back
**more**: `fdot2.bf16` ISel aborts, unaligned W4 K-splits, scale-baked
W4 zero-points, GDN HIP selected on BF16. hippihx locks the *proper*
consume contracts so a later migrate does not copy those defects.
See [Dest extras defects](#dest-extras-defects-do-not-copy).

**Consume ABI started.** `include/hippihx/v1.h` + `tiles/v1_abi.cpp` ship
the torch-free C entry family (`hippihx_v1_plan` / `hippihx_v1_run`) that
extras will wrap as one `torch.ops.hippihx.*` per op. `hippihx_v1_run`
returns `HIPPIHX_V1_ERR_NOT_READY` until a body migrates. This is migrate
criterion **#3** (one HIP entry extras can bind) — not a body dump.

Delta since `d71721c79547`: dest extras is **one commit ahead**. GitHub
squash-merged [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
(`rdna_ar` Uncached+push). PRs **#2** (`later/glm53-flash-awq-glm5next`
@ `4b31f2eede51`) and **#3** (a17t WIP, last unique HIP still
`4b9badd0a97d`) did not move. `main` is unrelated upstream vLLM
(`c00091e02670`). Issues are disabled.

The PR title said review-only / not dest. Dest extras **did** land the
sources (`csrc/rocm/rdna_allreduce.{cu,cuh}` + communicator hook). That
is dest *presence*, not dest-on-by-default and not a hippihx migrate:
`os.getenv("VLLM_RDNA_AR", "0") == "1"` stays opt-in; occupancy pin
stays closed; `rdna_allreduce.cu` is still ATen/`torch/all.h`. The
`.py` module docstring still says “enabled by default”; the
communicator gate is the dest contract — do not treat the docstring as
dest-on. Pick **Aron Hsiao** unique commits from
`cursor/leapdragon-cherry-d2a4` (or leapdragon source), **not** squash
`a4060647` (Author BlivionIaG, `Co-authored-by` Aron / Claude Fable 5).

hippihx is the zoo. extras is serve wiring. Copying ATen-coupled
`csrc/rocm/*.cu` here would freeze a second ISA copy while dest is still
hunting GDN capture bugs, with no consume path (`torch.ops.hippihx.*`
does not exist). Dual copies are how serve bugs accrete.

## Verdict

| Source | Pertinent to hippihx? | Action |
|---|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Yes, later** — zoo class | Wait. Record observed locks. **Do not copy dest W4 ZP / K-split / bf16 DOT.** Migrate only when extras can *call* hippihx. |
| Dest tip 2026-09-06…08 (cudagraph zeros, eager_break, spec/MTP gates, GDN probes, profiling) | **No** | Stay in extras. That is page-commit / capture / dispatcher work. |
| Dest tip leapdragon `rdna_ar` (merged PR #1 @ `a4060647`) | **Later** — `comm.pcie` | On dest extras, default **off**, occupancy pin closed. Do not dump the ATen wrapper. |
| PR #2 GLM-5.3 KDA/DSA (`later/glm53-…`) | **Later** — `kda_scan` / `dsa_nope` / `qsa_indexer` | Product-named `glm5_*` files. Do not name tiles after GLM. |
| PR #3 a17t `[WIP] Similar work, different fork` | **No** | WIP, untested, mixed Triton+HIP+qwen4_exp serve. Duplicate W4 family. |

## Why bodies stay in extras for now

1. Every dest HIP file includes `torch/all.h` + ATen CUDAGuard. hippihx
   tiles are torch-free `plan` / `bind` / `run` stubs. A dump is not a
   migrate.
2. Scaffold non-goal: do not land production FA / EXL3 / AWQ bodies
   until LDS / `__launch_bounds__` are locked *here*.
3. UNC-26 (EXL3) is In Progress. FA occupancy pin stays closed.
4. GDN HIP prefill is default-off (chunk-boundary corruption). GDN hybrid
   + piecewise cudagraph still produces garbage on dest
   (`docs/profiling/2026-09-08-cudagraph-gdn-root-cause.md`). Eager is
   correct; capture is not. `causal_conv1d_update` is implicated as a
   captured non-splitting op — serve bug, not a missing zoo file.
5. CONTRIBUTING: do not edit `opengfx1030/vllm-rdna` from this repo.
   Without a hippihx consume bind, a copy here cannot replace extras.

## Dest file → tile (when migrate is allowed)

Observed at `d71721c79547` (ISA files) plus dest `rdna_ar` @
`a4060647cfbb`. Numbers are extras *observations*, not dest locks.
Fill tile READMEs; do not invent tok/s.

| extras path | hippihx tile | Notes |
|---|---|---|
| `csrc/rocm/fa_rdna2.cu` | `attn/fa_fdot2` | ~33 KiB decode / ~48 KiB prefill smem. Occupancy pin closed. |
| `csrc/rocm/gdn_decode_rdna2.cu` | `attn/gdn_scan` | Register-resident 16 VGPR/thread, no LDS. `NULL_BLOCK_ID=0` is a **vLLM sentinel** — zoo contract is “invalid slot → zero out, do not touch state”, not that constant. |
| `csrc/rocm/gdn_prefill_*_rdna2.cu` | `attn/gdn_scan` | `o` kernel LDS ≈ 45312 B. Prefill chain default-off on dest. |
| `csrc/rocm/q_gemm_rdna2.cu` + `q_gemm_rdna2_{prefill,awq_prefill}.cu` + `qdq_4_rdna2.cuh` | `gemm/w4a16_fdot2` | GPTQ and AWQ are pack/zeros modes, **one** GEMM family. `LDS_PAD=8`. High-M AWQ uses the exllama-clone prefill tile. |
| `csrc/rocm/moe_q_gemm_rdna2.cu` | `moe/routed` | Reuses W4 helpers. |
| `csrc/rocm/exl3_dot2_{dense,moe,dequant,hadamard}.*` | `gemm/exl3_3inst` | `LDS_PAD=8` on A-staging. No `__launch_bounds__` on DOT kernels (VGPR is the occupancy lever). Produce stays `-cb 3inst` **outside**. |
| `csrc/rocm/causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, wave32, `state_len≈3–4`, dim multiple of 32, register-only (no LDS). |
| `csrc/rocm/indexer_paged_mqa_rdna2.cu` | `attn/qsa_indexer` | Dest indexer; confirm class vs DSA later. |
| `csrc/rocm/sparse_mla_rdna2.cu` | `attn/dsa_nope` | Sparse MLA class, not a product fuse. |
| `csrc/rocm/rdna_allreduce.{cu,cuh}` (merged PR #1) | `comm/pcie` | Uncached+push Later. Host-coherent flags. Boot self-test. Default off. Occupancy pin closed. INT8/Q8 wire preferred; no Finegrained. |
| `later/glm53-…` `glm5_kda_*.cu` | `attn/kda_scan` | Later. Do not keep the `glm5_` prefix. Do not retarget GDN 16/48 onto KDA 64×128. |
| `later/glm53-…` `glm5_dsa_*.cu` | `attn/dsa_nope` + `qsa_indexer` | Later. Same rename rule. |

## Stay in extras (not tiles)

| extras change | Why |
|---|---|
| `torch::zeros` / `torch.zeros` for FA `O` / `O_partial` / `M_partial`, GDN scratch, EXL3 buffers | Caller page-commit. Already an engine bind rule. |
| `eager_break_during_capture` (GDN, `do_kv_cache_update`) | Capture dispatcher. |
| RDNA_ATTN spec / MTP verify gates | Serve policy. |
| `VLLM_LOG_GDN_PTRS`, `VLLM_GDN_DBG`, step timing | Probes. No D2H under capture in the zoo. |
| HIP AOT RMSNorm `forward_rocm` | No tile class. Leave fused norms in extras until a class exists. |
| Platform gates, `_rocm_C` vs `load_inline` | Registration. |
| dest `rdna_all_reduce.py` boot self-test + `CudaCommunicator` hook | Serve. Zoo is the Uncached+push tile only. |
| `sync_remote_to_local.sh`, recipes, profiling docs | Serve / ops. |
| gfx1100 WMMA W4 (`q_gemm_rdna3_wmma.cu`) | WMMA is a gfx110x-only Later overlay, never on shared DOT. |
| MXFP4 / W8A16 FP8 / skinny GEMMs | Extra consume families. Add a tile only when dest locks one as dest. |

## Do not take from a17t PR #3

Unique vs dest: `awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`,
`gemv_w4_kpack_rdna2.cu`, `qdq_awq_rdna2.cuh`. Dest already has AWQ as a
**zeros mode** on `q_gemm_rdna2*` plus `q_gemm_rdna2_awq_prefill.cu`.
hippihx must not grow a second W4 GEMM. `qwen4_exp`, Triton fallbacks,
and serve recipes are extras (or another engine), not zoo.

Dest already dropped leapdragon `gemv_f16_rdna2` / `moe_skinny_int4_decode`
at PR #1 conflict resolution. Do not reintroduce skinny GEMV as a tile
without an extras dest lock.

## When a migrate *is* pertinent

All of:

1. Dest GDN hybrid is capture-safe (or GDN HIP stays opt-in and documented).
2. Tile README LDS / `__launch_bounds__` / wave / DOT unit are filled from
   extras observations (this review started that; EXL3 grain-v2 + FA
   `__launch_bounds__` 128/256 recorded @ `a4060647`).
3. hippihx ships one HIP entry that extras can bind as one V1 op.
   **Started:** `include/hippihx/v1.h` (`hippihx_v1_plan` / `hippihx_v1_run`).
   `run` is still `NOT_READY` (no body). Extras has not rewired yet.
4. extras is rewired to call it (that edit happens **in extras**, not from
   this tree). The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs, grow the V1 ABI. Kernel /
mode / env / AR tracker (all **unvalidated**):
[`README.md`](../README.md#unvalidated-extras-inventory).

## Dest extras defects (do not copy)

Dest tip is still `a4060647`. These are **live dest bugs / rolled-back
paths**, not tok/s. hippihx contracts must not reproduce them.

| Defect | Where on dest extras | Proper zoo lock | hippihx |
|---|---|---|---|
| `llvm.amdgcn.fdot2.bf16.bf16` ISel abort | Triton/fusion on BF16 vision interp, `triton_mrope`, BF16 causal conv. Dest HIP conv is fp16-oriented. Integration rolled BF16 back to **fp32-promoted scalar FMA** / PyTorch ref. | Never `fdot2.bf16`. DOT + GDN HIP are **fp16 activations**. BF16 leftover / conv = scalar FMA, fp32 mul. | `dot.hpp`, V1 `HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`, `sequence/causal_conv`, `moe/leftover_bf16` |
| Scale-baked W4 zero-point | `qdq_4_rdna2.cuh` `prep_zero_scale_fp16`: `0xE400 \| zero` then `scale * (-1024 - zero)` in `half`. All-zero weights were not exact zero. | Integer `q - zero`, then `* scale`. GPTQ `uint4b8` (+1) vs AWQ literal is pack/zeros, not a second GEMM. | `gemm/w4a16_fdot2` |
| Unaligned prefill K-split | `q_gemm_rdna2_prefill.cu` `compute_split_k`: K=640 can pick 16 splits of 40. Kernel reads **32-value** tiles (`K_STEP=32`). | Equal `k_per_split`, multiple of 32, inside LDS budget. Refuse the 40-wide split. | `gemm/w4a16_fdot2` |
| GDN HIP selected on BF16 | `_gdn_prefill_dispatch_available()` checked GPU + symbols, **not dtype**. Kernel correctly rejected `mixed_qkv must be fp16`. | `plan` / V1 refuse bf16 on `attn.gdn_scan`. Prefill HIP stays default-off (chunk-boundary). Capture still unsafe. | `attn/gdn_scan`, V1 dtype |
| Global EXL3 FP16 clip | Serve `Qwen2MoeMLP` clipped every model to FP16. Rolled back to EXL3-only on the integration fork. | Not a tile. Stay in extras. | — |
| QSA 2-warp BF16 prefill spill | gfx1030 6h×256 BF16 prefill: 2 warps exhaust VGPRs. | **4 warps** (128 threads) for that shape. | `attn/qsa_indexer` |
| `rdna_ar` flags not beside receiver staging | Dest squash of PR #1. Donor `3cfe000` excluded (mixed PLE). | Later: flags beside each rank’s uncached staging. Opt-in. Do not pick the mixed commit. | `comm/pcie` |

Serve rollbacks already on dest tip (keep as serve, not zoo):

- GDN HIP prefill **default-off** (`VLLM_GDN_HIP_PREFILL=0`) — chunk-boundary corruption; HIP chain still exists.
- RDNA_ATTN spec/MTP verify **default-off** (opt-in env). Auto-skip when spec is off was **reverted**.
- `eager_break_during_capture` on `do_kv_cache_update` was reverted then **reapplied**.
- FA `kv_splits` 16: reverted to 0.27.1 form then that revert was reverted (16 stays).
- GDN piecewise-capture experiments documented as **disproven**; eager is the correct control. Do not copy probe-driven ISA.

Skinny GEMM (`skinny_gemms.cu`) exists on dest extras and was used for
gfx1030 BF16 decode on the Flash-Next fork. **No hippihx tile** until
dest extras locks that family as dest. Do not reintroduce leapdragon
`gemv_f16` dropped at extras PR #1.

## Attribution (do not re-author)

**Dest extras HIP** (FA, EXL3, W4A16, GDN, causal_conv, sparse MLA,
indexer, GLM Later KDA/DSA) is **BlivionIaG** `<kev29lt@gmail.com>`.
Author **and** Committer on every picked dest commit. Do not use Cursor
or `cursoragent`.

**Foreign HIP** (leapdragon, a17t) keeps **their** Author **and**
Committer. Cherry-pick `-x`, force Committer = source Author, keep their
trailers, add none of ours. Recipe: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
Same bar as extras PR #1.

This hippihx review commit is **new documentation**. It is not a
kernel migrate and must not be used as a template for body imports.

| extras / PR path | Introduced | Keep as |
|---|---|---|
| `fa_rdna2.cu` | `b1b3fa938` BlivionIaG `<kev29lt@gmail.com>` | BlivionIaG. Later FA `torch::zeros` is extras serve — leave it there |
| `q_gemm_rdna2.cu` | `fabf51493` BlivionIaG | BlivionIaG |
| `q_gemm_rdna2_awq_prefill.cu` | `feb7b457e` BlivionIaG | BlivionIaG |
| `moe_q_gemm_rdna2.cu` | `b1b3fa938` BlivionIaG | BlivionIaG |
| `gdn_decode_rdna2.cu` | `55527010c` BlivionIaG | BlivionIaG. NULL_BLOCK_ID sentinel is extras serve until dest-locked |
| `exl3_dot2_*.cu` | `40850e6c5` BlivionIaG | BlivionIaG (Author **and** Committer), including the port and later perf commits |
| `causal_conv1d_rdna2.cu` | `5eb84b4fa` BlivionIaG | BlivionIaG |
| `sparse_mla_rdna2.cu` / `indexer_paged_mqa_rdna2.cu` | BlivionIaG | BlivionIaG |
| dest `rdna_ar` (merged PR #1) | Unique HIP: **Aron Hsiao** `<leapdragon@gmail.com>` | Pick unique commits (`af25c5329`…`ee6e48ea1`), Author **and** Committer Aron Hsiao. Keep `Co-Authored-By: Claude Fable 5`. **Do not pick squash `a4060647`.** No Blivion/Cursor trailers |
| PR #2 `glm5_kda_*` / `glm5_dsa_*` | BlivionIaG | BlivionIaG; rename off `glm5_` in a **follow-up hippihx** commit, not by rewriting their HIP |
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | If dest ever locks that family, pick **their** commits, not a rewrite |
