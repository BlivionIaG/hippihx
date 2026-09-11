# extras → hippihx backport review

Last checked `opengfx1030/vllm-rdna` `rdna_extras` @ `1046782fb8c4`
(2026-09-10 23:59 UTC). Dest **default branch is `rdna_extras`**, not
`main`. `main` is still unrelated upstream vLLM (`c00091e02670`,
2026-09-02) — **no dest HIP landed there**. Open extras PRs **#2** /
**#3** did not move. Draft extras PR **#5** (Flash-Next V620) is a
mixed other-fork; **skip**. **Do not copy kernel bodies into this tree
yet.**

Dest extras is **nine commits** past `a4060647cfbb`. 2026-09-10 journals
GDN hybrid **16k c=8 FPP13 green** via serve arenas. Findings
(`docs/profiling/2026-09-10-awq-vs-gptq-prefill-{review,microbench}.md`):
the separate AWQ prefill tile was slower and unused; dest **deleted**
`q_gemm_rdna2_awq_prefill.cu` and routes AWQ through
`gptq_gemm_rdna2_prefill` (`use_v2_format` / `zero_offset`, ConfigA for
`M>256`). That **matches** the zoo lock (one W4 family, pack/zeros).
Do not reintroduce a second W4 GEMM (a17t or dest's old high-M tile).

W4 scale-baked ZP, unaligned prefill K-splits, `fdot2.bf16`, and GDN
HIP selected on BF16 are **still live dest defects**. hippihx V1 still
refuses those. See [Dest extras defects](#dest-extras-defects-do-not-copy).

**Consume ABI started.** `include/hippihx/v1.h` + `tiles/v1_abi.cpp` ship
the torch-free C entry family (`hippihx_v1_plan` / `hippihx_v1_run`) that
extras will wrap as one `torch.ops.hippihx.*` per op. `hippihx_v1_run`
returns `HIPPIHX_V1_ERR_NOT_READY` until a body migrates. This is migrate
criterion **#3** (one HIP entry extras can bind) — not a body dump.

Delta since `a4060647cfbb`: dest extras is **nine commits ahead**
(seven through `6c5ff94ef` GDN arenas/journal, then `aaaae85be` ConfigA
for large-M AWQ prefill + findings, `1046782fb` delete AWQ prefill
`.cu`). GitHub squash-merged [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
(`rdna_ar` Uncached+push) at `a4060647`. PRs **#2** (`later/glm53-flash-awq-glm5next`
@ `4b31f2eede51`) and **#3** (a17t WIP) did not move. PR **#4** closed
review-only after landing on dest. PR **#5** draft is not dest.
`main` is unrelated upstream vLLM (`c00091e02670`). Issues are disabled.
WIP branch `rdna_extras_wip_20260910` is a pre-`cafe95ef8` snapshot
(TRUE FULL experiments) — **not dest**.

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

hippihx is the zoo. extras is serve wiring. Dest now journals GDN
capture as fixed for one FPP13 cell; extras still has no
`torch.ops.hippihx.*` consume path, and dest HIP is still ATen-coupled
(plus persist keepalive). Copying `.cu` here would freeze a second ISA
copy. Dual copies are how serve bugs accrete.

## Verdict

| Source | Pertinent to hippihx? | Action |
|---|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Yes, later** — zoo class | Wait. Record observed locks. Dest now **one** W4 prefill (`gptq_gemm_rdna2_prefill` + `use_v2_format`; ConfigA for `M>256`). **Do not copy dest W4 ZP / K-split / bf16 DOT / persist keepalive.** Migrate only when extras can *call* hippihx. |
| Dest tip 2026-09-10 GDN state arenas + persist keepalive | **No** | Stay in extras. Arenas / `rdna2_graph_keepalive.cuh` / immortal `hipMalloc` are capture page-commit, not tiles. |
| Dest tip 2026-09-06…08 (cudagraph zeros, eager_break, spec/MTP gates, GDN probes) | **No** | Stay in extras. |
| Dest tip leapdragon `rdna_ar` (merged PR #1 @ `a4060647`) | **Later** — `comm.pcie` | Communicator gate still opt-in (`VLLM_RDNA_AR=1`). Occupancy pin closed. Do not dump the ATen wrapper. A recipe may export `=1`; that is not dest-on. |
| PR #2 GLM-5.3 KDA/DSA (`later/glm53-…`) | **Later** — `kda_scan` / `dsa_nope` / `qsa_indexer` | Product-named `glm5_*` files. Do not name tiles after GLM. |
| PR #3 a17t `[WIP] Similar work, different fork` | **No** | WIP, untested, mixed Triton+HIP+qwen4_exp serve. Duplicate W4 family. |
| PR #5 `[Draft] Enable FP16 Qwen3.8 Flash-Next` | **No** | Mixed other-fork vs dest extras. Tok/s / PLE / MTP serve. Do not merge into hippihx. |

## Why bodies stay in extras for now

1. Every dest HIP file includes `torch/all.h` + ATen CUDAGuard. hippihx
   tiles are torch-free `plan` / `bind` / `run` stubs. A dump is not a
   migrate.
2. Scaffold non-goal: do not land production FA / EXL3 / AWQ bodies
   until LDS / `__launch_bounds__` are locked *here*.
3. UNC-26 (EXL3) is In Progress. FA occupancy pin stays closed.
4. Dest journals GDN hybrid capture **fixed** for FPP13 16k c=8
   (`docs/profiling/2026-09-10-truefull-journal.md`) via **serve**
   conv/ssm arenas (not a zoo file). Prefill `o` varlen chunk-boundary
   is ISA-fixed (`i_t_local`). Prefill HIP is **opt-out**
   (`VLLM_GDN_HIP_PREFILL == "0"`), not a code default-off. Dispatch still
   does not check dtype. Persist keepalive / immortal `hipMalloc` stay
   extras. Do not copy those into tiles.
5. CONTRIBUTING: do not edit `opengfx1030/vllm-rdna` from this repo.
   Without a hippihx consume bind, a copy here cannot replace extras.

## Dest file → tile (when migrate is allowed)

Observed at dest tip `1046782fb8c4` (ISA files + PR #1 AR @
`a4060647cfbb`). Numbers are extras *observations*, not dest locks.
Fill tile READMEs; do not invent tok/s.

| extras path | hippihx tile | Notes |
|---|---|---|
| `csrc/rocm/fa_rdna2.cu` | `attn/fa_fdot2` | ~33 KiB decode / ~48 KiB prefill smem. Occupancy pin closed. @ `6c5ff94`: GQA-256 idle-wave shuffle + NaN guard; fp16 flash KV writer (`reshape_and_cache_flash_rdna2`, `__launch_bounds__(128, 4)`). Persist workspaces stay extras. |
| `csrc/rocm/gdn_decode_rdna2.cu` | `attn/gdn_scan` | Register-resident 16 VGPR/thread, no LDS. `NULL_BLOCK_ID=0` is a **vLLM sentinel** — zoo contract is “invalid slot → zero out, do not touch state”, not that constant. |
| `csrc/rocm/gdn_prefill_*_rdna2.cu` | `attn/gdn_scan` | `o` kernel LDS ≈ 45312 B. @ `6c5ff94`: varlen uses **local** chunk `i_t_local`. Prefill HIP opt-out (`VLLM_GDN_HIP_PREFILL==0`). |
| `csrc/rocm/q_gemm_rdna2.cu` + `q_gemm_rdna2_prefill.cu` + `qdq_4_rdna2.cuh` | `gemm/w4a16_fdot2` | GPTQ and AWQ are pack/zeros (`use_v2_format` → `zero_offset` 1/0), **one** GEMM family. `LDS_PAD=8` on decode. Prefill `select_config`: ConfigA for `M>256` & `N>=4096`, else ConfigC. Dest **deleted** `q_gemm_rdna2_awq_prefill.cu` @ `1046782`. Dest ZP still scale-baked. |
| `csrc/rocm/moe_q_gemm_rdna2.cu` | `moe/routed` | Reuses W4 helpers. |
| `csrc/rocm/exl3_dot2_{dense,moe,dequant,hadamard}.*` | `gemm/exl3_3inst` | `LDS_PAD=8` on A-staging. No `__launch_bounds__` on DOT kernels (VGPR is the occupancy lever). Produce stays `-cb 3inst` **outside**. |
| `csrc/rocm/causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, wave32, `state_len≈3–4`, dim multiple of 32, register-only (no LDS). @ `cafe95ef8`: FIR on **pre-shift** state, then shift (matches fwd). |
| `csrc/rocm/indexer_paged_mqa_rdna2.cu` | `attn/qsa_indexer` | Dest indexer; confirm class vs DSA later. |
| `csrc/rocm/sparse_mla_rdna2.cu` | `attn/dsa_nope` | Sparse MLA class, not a product fuse. |
| `csrc/rocm/rdna_allreduce.{cu,cuh}` (merged PR #1) | `comm/pcie` | Uncached+push Later. Host-coherent flags. Boot self-test. Communicator opt-in. Occupancy pin closed. INT8/Q8 wire preferred; no Finegrained. |
| `later/glm53-…` `glm5_kda_*.cu` | `attn/kda_scan` | Later. Do not keep the `glm5_` prefix. Do not retarget GDN 16/48 onto KDA 64×128. |
| `later/glm53-…` `glm5_dsa_*.cu` | `attn/dsa_nope` + `qsa_indexer` | Later. Same rename rule. |

## Stay in extras (not tiles)

| extras change | Why |
|---|---|
| `torch::zeros` / persist keepalive / immortal `hipMalloc` for FA `O` / GEMM `C` / GDN scratch | Caller page-commit. Zoo: scratch from `plan`, serve zeros. Do **not** copy `rdna2_graph_keepalive.cuh`. |
| GDN conv/ssm state arenas (PR #4 + `74f47b6af`) | Capture dispatcher / layer buffers. |
| `eager_break_during_capture` (GDN, `do_kv_cache_update`) | Capture dispatcher. |
| RDNA_ATTN spec / MTP verify gates | Serve policy. |
| `VLLM_LOG_GDN_PTRS`, `VLLM_GDN_DBG`, step timing | Probes. No D2H under capture in the zoo. |
| HIP AOT RMSNorm `forward_rocm` | No tile class. Leave fused norms in extras until a class exists. |
| Platform gates, `_rocm_C` vs `load_inline` | Registration. |
| dest `rdna_all_reduce.py` boot self-test + `CudaCommunicator` hook | Serve. Zoo is the Uncached+push tile only. |
| `sync_remote_to_local.sh`, recipes, profiling docs | Serve / ops. |
| gfx1100 WMMA W4 (`q_gemm_rdna3_wmma.cu`) | WMMA is a gfx110x-only Later overlay, never on shared DOT. |
| MXFP4 / W8A16 FP8 / skinny GEMMs | Extra consume families. Add a tile only when dest locks one as dest. |
| extras PR **#5** Flash-Next draft / `rdna_extras_wip_20260910` | Mixed other-fork / pre-reset snapshot. Not dest. |
| dest profiling findings (`2026-09-10-awq-vs-gptq-prefill-*.md`, decode-kernel-profile) | Serve / ops. Do not copy tok/s or µs tables into tile locks. |

## Do not take from a17t PR #3

Unique vs dest: `awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`,
`gemv_w4_kpack_rdna2.cu`, `qdq_awq_rdna2.cuh`. Dest already has AWQ as a
**zeros mode** on `q_gemm_rdna2*` (`use_v2_format`). Dest **deleted**
`q_gemm_rdna2_awq_prefill.cu` @ `1046782` after findings showed it unused
and slower. hippihx must not grow a second W4 GEMM. `qwen4_exp`, Triton
fallbacks, and serve recipes are extras (or another engine), not zoo.

Dest already dropped leapdragon `gemv_f16_rdna2` / `moe_skinny_int4_decode`
at PR #1 conflict resolution. Do not reintroduce skinny GEMV as a tile
without an extras dest lock.

## When a migrate *is* pertinent

All of:

1. Dest GDN hybrid is capture-safe (or GDN HIP stays opt-in and documented).
   **Dest journals FPP13 16k c=8 green** @ `6c5ff94` via serve arenas.
   Still extras; still no hippihx consume. Not a body dump.
2. Tile README LDS / `__launch_bounds__` / wave / DOT unit are filled from
   extras observations (this review started that; EXL3 grain-v2 + FA
   `__launch_bounds__` 128/256 recorded @ `a4060647`; FA flash KV writer
   + GQA-256 hygiene @ `6c5ff94`).
3. hippihx ships one HIP entry that extras can bind as one V1 op.
   **Started:** `include/hippihx/v1.h` (`hippihx_v1_plan` / `hippihx_v1_run`).
   `run` is still `NOT_READY` (no body). Extras has not rewired yet.
4. extras is rewired to call it (that edit happens **in extras**, not from
   this tree). The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs, grow the V1 ABI. Kernel /
mode / env / AR tracker (all **unvalidated**):
[`README.md`](../README.md#unvalidated-extras-inventory).

## Dest extras defects (do not copy)

Dest tip is `1046782fb8c4` (nine commits past `a4060647`). These are
**live dest bugs / rolled-back paths**, not tok/s. hippihx contracts
must not reproduce them. Dest **fixed / dropped**: GDN state arenas,
prefill `o` `i_t_local`, conv FIR order, **separate AWQ prefill `.cu`**.
Do not copy persist keepalive.

| Defect | Where on dest extras | Status @ `1046782` | Proper zoo lock | hippihx |
|---|---|---|---|---|
| `llvm.amdgcn.fdot2.bf16.bf16` ISel abort | Triton/fusion on BF16 vision interp, `triton_mrope`, BF16 causal conv. Dest HIP conv is fp16-oriented. | **Still live** (HIP conv stays scalar FMA) | Never `fdot2.bf16`. DOT + GDN HIP are **fp16 activations**. BF16 leftover / conv = scalar FMA, fp32 mul. | `dot.hpp`, V1 `HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`, `sequence/causal_conv`, `moe/leftover_bf16` |
| Scale-baked W4 zero-point | `qdq_4_rdna2.cuh` `prep_zero_scale_fp16`: `0xE400 \| zero` then `scale * (-1024 - zero)` in `half`. All-zero weights were not exact zero. | **Still live** | Integer `q - zero`, then `* scale`. GPTQ `uint4b8` (+1) vs AWQ literal is pack/zeros, not a second GEMM. | `gemm/w4a16_fdot2` |
| Unaligned prefill K-split | `q_gemm_rdna2_prefill.cu` `compute_split_k`: K=640 can pick 16 splits of 40. Kernel `K_STEP=32`. Generic `gemm_dynamic_kernel` still launches 40-wide. Findings proposed cap `split_k` at 8 and require `k_per_split % K_STEP == 0` — **not landed**. | **Still live** | Equal `k_per_split`, multiple of 32, inside LDS budget. Refuse the 40-wide split. | `gemm/w4a16_fdot2` |
| GDN HIP selected on BF16 | `_gdn_prefill_dispatch_available()` checks GPU + symbols, **not dtype**. Kernel still `mixed_qkv` fp16. | **Still live** | `plan` / V1 refuse bf16 on `attn.gdn_scan`. Prefill HIP is opt-out (`== "0"`). | `attn/gdn_scan`, V1 dtype |
| GDN prefill `o` varlen chunk index | `gdn_prefill_o_rdna2.cu` used global `i_t` so later sequences were OOB. | **Dest-fixed** (`i_t_local`) | Token offsets use per-sequence local chunk. | `attn/gdn_scan` |
| GDN piecewise capture state poison | Pool-block conv/ssm pages; per-step `block_table` views. | **Dest-fixed (serve)** — permanent BS-sized arenas | Scratch/state from `plan`; no realloc under capture. Arenas stay extras. | extras, not a tile |
| causal_conv update vs fwd FIR | Decode kernel shifted state **before** the FIR (mismatched fwd). | **Dest-fixed** (`cafe95ef8`) | FIR on pre-shift state, then shift. Scalar FMA. | `sequence/causal_conv` |
| Global EXL3 FP16 clip | Serve `Qwen2MoeMLP` clipped every model to FP16. Rolled back to EXL3-only on the integration fork. | Serve | Not a tile. Stay in extras. | — |
| QSA 2-warp BF16 prefill spill | gfx1030 6h×256 BF16 prefill: 2 warps exhaust VGPRs. Dest indexer decode is still 64-thread H=64 D=128. | **Still live** (Flash-Next observation) | **4 warps** (128 threads) for that shape. | `attn/qsa_indexer` |
| Separate AWQ prefill `.cu` | `q_gemm_rdna2_awq_prefill.cu` (exllama-clone, `BLOCK_M=16`) | **Dest-deleted** (`1046782`) after findings: unused + slower. AWQ uses `gptq_gemm_rdna2_prefill` + `use_v2_format`. | One W4 family. Pack/zeros only. Do not reintroduce. | `gemm/w4a16_fdot2` |

Serve rollbacks / capture notes (keep as serve, not zoo):

- GDN HIP prefill is **opt-out** (`VLLM_GDN_HIP_PREFILL == "0"`), not a
  code default-off. Recipes may still export `0`.
- RDNA_ATTN spec/MTP verify **default-off** (opt-in env).
- `eager_break_during_capture` on `do_kv_cache_update` stays extras.
- FA persist / immortal workspaces and `rdna2_graph_keepalive.cuh`
  (`c091c420b`) are mixed capture plumbing + debug `fprintf`. Skip that
  commit as a body pick; extract ISA only.
- GDN piecewise-capture **root cause** on dest is now arenas, not
  “eager is the only control”. Do not copy probe-driven ISA.

Skinny GEMM (`skinny_gemms.cu`) exists on dest extras (now wrapped in
persist keepalive). **No hippihx tile** until dest extras locks that
family as dest. Do not reintroduce leapdragon `gemv_f16` dropped at
extras PR #1.

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
| `q_gemm_rdna2_awq_prefill.cu` | `feb7b457e` BlivionIaG | **Deleted** dest @ `1046782`. Do not reintroduce. |
| `moe_q_gemm_rdna2.cu` | `b1b3fa938` BlivionIaG | BlivionIaG |
| `gdn_decode_rdna2.cu` | `55527010c` BlivionIaG | BlivionIaG. NULL_BLOCK_ID sentinel is extras serve until dest-locked |
| `exl3_dot2_*.cu` | `40850e6c5` BlivionIaG | BlivionIaG (Author **and** Committer), including the port and later perf commits |
| `causal_conv1d_rdna2.cu` | `5eb84b4fa` BlivionIaG | BlivionIaG |
| `sparse_mla_rdna2.cu` / `indexer_paged_mqa_rdna2.cu` | BlivionIaG | BlivionIaG |
| dest `rdna_ar` (merged PR #1) | Unique HIP: **Aron Hsiao** `<leapdragon@gmail.com>` | Pick unique commits (`af25c5329`…`ee6e48ea1`), Author **and** Committer Aron Hsiao. Keep `Co-Authored-By: Claude Fable 5`. **Do not pick squash `a4060647`.** No Blivion/Cursor trailers |
| PR #2 `glm5_kda_*` / `glm5_dsa_*` | BlivionIaG | BlivionIaG; rename off `glm5_` in a **follow-up hippihx** commit, not by rewriting their HIP |
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | If dest ever locks that family, pick **their** commits, not a rewrite |
