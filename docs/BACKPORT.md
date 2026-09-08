# extras → hippihx backport review

Reviewed `opengfx1030/vllm-rdna` `rdna_extras` @ `d71721c79547`
(2026-09-08) plus open PRs #1–#3. **Do not copy kernel bodies into this
tree yet.**

hippihx is the zoo. extras is serve wiring. Copying ATen-coupled
`csrc/rocm/*.cu` here would freeze a second ISA copy while dest is still
hunting GDN capture bugs, with no consume path (`torch.ops.hippihx.*`
does not exist). Dual copies are how serve bugs accrete.

## Verdict

| Source | Pertinent to hippihx? | Action |
|---|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Yes, later** — zoo class | Wait. Record observed locks. Migrate only when extras can *call* hippihx. |
| Dest tip 2026-09-06…08 (cudagraph zeros, eager_break, spec/MTP gates, GDN probes, profiling) | **No** | Stay in extras. That is page-commit / capture / dispatcher work. |
| PR #1 leapdragon `rdna_ar` | **Later** — `comm.pcie` | Review-only, AR default-off, occupancy pin closed. |
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

Observed at `d71721c79547`. Numbers are extras *observations*, not dest
locks. Fill tile READMEs; do not invent tok/s.

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
| leapdragon `rdna_allreduce.cuh` (PR #1) | `comm/pcie` | Uncached+push Later. INT8/Q8 wire. Default off. |
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
   extras observations (this review started that).
3. hippihx ships one HIP entry that extras can bind as one V1 op.
4. extras is rewired to call it (that edit happens **in extras**, not from
   this tree). The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs. Kernel / mode / env / AR
tracker (all **unvalidated**): [`README.md`](../README.md#unvalidated-extras-inventory).

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
| PR #1 `rdna_ar` | **Aron Hsiao** `<leapdragon@gmail.com>` | Aron Hsiao Author **and** Committer. Keep `Co-Authored-By: Claude Fable 5`. No Blivion/Cursor trailers |
| PR #2 `glm5_kda_*` / `glm5_dsa_*` | BlivionIaG | BlivionIaG; rename off `glm5_` in a **follow-up hippihx** commit, not by rewriting their HIP |
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | If dest ever locks that family, pick **their** commits, not a rewrite |
