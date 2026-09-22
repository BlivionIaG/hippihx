# extras → hippihx backport review

Lock: [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `f3dd65fa7063` (2026-09-20 20:29 UTC). Dest **default
branch is `rdna_extras`**, not `main` (`c00091e02670` upstream vLLM — no
dest HIP). **No** `torch.ops.hippihx.*`. Observe only: no `.cu` dump, no
tok/s, no extras maxdiff.

hippihx is the zoo. extras is serve wiring. Dest HIP is still
ATen-coupled. V1 C ABI (`include/hippihx/v1.h`, rev **3**,
`attention.*`) is the consume path; `hippihx_v1_run` is `NOT_READY`
until a body migrates and extras binds it.

## Observed SHAs

| SHA | What | Zoo |
|---|---|---|
| `a4060647cfbb` | extras PR **#1** squash (`rdna_ar` Uncached+push) | Later `comm.pcie`. Pick unique Aron Hsiao commits, **not** squash |
| `1046782fb8c4` | deleted `q_gemm_rdna2_awq_prefill.cu` | one W4 family; leftover `_awq_prefill_available` is extras dead code |
| `820465315bde` | Flash-Next QSA abort tracker | extras Triton; HIP `qsa_rdna2` / `VLLM_RDNA_QSA_HIP` still opt-in |
| `02adbfd4` | GDN decode fp16 SSM state | observe `attention/gdn_scan`; activations stay fp16 |
| `1ff73596d81a` | V1 FULL→PIECEWISE + persist keepalive | extras runner. Dest @ `f3dd65fa`: FULL redirect only when compiled + piecewise. Zoo does not own graph mode. |
| `849292ec` | vLLM custom AR cudagraph-correct | extras serve, not leapdragon `rdna_ar` |
| `4d25a0483912` | gfx1030 launcher `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` | extras; `envs.py` still False; `VLLM_RDNA_AR` still **0** |
| `f5cbbdfec494` / `7e70e2400542` | `bench_report.py` / newest-by-mtime | extras ops |
| `cb0d4418` / `9c9509b3` | Qwen4Exp HIP S6 default-on then revert | gates still **off** |
| `d0d577f1907c` | `_custom_ops` wrappers | extras Python; still opt-in |
| `8960a3bcbb17` | isolated HC HIP compute | extras product; capture still not dest-on |
| `d1b200b1` | FA GQA prefill O register-resident | observe `attention/fa_fdot2`; occupancy pin closed |
| `cd1231fd` | GDN prefill HIP opt-in (`VLLM_GDN_HIP_PREFILL=1`) | default Triton/FLA |
| `388a61b6f75f` | GDN one-shot `zero_()` wipe removed | zeros at allocation only; never wipe live state |
| `3bddd3c9a5d1` | Flash-Next production launcher (`serve_gfx1030_flashnext.sh`) | extras scripts. PIECEWISE + seq cap 6. leapdragon `VLLM_RDNA_AR` unset (still **0**). Zoo still no Finegrained. |
| `50120e13b468` | HC `_contig()` capture-safe per-shape cache | extras product Python. Gate still **off**. Remaining same-shape clobber. No V1 op. |
| `5c4ab9891910` | dest-reverted gfx1030 wvSplitK decode port | extras dense GEMM. Kernel asserts on gfx1030 under capture. Keeps `gemv_f16_rdna2` for decode `M<=8`. **No zoo tile.** Do not reintroduce `gemv_f16`. |
| `e45dd5cb2de8` | QSA `CircularBufferManager` empty-ring prefix hits | extras v1 core. Observe `attention/qsa_indexer`. Empty compression-group ring is valid. Do not copy tok/s. |
| `c59a23f625e0` | recovered extras probes / benches / HC Triton WIP | extras ops. HC Triton + graph-keepalive diagnostics **not** on the serve path. Do not dump. |
| `31003ff` / `ae5c7edc` / `37345ee9` / `aaa4775a` / `0dd38115` / `657bdba8` / `9c602943` / `609c9c0d` | Flash-Next launcher serve knobs | extras scripts. Vision-on, then `609c9c0d` FULL_AND_PIECEWISE (ROCm executes as PIECEWISE). Stay extras. Do not copy tok/s. |
| `741e5bc31ae5` | mamba spec-decode tables index by `req_idx` | extras worker (upstream vLLM #55506 port; Author Karl0007). Persistent per-request-slot tables. V1 `req_idx == batch_idx`. Stay extras. |
| `3b59ee16e553` | extras PR **#13** squash (T44b `rdna_ar` VRAM flags + wedge) | Later `comm.pcie`. Still opt-in (`VLLM_RDNA_AR=0`). Dest extras `MAX_KB` default **64**. Zoo lock **512**. Do **not** pick squash (Cursor rewrite). |
| `dbb1e7764aba` | extras PR **#14** merge (V620 Triton MoE JSON / ROCR amdsmi / PLE fp8) | Stay extras. HIP MoE ignores the JSON. Cursor rewrite — do not pick. |
| `4425834a26ac` | ROCm platform/worker startup (amdsmi import, hip fallback, `torch.cuda.init`) | Stay extras. Serve/platform only. No HIP body. |
| `3d6df9ed617a` | extras `exl3_dot2_*` `__HIP__RDNA__` guard for docker multi-arch | Stay extras. Zoo still one `--offload-arch` per fatbin. gfx1150 / gfx12xx not dest. Do not dump. |
| `ed94e3f3d299` | Qwen4Exp MTP proposer allowlist + skinny `w2_zp` | Stay extras. MTP still not dest. No HIP body. |
| `b33f9b66eb2b` | `get_device_name` torch fallback when amdsmi has no handles | Stay extras. Platform only. |
| `f3dd65fa7063` | extras PR **#15** merge (QSA live-context bound + folded **#12** PLE/MTP/graph-redirect) | Stay extras. Python/Triton/serve. No HIP body. Foreign George + Codex. Do not pick merge. Do not copy tok/s. |

Other dest-fixed ISA (keep in tile locks, not a dump): GDN prefill `o`
`i_t_local`; causal_conv FIR pre-shift then shift; ConfigH
(`K_STEP=64`) reverted. Live dest bugs: [Dest extras defects](#dest-extras-defects).

Open extras PRs **#2** (GLM Later), **#18**, **#19**. Draft **#9/#10**
sdot and **#17** resident MoE — skip. Closed **#3/#16** unmerged,
**#11** dest-reverted wvSplitK (**no zoo tile**), **#12** superseded
by dest **#15**. Merged **#13/#14/#15** dest *presence*, observe, do
not pick. `rdna_extras_wip_20260910` is not dest.

## Action

| Source | Action |
|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Later.** One W4 family (`use_v2_format`; ConfigA for `M>256`). Do not copy dest ZP / K-split / ConfigH / persist keepalive. Migrate when extras can *call* hippihx. |
| FA GQA-subgroup prefill + register-resident O | **Later** `attention/fa_fdot2`. Occupancy pin closed. |
| causal_conv out-stride + null-block | **Later** `sequence/causal_conv`. |
| GDN decode fp16 SSM state | **Later** `attention/gdn_scan`. Recurrence stays 16 fp32 VGPR. |
| leapdragon `rdna_ar` + PIX helpers + dest T44b (`3b59ee16`) | **Later** `comm.pcie`. Still opt-in. Dest extras `MAX_KB` **64**; zoo **512**. Do not pick squash. |
| PR **#2** GLM-5.3 KDA/DSA | **Later** `kda_scan` / `dsa_nope` / `qsa_indexer`. Do not name tiles `glm5_*`. |
| GDN arenas, `rdna2_graph_keepalive.cuh`, breakable cudagraphs, Hybrid W4 gfx10, Flash-Next HC/QSA/PLE/M-RoPE HIP, skinny GEMM / `gemv_f16_rdna2`, PLE schema, W4 MoE oracle, `new_zeros`/`zeros_like`, V1 FULL→PIECEWISE, vLLM custom AR, `bench_report.py`, seq cap 6, Flash-Next launcher knobs, HC `_contig()` cache, QSA Triton bounds, mamba spec-decode, T44b wedge, V620 MoE JSON / amdsmi / PLE, ROCm platform init, extras EXL3 docker arch-guard, Qwen4Exp MTP, recovered extras probes, `qwen4_exp/**` | **Stay extras.** Product gates default off (S6 revert). No new V1 op until dest locks a class. |
| PR **#3** a17t / explore **#9/#10** sdot / closed **#12** / closed **#16** / draft **#17** / PR **#18** / PR **#19** | **Skip.** Second W4 family, not dest, serve-only. **#16** unmerged. **#17** resident MoE. **#18** GPTQ `BLOCK_KN_SIZE` 256. **#19** MTP unquantized-weight detect. |
| PR **#5** / **#8** Flash-Next, extras **#11** wvSplitK, extras **#13** T44b, extras **#14**, extras **#15** | **Closed.** Dest-landed T44b / #14 / #15 are observe-only. Dest reverted wvSplitK (`5c4ab989`). Do not re-merge. Do not grow `gemv_f16`. |

Bodies stay in extras because every dest HIP file includes `torch/all.h`.
A dump is not a migrate. CONTRIBUTING: do not edit extras from this repo.

## Dest file → tile

Observed at `f3dd65fa7063`. Numbers are extras observations, not dest
locks. Fill tile READMEs.

| extras path | hippihx tile | Notes |
|---|---|---|
| `csrc/rocm/fa_rdna2.cu` | `attention/fa_fdot2` | ~33 KiB decode / ~48 KiB prefill smem. GQA-subgroup default. O accumulator register-resident @ `d1b200b1`. Persist O stay extras. |
| `csrc/rocm/gdn_decode_rdna2.cu` | `attention/gdn_scan` | 16 fp32 VGPR/thread. SSM `fp16` or `fp32` @ `02adbfd4`. `NULL_BLOCK_ID=0` is a vLLM sentinel — zoo is “invalid slot → zero out”. |
| `csrc/rocm/gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | `o` uses `i_t_local`. Prefill HIP opt-in @ `cd1231fd`. Dispatch still misses dtype. |
| `csrc/rocm/q_gemm_rdna2.cu` + `q_gemm_rdna2_prefill.cu` + `qdq_4_rdna2.cuh` | `gemm/w4a16_fdot2` | One family (`use_v2_format`). ConfigA for `M>256`. Dest deleted `q_gemm_rdna2_awq_prefill.cu` @ `1046782`. ZP still scale-baked. K-split still unaligned. |
| `csrc/rocm/moe_q_gemm_rdna2.cu` | `moe/routed` | Reuses W4 helpers. |
| `csrc/rocm/exl3_dot2_{dense,moe,dequant,hadamard}.*` | `gemm/exl3_3inst` | `LDS_PAD=8`. Dest @ `3d6df9ed` widened extras compile guard for docker multi-arch. Zoo still one arch per fatbin. Produce stays `-cb 3inst` outside. Do not dump. |
| `csrc/rocm/causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, wave32. FIR on pre-shift then shift. |
| `csrc/rocm/indexer_paged_mqa_rdna2.cu` | `attention/qsa_indexer` | DeepSeek-class indexer. |
| `csrc/rocm/qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | `VLLM_RDNA_QSA_HIP` default **off**. |
| `csrc/rocm/sparse_mla_rdna2.cu` | `attention/dsa_nope` | Sparse MLA class. |
| `csrc/rocm/rdna_allreduce.{cu,cuh}` | `comm/pcie` | Uncached+push Later. Dest @ `3b59ee16`: flags in uncached VRAM + T44b wedge. Occupancy pin closed. INT8/Q8 wire; no Finegrained. Do not dump. |
| `csrc/rocm/{mrope,hc,ple_short_conv,rdna_fused_glue}_rdna2.cu` | — | Product HIP. Stay extras. |
| `later/glm53-…` `glm5_*` | `kda_scan` / `dsa_nope` / `qsa_indexer` | Later. Drop `glm5_` prefix. |

Do not take a17t unique W4 (`awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`,
`gemv_w4_kpack_rdna2.cu`). Do not grow `gemv_f16`. Skinny GEMM stays
extras consume.

## When a migrate *is* pertinent

1. Dest GDN hybrid is capture-safe (or GDN HIP stays opt-in). Dest
   journals FPP13 16k c=8 green via serve arenas — still extras.
2. Tile README LDS / `__launch_bounds__` / wave / DOT unit are filled.
3. hippihx ships one HIP entry extras can bind. **Started:** V1
   `plan`/`run`. `run` is `NOT_READY`. extras @ `f3dd65fa7063` has no
   `torch.ops.hippihx.*`.
4. extras is rewired **in extras**. The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs. Tracker:
[`EXTRAS.md`](EXTRAS.md).

## Dest extras defects

Dest tip `f3dd65fa7063`. Live dest bugs / rolled-back paths — hippihx
must not reproduce them.

| Defect | Where | Status @ `f3dd65fa` | Zoo lock |
|---|---|---|---|
| `llvm.amdgcn.fdot2.bf16.bf16` ISel abort | Hybrid W4 gfx10 + bf16. Dest HIP has **no** `fdot2.bf16`. | **Serve-mitigated** (`59237b3`). | Never `fdot2.bf16`. DOT + GDN HIP = **fp16 act**. |
| Scale-baked W4 zero-point | `qdq_4_rdna2.cuh` `prep_zero_scale_fp16`: `0xE400 \| zero` then `scale * (-1024 - zero)` in `half`. | **Still live** | Integer `q - zero`, then `* scale`. |
| Unaligned prefill K-split | `compute_split_k`: K=640 → 16×40. Kernel `K_STEP=32`. Cap at 8 and `k_per_split % K_STEP == 0` **not landed**. | **Still live** | Equal `k_per_split`, multiple of 32. Refuse 40-wide. |
| Prefill ConfigH `K_STEP=64` | Garbage for `M>256`. | **Dest-reverted** (`7ac98a26`) | Keep ConfigA (`K_STEP=32`). |
| GDN HIP selected on BF16 | `_gdn_prefill_dispatch_available()` checks GPU + symbols, **not dtype**. | **Still live** | `plan` / V1 refuse bf16 on `attention.gdn_scan`. Prefill HIP opt-in. |
| GDN decode HIP skipped on fp16 SSM state | Dispatch required `ssm_state.dtype == float32`. | **Dest-fixed** (`02adbfd4`) | Act **fp16**. SSM **fp16 or fp32**. Refuse bf16 act. |
| GDN prefill `o` varlen chunk index | Used global `i_t`. | **Dest-fixed** (`i_t_local`) | Per-sequence local chunk. |
| GDN piecewise capture state poison | Pool-block conv/ssm pages. | **Dest-fixed (serve)** — arenas | Scratch from `plan`. Arenas stay extras. |
| causal_conv update vs fwd FIR | Shifted state before FIR. | **Dest-fixed** (`cafe95ef8`) | FIR on pre-shift, then shift. |
| causal_conv fwd out-stride / null-block | Wrong out stride. | **Dest-fixed** (`82b6f183da2a`) | `stride_o_token`; invalid slot → skip. |
| QSA / Flash-Next attention abort | Triton `forward_qsa` GPU trap (`820465`). `VLLM_RDNA_QSA_WARPS=2` did **not** fix it. | **Serve-mitigated** — eager (`8cf0dedb`); V1 FULL→PIECEWISE (`1ff73596`). HIP still **off**. | Indexer occupancy: **4 warps** gfx1030 BF16 6h×256. |
| Custom AR garbage under breakable graphs | `custom_all_reduce.py`: `registered=True` + `empty_like` on a real forward. | **Dest-fixed (serve)** (`849292ec`). Launcher default-on (`4d25a048`). | Never `empty_like` on a real forward. Zoo Uncached+push stays opt-in. |
| Separate AWQ prefill `.cu` | `q_gemm_rdna2_awq_prefill.cu` | **Dest-deleted** (`1046782`). `_awq_prefill_available` is extras dead code. | One W4 family. |
| Qwen4Exp HIP S6 default-on | `VLLM_RDNA_{HC_PREFILL,QSA,PLE_CONV}_HIP` | **Dest-reverted** (`cb0d4418` / `9c9509b3`) | Stay extras. Gates **off**. |
| Qwen4Exp HIP `_custom_ops` missing | No Python wrapper | **Dest-fixed (serve)** (`d0d577f1`) | Stay extras. |
| Qwen4Exp HC HIP compute / `GROUP_DIM` | `BLOCK<=512` reject, OOB, `new_empty` | **Dest-fixed isolated** (`8960a3bc`). Gate still **off** (PIECEWISE MoE faults). | Stay extras. Serve zeros. |
| HC `_contig()` graph-stale temps | `hc_rdna2.py` per-call `.contiguous()` | **Dest-fixed isolated** (`50120e13`) per-shape cache. Gate still **off**. Same-shape clobber still live. | Stay extras. Serve zeros. Per-call-site scratch from `plan`, not a shared shape cache. |
| GDN one-shot `zero_()` state wipe | First `gdn_decode_rdna2` after prefill | **Dest-fixed (serve)** (`388a61b6`) | Zeros **once at allocation**. |
| GDN batched-decode n≥8 | Corrupts recurrent state | **Still live.** Launchers cap `max-num-seqs` 6 (`b78006a4`, `3bddd3c9`). | Stay extras. No serve seq-cap in tiles. |
| Flash-Next FULL_AND_PIECEWISE one-request corruption | `compilation-config` FULL_AND_PIECEWISE at c=8 even with seq cap 6 | Dest serve @ `609c9c0d`: production launcher now FULL_AND_PIECEWISE (ROCm executes as PIECEWISE). Dest retraces the c=8 report to probe artifacts, not graphs. | Stay extras. Zoo does not own graph mode. |
| gfx1030 wvSplitK decode (`n<=5`) | `wvSplitK_hf_big_` device `assert(false)` under cudagraph capture | **Dest-reverted** (`5c4ab989`; was `c350fa218`). Decode stays `gemv_f16_rdna2` `M<=8`. | **No zoo tile.** Do not grow `gemv_f16`. |
| QSA prefix cache always-zero | `CircularBufferManager.find_longest_cache_hit` returned 0 at empty group ring | **Dest-fixed (serve)** (`e45dd5cb`) | Stay extras. Empty ring at group boundary is valid. |
| Mamba spec-decode batch-row tables | captured `data_ptrs` indexed by batch row under deferred postprocess | **Dest-fixed (serve)** (`741e5bc3`). V1 unchanged. | Stay extras. Persistent per-request-slot tables; index by `req_idx`. |

Serve notes (not zoo): GDN prefill HIP opt-in (`cd1231fd`); RDNA_ATTN
spec/MTP verify default-off; `eager_break_during_capture` stays extras;
FA persist / `rdna2_graph_keepalive.cuh` (`c091c420b`) is capture
plumbing — skip as a body pick. Dest Flash-Next launcher
(`3bddd3c9`) also sets `HSA_FORCE_FINE_GRAIN_PCIE=1` and
`VLLM_RDNA_FUSED_HC=0` — extras serve; zoo `comm.pcie` still no
Finegrained.

## Attribution (do not re-author)

**Dest extras HIP** is **BlivionIaG** `<kev29lt@gmail.com>`. Author
**and** Committer on every picked dest commit.

**Foreign HIP** (leapdragon, a17t, Aron Hsiao Flash-Next / recipe /
Hybrid W4) keeps **their** Author **and** Committer. Cherry-pick `-x`,
force Committer = source Author. Recipe: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
Same bar as extras PR #1.

This review is documentation, not a kernel migrate.

| extras / PR path | Introduced | Keep as |
|---|---|---|
| `fa_rdna2.cu` | `b1b3fa938` BlivionIaG `<kev29lt@gmail.com>` | BlivionIaG. Later FA `torch::zeros` is extras serve |
| `q_gemm_rdna2.cu` | `fabf51493` BlivionIaG | BlivionIaG |
| `q_gemm_rdna2_awq_prefill.cu` | `feb7b457e` BlivionIaG | **Deleted** dest @ `1046782`. Do not reintroduce. |
| `moe_q_gemm_rdna2.cu` | `b1b3fa938` BlivionIaG | BlivionIaG |
| `gdn_decode_rdna2.cu` | `55527010c` BlivionIaG | BlivionIaG. Dest @ `02adbfd4` fp16 SSM state |
| `exl3_dot2_*.cu` | `40850e6c5` BlivionIaG | BlivionIaG (Author **and** Committer) |
| `causal_conv1d_rdna2.cu` | `5eb84b4fa` BlivionIaG | BlivionIaG |
| `mrope_rdna2.cu` / `qsa_rdna2.cu` / `hc_rdna2.cu` / `ple_short_conv_rdna2.cu` | dest scaffolding | Blivion follow-ups. Opt-in; not dest-on. `_contig()` cache @ `50120e13` stays extras |
| `sparse_mla_rdna2.cu` / `indexer_paged_mqa_rdna2.cu` | BlivionIaG | BlivionIaG |
| dest `rdna_ar` (merged PR #1) | Unique HIP: **Aron Hsiao** `<leapdragon@gmail.com>` | Pick unique commits (`af25c5329`…`ee6e48ea1`), Author **and** Committer Aron Hsiao. Keep `Co-Authored-By: Claude Fable 5`. **Do not pick squash `a4060647`.** |
| dest Flash-Next / Hybrid W4 / recipe ports | Unique: **Aron Hsiao** `5765f57b4c41` / `c05af408775f` / `22bb2e8d06f3` | Their unique commits if dest-locked. Dest Blivion follow-ups stay Blivion |
| PR #2 `glm5_kda_*` / `glm5_dsa_*` | BlivionIaG | BlivionIaG; rename off `glm5_` in a follow-up hippihx commit |
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | Closed unmerged. If dest ever locks that family, pick **their** commits |
| Explore PRs **#9/#10** sdot | **Not taken** | Not dest |
| extras dest-presence serve (**#13** T44b / **#14** / **#15** / mamba `741e5bc3` / closed **#12**) | observe | Python/Triton/serve. No HIP migrate. Do not pick Cursor/George/Codex/Karl rewrites. Unique Aron Hsiao `rdna_ar` still pickable. |
| extras not dest (**#16** closed / **#17** draft / **#18** / **#19**) | **Not taken** | Resident MoE, GPTQ `BLOCK_KN_SIZE` 256, MTP detect. |
