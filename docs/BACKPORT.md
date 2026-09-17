# extras → hippihx backport review

Lock: [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `50120e13b468` (2026-09-17 06:11 UTC). Dest **default
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
| `1ff73596d81a` | V1 FULL→PIECEWISE + persist keepalive | extras runner. PR **#12** would narrow it — **not dest** |
| `849292ec` | vLLM custom AR cudagraph-correct | extras serve, not leapdragon `rdna_ar` |
| `4d25a0483912` | gfx1030 launcher `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` | extras; `envs.py` still False; `VLLM_RDNA_AR` still **0** |
| `f5cbbdfec494` / `7e70e2400542` | `bench_report.py` / newest-by-mtime | extras ops |
| `cb0d4418` / `9c9509b3` | Qwen4Exp HIP S6 default-on then revert | gates still **off** |
| `d0d577f1907c` | `_custom_ops` wrappers | extras Python; still opt-in |
| `8960a3bcbb17` | isolated HC HIP compute | extras product; capture still not dest-on |
| `d1b200b1` | FA GQA prefill O register-resident | observe `attention/fa_fdot2`; occupancy pin closed |
| `cd1231fd` | GDN prefill HIP opt-in (`VLLM_GDN_HIP_PREFILL=1`) | default Triton/FLA |
| `388a61b6f75f` | GDN one-shot `zero_()` wipe removed | zeros at allocation only; never wipe live state |
| `3bddd3c9a5d1` | Flash-Next production launcher (`serve_gfx1030_flashnext.sh`) | extras scripts. PIECEWISE + seq cap 6. FULL_AND_PIECEWISE still corrupts one request at c=8 even with the cap. leapdragon `VLLM_RDNA_AR` unset (still **0**). Zoo still no Finegrained. |
| `50120e13b468` | HC `_contig()` capture-safe per-shape cache | extras product Python. Gate still **off**. Remaining same-shape clobber. No V1 op. |

Other dest-fixed ISA (keep in tile locks, not a dump): GDN prefill `o`
`i_t_local`; causal_conv FIR pre-shift then shift; ConfigH
(`K_STEP=64`) reverted. Live dest bugs: [Dest extras defects](#dest-extras-defects).

Open extras PRs **#2** (GLM Later) and **#3** (a17t WIP). Draft **#9/#10**
sdot — skip. Closed **#11** wvSplitK dest-picked as `c350fa218` — extras
dense GEMM, **no zoo tile**. Draft **#12** Intel CPU PLE / V620 MTP
startup — serve-only, no kernels. `rdna_extras_wip_20260910` is not dest.

## Action

| Source | Action |
|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Later.** One W4 family (`use_v2_format`; ConfigA for `M>256`). Do not copy dest ZP / K-split / ConfigH / persist keepalive. Migrate when extras can *call* hippihx. |
| FA GQA-subgroup prefill + register-resident O | **Later** `attention/fa_fdot2`. Occupancy pin closed. |
| causal_conv out-stride + null-block | **Later** `sequence/causal_conv`. |
| GDN decode fp16 SSM state | **Later** `attention/gdn_scan`. Recurrence stays 16 fp32 VGPR. |
| leapdragon `rdna_ar` + PIX helpers | **Later** `comm.pcie`. Still opt-in (`VLLM_RDNA_AR=1`). Hop class is `hippihx.comm.fabric`; `lspci`/ACS stay extras. |
| PR **#2** GLM-5.3 KDA/DSA | **Later** `kda_scan` / `dsa_nope` / `qsa_indexer`. Do not name tiles `glm5_*`. |
| GDN arenas, `rdna2_graph_keepalive.cuh`, breakable cudagraphs, Hybrid W4 gfx10, Flash-Next HC/QSA/PLE/M-RoPE HIP, wvSplitK, PLE schema, W4 MoE oracle, `new_zeros`/`zeros_like`, V1 FULL→PIECEWISE, vLLM custom AR, `bench_report.py`, seq cap 6, Flash-Next production launcher, HC `_contig()` cache, `qwen4_exp/**` | **Stay extras.** Product gates default off (S6 revert). No new V1 op until dest locks a class. |
| PR **#3** a17t / explore **#9/#10** sdot / PR **#12** | **Skip.** Second W4 family, not dest, serve-only. |
| PR **#5** / **#8** Flash-Next, **#11** wvSplitK | **Closed.** Dest already has them. Do not re-merge. |

Bodies stay in extras because every dest HIP file includes `torch/all.h`.
A dump is not a migrate. CONTRIBUTING: do not edit extras from this repo.

## Dest file → tile

Observed at `50120e13b468`. Numbers are extras observations, not dest
locks. Fill tile READMEs.

| extras path | hippihx tile | Notes |
|---|---|---|
| `csrc/rocm/fa_rdna2.cu` | `attention/fa_fdot2` | ~33 KiB decode / ~48 KiB prefill smem. GQA-subgroup default. O accumulator register-resident @ `d1b200b1`. Persist O stay extras. |
| `csrc/rocm/gdn_decode_rdna2.cu` | `attention/gdn_scan` | 16 fp32 VGPR/thread. SSM `fp16` or `fp32` @ `02adbfd4`. `NULL_BLOCK_ID=0` is a vLLM sentinel — zoo is “invalid slot → zero out”. |
| `csrc/rocm/gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | `o` uses `i_t_local`. Prefill HIP opt-in @ `cd1231fd`. Dispatch still misses dtype. |
| `csrc/rocm/q_gemm_rdna2.cu` + `q_gemm_rdna2_prefill.cu` + `qdq_4_rdna2.cuh` | `gemm/w4a16_fdot2` | One family (`use_v2_format`). ConfigA for `M>256`. Dest deleted `q_gemm_rdna2_awq_prefill.cu` @ `1046782`. ZP still scale-baked. K-split still unaligned. |
| `csrc/rocm/moe_q_gemm_rdna2.cu` | `moe/routed` | Reuses W4 helpers. |
| `csrc/rocm/exl3_dot2_{dense,moe,dequant,hadamard}.*` | `gemm/exl3_3inst` | `LDS_PAD=8`. Produce stays `-cb 3inst` outside. |
| `csrc/rocm/causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, wave32. FIR on pre-shift then shift. |
| `csrc/rocm/indexer_paged_mqa_rdna2.cu` | `attention/qsa_indexer` | DeepSeek-class indexer. |
| `csrc/rocm/qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | `VLLM_RDNA_QSA_HIP` default **off**. |
| `csrc/rocm/sparse_mla_rdna2.cu` | `attention/dsa_nope` | Sparse MLA class. |
| `csrc/rocm/rdna_allreduce.{cu,cuh}` | `comm/pcie` | Uncached+push Later. Occupancy pin closed. INT8/Q8 wire; no Finegrained. |
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
   `plan`/`run`. `run` is `NOT_READY`. extras @ `50120e13b468` has no
   `torch.ops.hippihx.*`.
4. extras is rewired **in extras**. The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs. Tracker:
[`EXTRAS.md`](EXTRAS.md).

## Dest extras defects

Dest tip `50120e13b468`. Live dest bugs / rolled-back paths — hippihx
must not reproduce them.

| Defect | Where | Status @ `50120e13` | Zoo lock |
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
| Flash-Next FULL_AND_PIECEWISE one-request corruption | `compilation-config` FULL_AND_PIECEWISE at c=8 even with seq cap 6 | **Still live** on FULL. Dest production launcher uses PIECEWISE (`3bddd3c9`). | Stay extras. Zoo does not own graph mode. |

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
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | If dest ever locks that family, pick **their** commits |
| Explore PRs **#9/#10** sdot | **Not taken** | Not dest |
| extras PR **#12** CPU PLE / MTP startup | **Not taken** | Serve-only. Foreign: George Muravei-Alkhavoi. No HIP body. |
