# extras → hippihx backport review

Last checked `opengfx1030/vllm-rdna` `rdna_extras` @ `388a61b6f75f`
(2026-09-16 22:57 UTC). Dest **default branch is `rdna_extras`**, not
`main`. `main` is still unrelated upstream vLLM (`c00091e02670`,
2026-09-02) — **no dest HIP landed there**. **Do not copy kernel bodies
into this tree yet.** extras still has no `torch.ops.hippihx.*`.

Dest extras is **4 commits** past the previous hippihx lock
`8960a3bcbb17` (HC HIP isolated compute), **5** past `d0d577f1907c`,
**8** past `7e70e2400542` (bench_report mtime), **9** past
`f5cbbdfec494`, **10** past `4d25a0483912` (custom AR launcher), **12**
past `1ff73596d81a`, **44** past `820465315bde`, and **87** past
`1046782fb8c4` (AWQ prefill `.cu` deleted; one W4 family). That W4 lock
still holds. Dest **also** landed Flash-Next / Qwen4Exp **as serve**
(merged extras PR **#8**, stacked on recipe PR **#6**) plus opt-in HIP
scaffolding (`hc_rdna2.cu`, `qsa_rdna2.cu`, `ple_short_conv_rdna2.cu`,
`mrope_rdna2.cu`, `rdna_fused_glue.cu`). HC/QSA/PLE HIP gates default
**off**; dest tried S6 default-on (`cb0d4418`) and **reverted**
(`9c9509b3`). Triton remains dest SoT for those. GDN prefill HIP is now
**opt-in** (`VLLM_GDN_HIP_PREFILL=1`; default Triton/FLA @ `cd1231fd`).
`d1b200b1` dest-observed FA GQA prefill register-resident O (occupancy
pin closed — do not dump). `b78006a4` extras launcher caps
`max-num-seqs` at 6 (GDN batched-decode n≥8 still corrupts). `388a61b6`
removed the one-shot GDN `zero_()` state wipes. Do not dump
`qwen4_exp/` or `.cu` bodies here. Do not copy tok/s or extras maxdiff.

**Remove / no longer dest-state:** extras PR **#5** is **closed**. Dest
imported V620 TunableOp tables (`ca83d922`) and dest-landed Flash-Next
via PR **#8** (Aron Hsiao port `5765f57b4c41` + Blivion follow-ups). Do
not treat PR #5 as a mixed other-fork to merge. TunableOp CSVs and PLE
IPC stay extras. Explore PRs **#9** (W4A8 sdot4) and **#10** (W4A4
sdot8) are marked **not dest** — no sdot tile, no second W4 family.

W4 scale-baked ZP, unaligned prefill K-split (K=640 → 16×40), and GDN
HIP selected on BF16 are **still live**. Dest tried ConfigH (`K_STEP=64`)
and **reverted** it (`7ac98a26`) — garbage for `M>256`. Keep ConfigA.
Flash-Next QSA abort is a Triton `forward_qsa` GPU trap; dest tile env
`VLLM_RDNA_QSA_WARPS=2` did **not** fix it. hippihx V1 still refuses
bf16 on DOT/GDN. See [Dest extras defects](#dest-extras-defects-do-not-copy).

**Consume ABI started.** `include/hippihx/v1.h` + `tiles/v1_abi.cpp` ship
the torch-free C entry family (`hippihx_v1_plan` / `hippihx_v1_run`) that
extras will wrap as one `torch.ops.hippihx.*` per op. ABI rev **3**
renames qualnames `attn.*` → `attention.*` (ids unchanged; extras has
not bound). `hippihx_v1_run` returns `HIPPIHX_V1_ERR_NOT_READY` until a
body migrates.

Delta since `a4060647cfbb`: dest extras is **far ahead** (GDN arenas,
one W4 family @ `1046782`, then through `820465315bde` into tip
`1ff73596d81a`: causal_conv out-stride/null-block, M-RoPE HIP, FA
GQA-subgroup prefill, W4 TP=4 via breakable cudagraphs, Hybrid W4 gfx10
ungate, Flash-Next serve + opt-in HIP scaffolding, PIX topology helpers,
QSA abort diagnostic, then GDN decode fp16 SSM state, GDN prefill
backend `'rdna2'`, W4 MoE oracle, moe_align prealloc, PLE `Tensor?`
schema, page-commit `new_zeros`/`zeros_like`, wvSplitK n≤5, V1
FULL→PIECEWISE + persist keepalive, then custom AR under breakable
cudagraphs (`849292ec`) and dest gfx1030 launcher default-on for
`VLLM_FORCE_CUSTOM_ALL_REDUCE` (`4d25a048`; leapdragon `rdna_ar` still
opt-in), then extras `bench_report.py` (`f5cbbdfe`), then bench_report
newest-by-mtime (`7e70e240`), dest tried Qwen4Exp HIP default-on then
reverted (`cb0d4418` / `9c9509b3`), then extras `_custom_ops` wrappers
(`d0d577f1`), then extras HC HIP isolated compute (`8960a3bc`), then FA
GQA register-resident O (`d1b200b1`), GDN prefill HIP opt-in
(`cd1231fd`), extras seq cap 6 (`b78006a4`), GDN `zero_()` wipe
removed (`388a61b6`).
GitHub squash-merged
[PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1) (`rdna_ar`
Uncached+push) at `a4060647`. Open: PRs **#2** (GLM Later) and **#3**
(a17t WIP). Draft **#12** (Intel CPU PLE / V620 MTP startup) — **not
dest**, no kernels, extras serve. **#4** closed after landing. **#5**
closed (dest imported TunableOp; Flash-Next dest-landed via **#8**).
**#6** merged (recipes + Hybrid W4 gfx10). **#7** merged (PIX topology;
AR policy unchanged). **#8** closed after landing Flash-Next. Closed
**#11** (wvSplitK) was cherry-picked as `c350fa218` — **skip the PR**,
dest already has it. Draft **#9/#10** explore sdot — **skip**. `main`
is unrelated upstream vLLM (`c00091e02670`). Issues are disabled. WIP
`rdna_extras_wip_20260910` is a pre-`cafe95ef8` snapshot — **not dest**.

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

hippihx is the zoo. extras is serve wiring. Dest journals GDN
capture as fixed for one FPP13 cell and dest-landed Flash-Next as
serve; extras still has no `torch.ops.hippihx.*` consume path, and dest
HIP is still ATen-coupled (plus persist keepalive). Copying `.cu` here
would freeze a second ISA copy. Dual copies are how serve bugs accrete.

## Verdict

| Source | Pertinent to hippihx? | Action |
|---|---|---|
| Dest tip ISA (`fa_rdna2`, EXL3, W4A16, GDN, causal_conv) | **Yes, later** — zoo class | Wait. Record observed locks. **One** W4 prefill (`gptq_gemm_rdna2_prefill` + `use_v2_format`; ConfigA for `M>256`). **Do not copy dest W4 ZP / K-split / ConfigH / persist keepalive.** Do not reintroduce ConfigH (`K_STEP=64`). Migrate only when extras can *call* hippihx. |
| Dest tip FA GQA-subgroup prefill (`ecfec4e412ad`) | **Yes, later** — `attention/fa_fdot2` observation | `VLLM_FA_RDNA2_GQA_MODE` default `subgroup`. Occupancy pin closed. Dest @ `d1b200b1`: GQA prefill O accumulator is register-resident (was smem). Do not dump `.cu`. Do not copy tok/s. Persist O stay extras. |
| Dest tip causal_conv out-stride + null-block (`82b6f183da2a`) | **Yes, later** — `sequence/causal_conv` | Record ISA. FIR still pre-shift then shift. |
| Dest tip M-RoPE / gated RMS / Flash-Next HIP scaffolding | **Watch / extras** | `mrope_rdna2.cu` fp16 scalar, no LDS, no fdot2. HC/QSA/PLE HIP **opt-in** (`VLLM_RDNA_{HC_PREFILL,QSA,PLE_CONV}_HIP` default **off**). Dest tried S6 default-on (`cb0d4418`) and **reverted** (`9c9509b3`). Wrappers @ `d0d577f1`. Isolated HC HIP compute dest-fixed @ `8960a3bc`; capture still not dest-on. `rdna_fused_glue.cu` is product fuse. No new V1 op until dest locks a class and extras can bind. |
| Dest tip GDN arenas + persist keepalive + TP=4 W4 serve | **No** | Stay extras. Arenas / `rdna2_graph_keepalive.cuh` / breakable cudagraphs / PYNCCL. |
| Dest tip Hybrid W4A16 gfx10 ungate | **No** | extras linear backend. RDNA2 W4 stays auto default on gfx1030. Not a second W4 family. |
| Dest tip leapdragon `rdna_ar` (PR #1) + PIX helpers (PR #7) | **Later** — `comm.pcie` | AR still opt-in (`VLLM_RDNA_AR=1`). Occupancy pin closed. Hop class is `hippihx.comm.fabric`; `lspci`/ACS helpers stay extras. Dest `c350fa218` device-index normalize is extras. |
| Dest tip GDN decode fp16 SSM state (`02adbfd4`) | **Yes, later** — `attention/gdn_scan` | Activations stay **fp16**. Dest HIP now accepts SSM state `fp16` or `fp32`; recurrence is still 16 **fp32** VGPR/thread. Flash-Next `mamba_cache_dtype=auto` is fp16, so HIP decode now fires. Do not dump `.cu`. Do not copy tok/s. |
| Dest tip GDN prefill backend `'rdna2'` (`ce3c9397`) | **No** | extras log/select. HIP chain was already the dispatch; dest only named it. Dispatch still misses dtype. Dest @ `cd1231fd`: prefill HIP is **opt-in** (`=1`); default Triton/FLA. |
| Dest tip page-commit `new_zeros` / `zeros_like` (GDN prefill fallback `143e2bf3`, HC HIP `3092d635`, HC Triton in `1ff73596`) | **No** | extras page-commit. Zoo: scratch from `plan`, serve zeros **once at allocation**. Dest @ `388a61b6` removed one-shot GDN `zero_()` wipes after prefill wrote state. Do not copy wrappers. |
| Dest tip W4 MoE oracle `RDNA2_W4A16` (`b549c229`) + moe_align prealloc (`4b799bf4`) | **No** | extras serve of existing `moe_q_gemm_rdna2`. Same W4 family. Graph-safety buffers stay extras. |
| Dest tip PLE `Tensor?` schema (`b26763e7`) | **No** | extras `.so` load (67 schemas). PLE HIP still opt-in / default off. |
| Dest tip wvSplitK n≤5 (`c350fa218`, closed extras PR **#11**) | **No** | extras dense GEMM dispatch. **No zoo tile.** Do not grow `gemv_f16`. |
| Dest tip V1 FULL→PIECEWISE + persist keepalive (`1ff73596`) | **No** | extras V1 runner. Persist keepalive / capture stay extras. extras PR **#12** would restrict the ROCm FULL redirect to compiled piecewise — **not dest**. Do not copy. |
| Dest tip custom AR under breakable cudagraphs (`849292ec`) | **No** | extras `custom_all_reduce.py` (vLLM/ROCm path, not leapdragon `rdna_ar`). Drop registered-buffer shortcut; never `empty_like` on a real forward; cache output; ask `capturing_segment`. Matches zoo: serve zeros, no D2H. Do not dump. Do not copy tok/s. |
| Dest tip gfx1030 launcher custom AR default-on (`4d25a048`) | **No** | extras `serve_gfx1030_full.sh`: `VLLM_FORCE_CUSTOM_ALL_REDUCE` default **1**. `envs.py` still False. leapdragon `VLLM_RDNA_AR` still **0**. |
| Dest tip `bench_report.py` (`f5cbbdfe` / mtime `7e70e240`) | **No** | extras ops reporter (PP / TG / TTFT). Newest result dir by mtime, not lexical name. Do not copy tok/s into tile locks. |
| Dest tip Qwen4Exp HIP S6 default-on then revert (`cb0d4418` / `9c9509b3`) | **No** | extras `envs.py` only. Gates still default **off**. Do not copy tok/s. |
| Dest tip Qwen4Exp `_custom_ops` wrappers (`d0d577f1`) | **No** | extras Python wrappers for existing HC/QSA/PLE HIP. Gates still off. Do not dump `.cu`. |
| Dest tip Qwen4Exp HC HIP compute (`8960a3bc`) | **No** | extras product HC. Isolated compute dest-fixed (runtime `GROUP_DIM`, contig fp16, `new_zeros`, weight index). Gate still **off**: PIECEWISE capture still faults in MoE. Do not dump `.cu`. Do not copy extras maxdiff. |
| Dest tip FA GQA register-resident O (`d1b200b1`) | **Yes, later** — `attention/fa_fdot2` | Observe. Occupancy pin closed. Do not dump `.cu`. Do not copy tok/s. |
| Dest tip GDN prefill HIP opt-in (`cd1231fd`) | **No** | extras env. Default Triton/FLA. HIP chain `VLLM_GDN_HIP_PREFILL=1`. Do not copy tok/s. |
| Dest tip gfx1030 `max-num-seqs` 6 (`b78006a4`) | **No** | extras launcher. GDN batched-decode n≥8 still corrupts. Stay extras. |
| Dest tip GDN `zero_()` wipe removed (`388a61b6`) | **No** | extras serve. Zeros at allocation only; never wipe live state after prefill. HIP decode kernel kept — do not dump. |
| PR #2 GLM-5.3 KDA/DSA (`later/glm53-…`) | **Later** — `kda_scan` / `dsa_nope` / `qsa_indexer` | Product-named `glm5_*` files. Do not name tiles after GLM. |
| PR #3 a17t `[WIP] Similar work, different fork` | **No** | WIP, mixed Triton+HIP+qwen4_exp. Duplicate W4 family. |
| PR #5 Flash-Next draft | **Closed** | Dest absorbed TunableOp + Flash-Next via **#8**. Do not merge the old other-fork PR. |
| Explore PRs **#9** / **#10** (W4A8 sdot4 / W4A4 sdot8) | **No** | Marked not dest. No sdot tile. |
| extras PR **#11** wvSplitK | **Closed** | Dest-picked as `c350fa218`. extras dense GEMM. |
| extras PR **#12** Intel CPU PLE / V620 MTP startup (`955322d6`) | **No** | Draft. No kernels. CPU PLE / `w2_zp` / GDN constructor device / FULL-redirect narrowing. Stay extras. Do not copy dest `1ff73596` FULL→PIECEWISE away. Do not copy startup seconds. Findings vs `3e1a0e1aa` are stale on dest-picked wvSplitK. |

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
is ISA-fixed (`i_t_local`). Prefill HIP is **opt-in**
(`VLLM_GDN_HIP_PREFILL=1`; default Triton/FLA @ `cd1231fd`). Dispatch still
   does not check dtype. Persist keepalive / immortal `hipMalloc` stay
   extras. Do not copy those into tiles.
5. CONTRIBUTING: do not edit `opengfx1030/vllm-rdna` from this repo.
   Without a hippihx consume bind, a copy here cannot replace extras.

## Dest file → tile (when migrate is allowed)

Observed at dest tip `388a61b6f75f` (ISA files + PR #1 AR @
`a4060647cfbb`; previous lock `8960a3bcbb17`). Numbers are extras
*observations*, not dest locks. Fill tile READMEs; do not invent tok/s.

| extras path | hippihx tile | Notes |
|---|---|---|
| `csrc/rocm/fa_rdna2.cu` | `attention/fa_fdot2` | ~33 KiB decode / ~48 KiB prefill smem. Occupancy pin closed. GQA-256 idle-wave + NaN guard. @ `ecfec4e412ad`: GQA-subgroup prefill (`HEADS_PER_CTA=2`, dual-acc fdot2); env `VLLM_FA_RDNA2_GQA_MODE` default `subgroup`. True-GQA dropped. Dest @ `d1b200b1`: GQA prefill O accumulator is register-resident (was smem). fp16 flash KV writer used for non-native KV. Persist workspaces stay extras. Do not dump. Do not copy tok/s. |
| `csrc/rocm/gdn_decode_rdna2.cu` | `attention/gdn_scan` | Register-resident 16 **fp32** VGPR/thread, no LDS. Dest @ `02adbfd4`: SSM state `fp16` or `fp32` (load/store `ST`; recurrence fp32). Activations stay fp16. `NULL_BLOCK_ID=0` is a **vLLM sentinel** — zoo contract is “invalid slot → zero out, do not touch state”, not that constant. |
| `csrc/rocm/gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | `o` kernel LDS ≈ 45312 B. Varlen uses **local** chunk `i_t_local`. Prefill HIP **opt-in** (`VLLM_GDN_HIP_PREFILL=1`; default Triton/FLA @ `cd1231fd`). Dispatch still misses dtype. |
| `csrc/rocm/q_gemm_rdna2.cu` + `q_gemm_rdna2_prefill.cu` + `qdq_4_rdna2.cuh` | `gemm/w4a16_fdot2` | GPTQ and AWQ are pack/zeros (`use_v2_format` → `zero_offset` 1/0), **one** GEMM family. Prefill `select_config`: ConfigA for `M>256` & `N>=4096`, else ConfigC. Dest **deleted** `q_gemm_rdna2_awq_prefill.cu` @ `1046782`. Dest **reverted** ConfigH (`K_STEP=64`). Dest ZP still scale-baked. K-split still unaligned. |
| `csrc/rocm/moe_q_gemm_rdna2.cu` | `moe/routed` | Reuses W4 helpers. |
| `csrc/rocm/exl3_dot2_{dense,moe,dequant,hadamard}.*` | `gemm/exl3_3inst` | `LDS_PAD=8` on A-staging. No `__launch_bounds__` on DOT kernels (VGPR is the occupancy lever). Produce stays `-cb 3inst` **outside**. |
| `csrc/rocm/causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, wave32, `state_len≈3–4`, register-only. FIR on **pre-shift** then shift. @ `82b6f183da2a`: separate `stride_o_token`; null-block early return. |
| `csrc/rocm/indexer_paged_mqa_rdna2.cu` | `attention/qsa_indexer` | Dest DeepSeek-class indexer. Confirm class vs Flash-Next QSA later. |
| `csrc/rocm/qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | Flash-Next store/compress/MQA HIP. Gate `VLLM_RDNA_QSA_HIP` default **off**. Not dest-on. Do not dump. |
| `csrc/rocm/sparse_mla_rdna2.cu` | `attention/dsa_nope` | Sparse MLA class, not a product fuse. |
| `csrc/rocm/rdna_allreduce.{cu,cuh}` (merged PR #1) | `comm/pcie` | Uncached+push Later. Host-coherent flags. Boot self-test. Communicator opt-in. Occupancy pin closed. INT8/Q8 wire preferred; no Finegrained. |
| `csrc/rocm/mrope_rdna2.cu` | — | fp16, one program/token, no LDS, no fdot2. Stay extras until a rotary class exists. |
| `csrc/rocm/{hc_rdna2,ple_short_conv_rdna2,rdna_fused_glue}.cu` | — | Flash-Next product HIP. HC/PLE HIP default off; fused HC/SE decode default on. Stay extras. |
| `later/glm53-…` `glm5_kda_*.cu` | `attention/kda_scan` | Later. Do not keep the `glm5_` prefix. Do not retarget GDN 16/48 onto KDA 64×128. |
| `later/glm53-…` `glm5_dsa_*.cu` | `attention/dsa_nope` + `qsa_indexer` | Later. Same rename rule. |

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
| extras PR **#5** Flash-Next draft / `rdna_extras_wip_20260910` | **Closed / not dest tip.** Dest absorbed TunableOp + Flash-Next via PR **#8**. Do not merge the old other-fork. |
| dest profiling findings / `docs/rdna2/bench_27b_awq_matrix.md` | Serve / ops. Do not copy tok/s or µs tables into tile locks. |
| `rdna2_w4a16.py` `_awq_prefill_available` + dead `awq_prefill` apply branch | Extras dead code after HIP delete. `select_kernel` already returns `"prefill"` for AWQ. |
| `vllm/models/qwen4_exp/**` + PLE offload / TunableOp CSVs | Product serve. Stay extras. |
| `mrope_rdna2.cu` / `hc_rdna2.cu` / `qsa_rdna2.cu` / `ple_short_conv_rdna2.cu` / `rdna_fused_glue.cu` | ATen-coupled scaffolding. HC/QSA/PLE HIP default off (S6 default-on reverted). Wrappers @ `d0d577f1`. Isolated HC compute @ `8960a3bc`; capture still not dest-on. No V1 op yet. |
| PIX topology helpers (PR **#7**) | Serve. AR policy unchanged (still opt-in). |
| Hybrid W4A16 gfx10 ungate | extras linear backend, not a hippihx family. |
| Explore PRs **#9** / **#10** sdot | Not dest. |
| V1 runner FULL→PIECEWISE + persist keepalive (`1ff73596`) | extras capture. Zoo does not own graph mode. |
| W4 MoE oracle `RDNA2_W4A16` + moe_align / scale-zero prealloc | extras graph-safety around existing `moe_q_gemm_rdna2`. |
| wvSplitK n≤5 decode (`c350fa218`) | extras dense GEMM. No tile. |
| PLE `Tensor?` schema (`b26763e7`) | extras `.so` load. HIP still opt-in. |
| HC / GDN `new_zeros` / `zeros_like` | extras page-commit. Confirm zoo: serve zeros. |
| extras vLLM custom AR (`849292ec` / launcher `4d25a048`) | extras capture + gfx1030 launcher. Not `comm/pcie` Uncached+push. Never `empty_like` on a real forward. |
| extras `bench_report.py` (`f5cbbdfe` / mtime `7e70e240`) | extras ops. Newest by mtime. Do not copy tok/s. |
| extras Qwen4Exp HIP S6 default-on (`cb0d4418`) then revert (`9c9509b3`) | dest-reverted. Gates still default off. Do not copy tok/s. |
| extras Qwen4Exp `_custom_ops` wrappers (`d0d577f1`) | extras Python wrappers. Still opt-in. Do not dump `.cu`. |
| extras Qwen4Exp HC HIP compute (`8960a3bc`) | extras product HC. Isolated compute dest-fixed; capture still not dest-on. Do not dump `.cu`. Do not copy extras maxdiff. |
| extras FA GQA register-resident O (`d1b200b1`) | ISA observation. Occupancy pin closed. Do not dump `.cu`. Do not copy tok/s. |
| extras GDN prefill HIP opt-in (`cd1231fd`) | extras env. Default Triton/FLA. Do not copy tok/s. |
| extras gfx1030 `max-num-seqs` 6 (`b78006a4`) | extras launcher. GDN batched-decode n≥8 still live. |
| extras GDN `zero_()` wipe removed (`388a61b6`) | extras serve. Zeros at allocation only. Do not dump HIP decode. |
| extras PR **#12** Intel CPU PLE / V620 MTP startup | Draft. No kernels. Stay extras. Do not copy FULL→PIECEWISE narrowing or startup seconds. |
| extras PR **#11** wvSplitK | **Closed / dest-picked** as `c350fa218`. Do not re-merge. |

## Do not take from a17t PR #3

Unique vs dest: `awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`,
`gemv_w4_kpack_rdna2.cu`, `qdq_awq_rdna2.cuh`. Dest already has AWQ as a
**zeros mode** on `q_gemm_rdna2*` (`use_v2_format`). Dest **deleted**
`q_gemm_rdna2_awq_prefill.cu` @ `1046782`. hippihx must not grow a second
W4 GEMM. Dest `qwen4_exp` (PR **#8**) is extras serve, not an a17t tile
to copy. Triton fallbacks and recipes stay extras.

Dest skinny GEMM is extras consume (default on). Do not grow a
`gemv_f16` zoo tile. Explore sdot4/sdot8 (PRs **#9/#10**) are not dest.

## When a migrate *is* pertinent

All of:

1. Dest GDN hybrid is capture-safe (or GDN HIP stays opt-in and documented).
   **Dest journals FPP13 16k c=8 green** @ `6c5ff94` via serve arenas.
   Still extras; still no hippihx consume. Not a body dump.
2. Tile README LDS / `__launch_bounds__` / wave / DOT unit are filled from
   extras observations (this review started that; EXL3 grain-v2 + FA
   `__launch_bounds__` 128/256 recorded @ `a4060647`; FA flash KV writer
   + GQA-256 hygiene @ `6c5ff94`; GQA-subgroup prefill @ `ecfec4e412ad`).
3. hippihx ships one HIP entry that extras can bind as one V1 op.
   **Started:** `include/hippihx/v1.h` (`hippihx_v1_plan` / `hippihx_v1_run`).
   `run` is still `NOT_READY` (no body). Extras has not rewired yet
   (`388a61b6f75f` still has no `torch.ops.hippihx.*`).
4. extras is rewired to call it (that edit happens **in extras**, not from
   this tree). The extras copy is then deleted.

Until then: observe, lock numbers, keep stubs, grow the V1 ABI. Kernel /
mode / env / AR tracker (all **unvalidated**):
[`EXTRAS.md`](EXTRAS.md).

## Dest extras defects (do not copy)

Dest tip is `388a61b6f75f` (4 commits past `8960a3`; 5 past `d0d577`;
8 past `7e70e2`; 9 past `f5cbbd`; 10 past `4d25a0`; 12 past `1ff735`;
44 past `820465`; 87 past `1046782`). These are **live dest bugs /
rolled-back paths**, not tok/s. hippihx contracts must not reproduce
them. Dest **fixed / dropped**: GDN state arenas, prefill `o`
`i_t_local`, conv FIR order, **separate AWQ prefill `.cu`**,
causal_conv out-stride/null-block, ConfigH (`K_STEP=64`), gfx10 hybrid
bf16 *serve* abort (reject combo), GDN decode HIP skipped on fp16 SSM
state, custom AR garbage under breakable cudagraphs (`849292ec`),
Qwen4Exp HIP S6 default-on (`9c9509b3` revert), Qwen4Exp `_custom_ops`
wrappers (`d0d577f1`), isolated HC HIP compute (`8960a3bc`; gate still
off), GDN one-shot `zero_()` wipes (`388a61b6`). Do not copy persist
keepalive, FULL→PIECEWISE, or tok/s.

| Defect | Where on dest extras | Status @ `388a61` | Proper zoo lock | hippihx |
|---|---|---|---|---|
| `llvm.amdgcn.fdot2.bf16.bf16` ISel abort | Hybrid W4 gfx10 + bf16 used to abort Triton off skinny. Dest HIP has **no** `fdot2.bf16`. HIP conv stays scalar FMA. | **Dest-mitigated in serve** (`59237b3`: reject gfx10 hybrid bf16). HIP lock unchanged. | Never `fdot2.bf16`. DOT + GDN HIP are **fp16 activations**. BF16 leftover / conv = scalar FMA, fp32 mul. | `dot.hpp`, V1 `HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`, `sequence/causal_conv`, `moe/leftover_bf16` |
| Scale-baked W4 zero-point | `qdq_4_rdna2.cuh` `prep_zero_scale_fp16`: `0xE400 \| zero` then `scale * (-1024 - zero)` in `half`. All-zero weights were not exact zero. | **Still live** | Integer `q - zero`, then `* scale`. GPTQ `uint4b8` (+1) vs AWQ literal is pack/zeros, not a second GEMM. | `gemm/w4a16_fdot2` |
| Unaligned prefill K-split | `q_gemm_rdna2_prefill.cu` `compute_split_k`: K=640 can pick 16 splits of 40. Kernel `K_STEP=32`. Findings proposed cap `split_k` at 8 and require `k_per_split % K_STEP == 0` — **not landed**. | **Still live** | Equal `k_per_split`, multiple of 32, inside LDS budget. Refuse the 40-wide split. | `gemm/w4a16_fdot2` |
| Prefill ConfigH `K_STEP=64` | Tried as faster large-M tile; garbage for `M>256`. | **Dest-reverted** (`7ac98a26`); later dropped from the kernel. Inner loop now `K_STEP/8`. | Keep ConfigA (`K_STEP=32`) for `M>256`. Do not reintroduce ConfigH. | `gemm/w4a16_fdot2` |
| GDN HIP selected on BF16 | `_gdn_prefill_dispatch_available()` checks GPU + symbols, **not dtype**. Kernel still `mixed_qkv` fp16. | **Still live** | `plan` / V1 refuse bf16 on `attention.gdn_scan`. Prefill HIP is opt-in (`=1`). | `attention/gdn_scan`, V1 dtype |
| GDN decode HIP skipped on fp16 SSM state | Dispatch required `ssm_state.dtype == float32`. Flash-Next `--dtype float16` + `mamba_cache_dtype=auto` is fp16, so HIP never fired. | **Dest-fixed** (`02adbfd4`): `ST` = `float` or `half`; recurrence stays 16 fp32 VGPR. | Activations **fp16**. SSM state **fp16 or fp32**. Refuse bf16 act. Do not dump the `ST` template. | `attention/gdn_scan` |
| GDN prefill `o` varlen chunk index | `gdn_prefill_o_rdna2.cu` used global `i_t` so later sequences were OOB. | **Dest-fixed** (`i_t_local`) | Token offsets use per-sequence local chunk. | `attention/gdn_scan` |
| GDN piecewise capture state poison | Pool-block conv/ssm pages; per-step `block_table` views. | **Dest-fixed (serve)** — permanent BS-sized arenas | Scratch/state from `plan`; no realloc under capture. Arenas stay extras. | extras, not a tile |
| causal_conv update vs fwd FIR | Decode kernel shifted state **before** the FIR (mismatched fwd). | **Dest-fixed** (`cafe95ef8`) | FIR on pre-shift state, then shift. Scalar FMA. | `sequence/causal_conv` |
| causal_conv fwd out-stride / null-block | Fwd used the wrong out stride; null slots wrote. | **Dest-fixed** (`82b6f183da2a`) | Use `stride_o_token`; invalid slot → skip. | `sequence/causal_conv` |
| Global EXL3 FP16 clip | Serve `Qwen2MoeMLP` clipped every model to FP16. Rolled back to EXL3-only on the integration fork. | Serve | Not a tile. Stay in extras. | — |
| QSA / Flash-Next attention abort | Triton `forward_qsa` GPU trap under capture (`820465` tracker). Dest `VLLM_RDNA_QSA_WARPS=2` did **not** fix it. | **Serve-mitigated** — eager QSA break (`8cf0dedb`); dest V1 maps FULL→PIECEWISE (`1ff73596`). HIP `qsa_rdna2.cu` still opt-in / default **off**. | Indexer occupancy observation: **4 warps** for gfx1030 BF16 6h×256. Do not dump Triton QSA or `qsa_rdna2.cu`. | `attention/qsa_indexer` (watch) |
| Custom AR garbage under breakable graphs | vLLM `custom_all_reduce.py`: `registered=True` shortcut + `empty_like` on a real forward (not leapdragon `rdna_ar`). | **Dest-fixed (serve)** (`849292ec`). gfx1030 launcher now defaults `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` (`4d25a048`). | Never `empty_like` on a real forward. Zoo Uncached+push stays opt-in (`VLLM_RDNA_AR=0`). Do not dump. | extras; `comm/pcie` Later |
| Separate AWQ prefill `.cu` | `q_gemm_rdna2_awq_prefill.cu` (exllama-clone, `BLOCK_M=16`) | **Dest-deleted** (`1046782`). HIP binding gone. Python leftover `_awq_prefill_available` is extras dead code. Dest path doc still *names* the deleted op — do not treat that as dest. | One W4 family. Pack/zeros only. Do not reintroduce. | `gemm/w4a16_fdot2` |
| Qwen4Exp HIP S6 default-on | `envs.py` `VLLM_RDNA_{HC_PREFILL,QSA,PLE_CONV}_HIP` | **Dest-reverted** (`cb0d4418` then `9c9509b3`). Gates still default **off**. | Stay extras. No V1 op. Do not copy tok/s. | extras |
| Qwen4Exp HIP `_custom_ops` missing | HC/QSA/PLE called ops with no Python wrapper (inert / AttributeError if gated on) | **Dest-fixed (serve)** (`d0d577f1`). Gates still off. | Stay extras. Do not dump product HC. | extras |
| Qwen4Exp HC HIP compute / `GROUP_DIM` | `hc_rdna2.cu` + `hc_rdna2.py`: `BLOCK<=512` reject, `float local[BLOCK]`, weight index OOB, strided views, `new_empty` | **Dest-fixed isolated** (`8960a3bc`). Gate still **off**: PIECEWISE capture still faults in MoE. | Stay extras. Serve zeros. No V1 op. Do not dump `.cu`. Do not copy extras maxdiff. | extras |
| GDN one-shot `zero_()` state wipe | `qwen_gdn_linear_attn.py` zeroed paged ssm/conv cache at first `gdn_decode_rdna2` after prefill wrote state | **Dest-fixed (serve)** (`388a61b6`). KV already zeros at allocate. | Zeros **once at allocation**. Never wipe live state. Do not dump HIP decode. | extras; `attention/gdn_scan` |
| GDN batched-decode n≥8 | hybrid GDN corrupts recurrent state when ≥8 sequences decode in one batch | **Still live.** extras launcher caps `max-num-seqs` 6 (`b78006a4`). | Stay extras. Do not copy tok/s. Zoo: zeroed state from `plan`; no serve seq-cap in tiles. | extras |

Serve rollbacks / capture notes (keep as serve, not zoo):

- GDN HIP prefill is **opt-in** (`VLLM_GDN_HIP_PREFILL=1`); default is
  Triton/FLA (`cd1231fd`). Recipes may still export `0` or `1`.
- RDNA_ATTN spec/MTP verify **default-off** (opt-in env).
- `eager_break_during_capture` on `do_kv_cache_update` stays extras.
- FA persist / immortal workspaces and `rdna2_graph_keepalive.cuh`
  (`c091c420b`) are mixed capture plumbing + debug `fprintf`. Skip that
  commit as a body pick; extract ISA only.
- GDN piecewise-capture **root cause** on dest is now arenas, not
  “eager is the only control”. Do not copy probe-driven ISA.

Skinny GEMM (`skinny_gemms*.cu`, `VLLM_ROCM_USE_SKINNY_GEMM` default
True, `VLLM_ROCM_MOE_SKINNY` default on after `59237b3`) is extras
consume. **No hippihx tile.** Do not reintroduce leapdragon `gemv_f16`
as a zoo family. Explore sdot4/sdot8 (PRs **#9/#10**) are not dest.

Flash-Next QSA (`820465` tracker) stays extras. Eager Triton now
serves (`8cf0dedb`); dest V1 maps FULL→PIECEWISE (`1ff73596`). HIP
`qsa_rdna2.cu` is still default-off. Do not dump.

## Attribution (do not re-author)

**Dest extras HIP** (FA, EXL3, W4A16, GDN, causal_conv, sparse MLA,
indexer, M-RoPE, dest follow-ups on Flash-Next HIP) is **BlivionIaG**
`<kev29lt@gmail.com>`. Author **and** Committer on every picked dest
commit. Do not use Cursor or `cursoragent`.

**Foreign HIP** (leapdragon, a17t, Aron Hsiao Flash-Next / recipe / Hybrid
W4 port commits on dest extras) keeps **their** Author **and**
Committer. Cherry-pick `-x`, force Committer = source Author, keep their
trailers, add none of ours. Recipe: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
Same bar as extras PR #1. Do not pick dest Blivion follow-ups as if they
were Aron’s unique commits.

This hippihx review commit is **new documentation**. It is not a
kernel migrate and must not be used as a template for body imports.

| extras / PR path | Introduced | Keep as |
|---|---|---|
| `fa_rdna2.cu` | `b1b3fa938` BlivionIaG `<kev29lt@gmail.com>` | BlivionIaG. Later FA `torch::zeros` is extras serve — leave it there |
| `q_gemm_rdna2.cu` | `fabf51493` BlivionIaG | BlivionIaG |
| `q_gemm_rdna2_awq_prefill.cu` | `feb7b457e` BlivionIaG | **Deleted** dest @ `1046782`. Do not reintroduce. |
| `moe_q_gemm_rdna2.cu` | `b1b3fa938` BlivionIaG | BlivionIaG |
| `gdn_decode_rdna2.cu` | `55527010c` BlivionIaG | BlivionIaG. Dest @ `02adbfd4` fp16 SSM state. NULL_BLOCK_ID sentinel is extras serve until dest-locked |
| `exl3_dot2_*.cu` | `40850e6c5` BlivionIaG | BlivionIaG (Author **and** Committer), including the port and later perf commits |
| `causal_conv1d_rdna2.cu` | `5eb84b4fa` BlivionIaG | BlivionIaG. Out-stride / null-block @ `82b6f183da2a` |
| `mrope_rdna2.cu` | `cf055ad1d47c` BlivionIaG | BlivionIaG. Stay extras until a rotary class exists |
| `qsa_rdna2.cu` / `hc_rdna2.cu` / `ple_short_conv_rdna2.cu` | dest 2026-09-14 scaffolding | Blivion dest follow-ups. Opt-in; not dest-on. Do not dump |
| `sparse_mla_rdna2.cu` / `indexer_paged_mqa_rdna2.cu` | BlivionIaG | BlivionIaG |
| dest `rdna_ar` (merged PR #1) | Unique HIP: **Aron Hsiao** `<leapdragon@gmail.com>` | Pick unique commits (`af25c5329`…`ee6e48ea1`), Author **and** Committer Aron Hsiao. Keep `Co-Authored-By: Claude Fable 5`. **Do not pick squash `a4060647`.** No Blivion/Cursor trailers |
| dest Flash-Next / Hybrid W4 / recipe ports | Unique: **Aron Hsiao** `5765f57b4c41` / `c05af408775f` / `22bb2e8d06f3` | Pick **their** unique commits if dest-locked. Dest Blivion follow-ups stay Blivion |
| PR #2 `glm5_kda_*` / `glm5_dsa_*` | BlivionIaG | BlivionIaG; rename off `glm5_` in a **follow-up hippihx** commit, not by rewriting their HIP |
| PR #3 a17t unique W4 (`d53572644`, later Simon Siebert) | **Not taken** | If dest ever locks that family, pick **their** commits, not a rewrite |
| Explore PRs **#9/#10** sdot | **Not taken** | Not dest |
| extras PR **#12** CPU PLE / MTP startup | **Not taken** | Serve-only. Foreign: George Muravei-Alkhavoi. If dest picks, keep **their** Author. No HIP body. |
