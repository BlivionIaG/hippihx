# extras inventory (unvalidated)

Moved out of the root README so the zoo stays lean. This is a tracker,
not a claim that any row works. Bodies stay in extras until extras can
consume one V1 op. Details: [`BACKPORT.md`](BACKPORT.md).

## Unvalidated extras inventory

Snapshot of [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `8960a3bcbb17` (2026-09-16 16:18 UTC; 1 commit past
`d0d577f1907c`, 4 past `7e70e2400542`, 5 past `f5cbbdfec494`, 6 past
`4d25a0483912`, 8 past `1ff73596d81a`, 40 past `820465315bde`, 83 past
`1046782fb8c4`, which
deleted the unused AWQ prefill `.cu`; merged extras
[PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1) is still
`a4060647cfbb`) plus open extras PRs **#2–#3**. Dest default branch is
**`rdna_extras`**. `main` is still upstream vLLM `c00091e02670` — no
dest HIP there. **No** `torch.ops.hippihx.*` consume bind yet.

**Unvalidated.** Not dest. Not silicon-signed. No tok/s. hippihx still
ships stubs; bodies stay in extras until a consume bind exists. This
list is a tracker, not a claim that any row works.

Dest extras journals GDN FPP13 16k c=8 capture **green** via serve
arenas, **deleted** the separate AWQ prefill `.cu` (one W4 family),
**reverted** ConfigH, dest-landed Flash-Next **as serve** (HIP
HC/QSA/PLE still opt-in / default off; dest tried S6 default-on
`cb0d4418` and reverted `9c9509b3`; wrappers @ `d0d577f1`; isolated HC
HIP compute @ `8960a3bc`, gate still off), and since
`820465` also: GDN
decode **fp16 SSM state** (`02adbfd4`), GDN prefill backend `'rdna2'`,
W4 MoE oracle `RDNA2_W4A16`, moe_align prealloc, PLE `Tensor?` schema
(67 `_rocm_C` schemas), page-commit `new_zeros`/`zeros_like`, wvSplitK
n≤5, V1 FULL→PIECEWISE + persist keepalive, custom AR under breakable
cudagraphs (`849292ec`), dest gfx1030 launcher default-on for
`VLLM_FORCE_CUSTOM_ALL_REDUCE` (`4d25a048`; leapdragon `rdna_ar` still
opt-in), extras `bench_report.py` (`f5cbbdfe` / newest-by-mtime
`7e70e240`), Qwen4Exp HIP wrappers (`d0d577f1`), HC HIP isolated
compute (`8960a3bc`). Do **not** copy dest W4
scale-baked ZP, unaligned prefill K-splits, persist keepalive, GDN
HIP-on-BF16 dispatch, ConfigH, Triton QSA, or tok/s. V1 refuses bf16 on
DOT and GDN HIP. Details:
[`BACKPORT.md`](BACKPORT.md#dest-extras-defects-do-not-copy).

Status key: **extras** = live on dest tip (unvalidated here) ·
**Later** = side branch / review-only · **skip** = do not take ·
**stub** = hippihx contract only.

### Kernels (HIP)

| Kernel / op | extras | hippihx tile | Notes |
|---|---|---|---|
| FA paged decode / prefill / split-K / short / GQA | `fa_rdna2.cu` | `attention/fa_fdot2` | `fdot2`. Occupancy pin closed. GQA-256 NaN/idle-wave. GQA-subgroup prefill default (`VLLM_FA_RDNA2_GQA_MODE=subgroup`). Persist O stay extras. **unvalidated** |
| FA INT8 KV writer | `reshape_and_cache_int8_rdna2` | `attention/fa_fdot2` | INT8 cache layout for RDNA_ATTN. **unvalidated** |
| FA fp16 flash KV writer | `reshape_and_cache_flash_rdna2` | `attention/fa_fdot2` | Non-native KV write. `__launch_bounds__(128, 4)`. fp16 only. **unvalidated** |
| W4A16 dense decode | `q_gemm_rdna2.cu` | `gemm/w4a16_fdot2` | GPTQ + AWQ = pack/zeros, one GEMM. Dest ZP is scale-baked `half` — zoo uses integer `q-zero` then scale. **unvalidated** |
| W4A16 prefill | `q_gemm_rdna2_prefill.cu` | `gemm/w4a16_fdot2` | Unified GPTQ+AWQ. ConfigA for `M>256`. Dest **reverted** ConfigH. Dest `compute_split_k` can pick K=640→16×40; zoo requires equal ×32. **unvalidated** |
| W4A16 AWQ high-M prefill | ~~`q_gemm_rdna2_awq_prefill.cu`~~ | `gemm/w4a16_fdot2` | **Dest-deleted** @ `1046782`. Do not reintroduce. |
| W4A16 MoE | `moe_q_gemm_rdna2.cu` | `moe/routed` | Dest oracle now selects `RDNA2_W4A16` (`b549c229`). Same W4 family. moe_align prealloc is extras. **unvalidated** |
| EXL3 dense / MoE / dequant / Hadamard / trellis decode | `exl3_dot2_*.cu` | `gemm/exl3_3inst` | Consume `-cb 3inst`. Produce outside. UNC-26. **unvalidated** |
| GDN packed decode | `gdn_decode_rdna2.cu` | `attention/gdn_scan` | Register-resident. Dest journals FPP13 16k c=8 green via **serve arenas**. Dest @ `02adbfd4`: SSM state **fp16 or fp32**; recurrence fp32 VGPR. **fp16 act only.** **unvalidated** |
| GDN prefill chain (prep / kkt / solve_wy / delta_h / o) | `gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | Opt-out (`VLLM_GDN_HIP_PREFILL=0`). `o` varlen `i_t_local` dest-fixed. Dispatch still misses dtype. **unvalidated** |
| causal_conv1d update + fwd | `causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, `state_len≈3–4`. FIR on pre-shift then shift. Fwd `stride_o_token` + null-block. BF16: fp32-promoted mul, **not** `fdot2.bf16`. **unvalidated** |
| Paged MQA indexer | `indexer_paged_mqa_rdna2.cu` | `attention/qsa_indexer` | DeepSeek V4 Lightning class. gfx1030 BF16 6h×256: **4 warps**. **unvalidated** |
| Flash-Next QSA store/compress/MQA | `qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | `VLLM_RDNA_QSA_HIP` default **off**. Eager Triton now serves (`8cf0dedb`); dest V1 maps FULL→PIECEWISE. **unvalidated** |
| Sparse MLA decode / prefill | `sparse_mla_rdna2.cu` | `attention/dsa_nope` | **unvalidated** |
| M-RoPE forward | `mrope_rdna2.cu` | — | fp16, no LDS, no fdot2. No tile until dest locks a rotary class. **unvalidated** |
| Flash-Next HC / PLE conv / fused glue | `hc_rdna2.cu`, `ple_short_conv_rdna2.cu`, `rdna_fused_glue.cu` | — | Product HIP. HC/PLE default off (S6 default-on reverted `9c9509b3`); wrappers @ `d0d577f1`; isolated HC compute @ `8960a3bc` (capture still not dest-on). Stay extras. **unvalidated** |
| W8A16 / W8A16-FP8 / W8A8-FP8 dense+MoE | `moe_w8a16*.cu`, `gemm_w8a8_fp8_dense_rdna2.cu`, `q_gemm_w8a16_fp8_rdna2.cu` | — | No hippihx tile yet. **unvalidated** |
| MXFP4 dense + MoE | `mxfp4_dot2_*.cu` | — | No hippihx tile yet. **unvalidated** |
| gfx1100 W4 WMMA | `q_gemm_rdna3_wmma.cu` | — | WMMA Later overlay, not shared DOT. **unvalidated** |
| Skinny GEMM / INT4 skinny | `skinny_gemms*.cu` | — | Dest default ON. **No tile.** Do not reintroduce leapdragon `gemv_f16`. **unvalidated** |
| RMSNorm / gated RMS HIP AOT | `layernorm.cu` | — | Serve fused-norm; no tile. **unvalidated** |
| GLM-5.3 KDA decode + prefill | `glm5_kda_*.cu` (PR **#2**) | `attention/kda_scan` | Later. Drop `glm5_` name. **unvalidated** |
| GLM-5.3 DSA indexer + MLA-NoPE | `glm5_dsa_*.cu` (PR **#2**) | `attention/dsa_nope`, `qsa_indexer` | Later. **unvalidated** |
| leapdragon push AR | `rdna_allreduce.{cu,cuh}` (merged PR **#1**) | `comm/pcie` | Dest extras, default **off**. PIX helpers (PR **#7**) stay extras. Occupancy pin closed. Aron Hsiao unique AR commits. **unvalidated** |
| a17t extra AWQ GEMM / GEMV | `awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`, `gemv_w4_kpack_rdna2.cu` (PR **#3**) | — | **skip** — second W4 family |
| Explore W4A8 sdot4 / W4A4 sdot8 | extras PRs **#9/#10** | — | **skip** — not dest |

### Modes / dispatch

| Mode | What extras does | Default (extras) | hippihx |
|---|---|---|---|
| W4 decode vs prefill vs Exllama | M/K/N buckets in `rdna2_w4a16.py` | decode `M≤32` & `K≥4096`; AWQ `M>32` → GPTQ prefill (ConfigA if `M>256`); GPTQ `M>256` still Exllama | one `w4a16_fdot2` tile |
| W4 pack | GPTQ `uint4b8` (+1 zeros) vs AWQ `uint4` (literal zeros) | same kernel, `use_v2_format` | pack/zeros, not a second GEMM |
| EXL3 codebook | `cb==0` 3inst produce, `cb==1` mcg compile, `cb==2` mul1 not produced | 3inst dest | consume only |
| EXL3 memory | `full` int16 trellis vs `packed` stub | `full` | — |
| EXL3 prefill | decode-trellis prefill vs fused GEMM | `VLLM_EXL3_PREFILL_DECODE=1` | — |
| GDN decode HIP | packed HIP vs Triton/FLA | on unless `VLLM_GDN_DECODE_RDNA2=0`; dest @ `02adbfd4` HIP fires on fp16 SSM state | `gdn_scan` |
| GDN prefill HIP | 5-kernel chain | **opt-out** (`== "0"`); dest log backend `'rdna2'` (`ce3c9397`) | `gdn_scan` |
| FA backend | `RDNA_ATTN` when gfx10x | `VLLM_USE_RDNA2_FA` (envs.py default false; backend reads `"1"`) | `fa_fdot2` |
| FA spec/MTP gate | opt-in abort on verify-shaped batches | **off** | serve, not a tile |
| MLA sparse HIP | indexer + sparse MLA | `VLLM_USE_RDNA2_MLA=1` | indexer / `dsa_nope` |
| causal conv HIP | update (decode) + fwd (prefill) | on unless set `0`; fwd dest-enabled with out-stride | `causal_conv` |
| FA GQA prefill | subgroup / true / off | **`subgroup`** (`VLLM_FA_RDNA2_GQA_MODE`) | `fa_fdot2` observation |
| Flash-Next HC/QSA/PLE HIP | dest scaffolding | **off** (`VLLM_RDNA_{HC_PREFILL,QSA,PLE_CONV}_HIP`); dest tried default-on `cb0d4418` and reverted `9c9509b3`; wrappers @ `d0d577f1`; isolated HC compute @ `8960a3bc` (PIECEWISE capture still faults in MoE); PLE schema loads (`b26763e7`) | extras until dest-on |
| Hybrid W4A16 gfx10 | extras linear backend | ungated; RDNA2 W4 stays auto default on gfx1030 | not a second W4 family |
| gfx1030 wvSplitK n≤5 | extras dense GEMM (`c350fa218`) | dest-on for n≤5 FP16/BF16 decode | **no tile** |
| V1 FULL_AND_PIECEWISE | dest maps FULL→PIECEWISE + persist keepalive | extras runner (`1ff73596`) | serve, not a tile |
| Custom AR (vLLM/ROCm) | force custom all-reduce on PCIe | **on** in dest gfx1030 launcher (`4d25a048`); `envs.py` still False; cudagraph-correct @ `849292ec` | serve, not a tile |
| leapdragon `rdna_ar` | size-gated Uncached+push | **off** (`VLLM_RDNA_AR=0`; dest extras after merged #1) | `comm/pcie` Later |

### Env (extras-added / extras-used)

Serve knobs. hippihx does not read these. **Unvalidated.** Debug probes stay extras (no D2H under capture in the zoo).

| Env | Default (as read in extras) | Role |
|---|---|---|
| `VLLM_USE_RDNA2_FA` | envs.py `False`; `rdna_attn` treats missing as `"1"` | FA-RDNA2 / `RDNA_ATTN` |
| `VLLM_USE_RDNA2_MLA` | off unless `"1"` | sparse MLA + paged MQA HIP |
| `VLLM_FARDNA2_ENABLE_SPEC_GATE` | `"0"` | MTP-verify abort (opt-in) |
| `VLLM_FARDNA2_SPEC_VERIFY_Q_LEN` | `"3"` | spec-gate q len |
| `VLLM_GDN_DECODE_RDNA2` | on (`!= "0"`) | GDN decode HIP |
| `VLLM_GDN_HIP_PREFILL` | off if `"0"` (opt-out; not a code default-off) | GDN prefill HIP chain |
| `VLLM_GDN_HIP_KERNELS` | recipe `1` | recipe umbrella; confirm vs the two gates above |
| `VLLM_GDN_DECODE_KERNEL` | `"cuda"` | FLA packed-decode path name (`cuda`/`triton`) |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` | `1` | FLA packed decode |
| `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` | `"1"` | conv update HIP |
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD` | `"1"` | conv fwd HIP |
| `VLLM_RDNA_FORCE_FP16` | recipe `1` | force fp16 (no BF16 emu) |
| `VLLM_EXL3_MEMORY_MODE` | `full` | `full` / `packed` |
| `VLLM_EXL3_PREFILL_DECODE` | `"1"` | trellis-decode prefill |
| `VLLM_EXL3_M_MAX` | `64` (`0` = no CG-PATH) | EXL3 capture cap |
| `VLLM_EXL3_DEQUANT_ALL` | off | dequant leftover / mul1 |
| `VLLM_EXL3_FOLDED_CACHE` | unset | folded-weight cache dir |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | skinny GEMM |
| `VLLM_ROCM_MOE_SKINNY` | `1` | MoE skinny (restored dest-on after a brief disable) |
| `VLLM_FA_RDNA2_GQA_MODE` | `subgroup` | FA GQA-subgroup prefill |
| `VLLM_RDNA_HC_PREFILL_HIP` | `"0"` | Flash-Next HC prefill HIP (opt-in) |
| `VLLM_RDNA_QSA_HIP` | `"0"` | Flash-Next QSA HIP (opt-in) |
| `VLLM_RDNA_PLE_CONV_HIP` | `"0"` | Flash-Next PLE conv HIP (opt-in) |
| `VLLM_RDNA_FUSED_HC` / `VLLM_RDNA_FUSED_SE` | `"1"` | fused decode glue (extras product) |
| `VLLM_ROCM_USE_AITER` | `False` | AITER (CDNA; not dest gfx1030) |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR` | `True` | AITER AR (CDNA) |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | envs.py `False`; dest gfx1030 launcher default `"1"` (`4d25a048`) | force custom AR without full P2P |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | unset | `1stage` / `2stage` |
| `VLLM_ROCM_QUICK_REDUCE_*` | unset | ROCm quick-reduce size/quant knobs |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `1` | symmetric-memory AR |
| `VLLM_RDNA_AR` | `"0"` (dest extras; merged PR **#1**) | leapdragon push AR (opt-in; communicator gate, not the stale “enabled by default” docstring) |
| `VLLM_RDNA_AR_BLOCKS` | auto (dest extras; merged PR **#1**) | AR block cap |
| `VLLM_RDNA_AR_PACE` | `0` (dest extras; merged PR **#1**) | AR store pace |
| `VLLM_RDNA_AR_MAX_KB` | `512` (dest extras; merged PR **#1**) | AR fast-path size cap |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` (auto-on in some configs) | capture dispatcher |
| `VLLM_LOG_GDN_PTRS` | off | GDN pointer probe |
| `VLLM_GDN_DBG` | off | GDN debug print |
| `VLLM_EXL3_DEBUG` / `_HADAMARD_DBG` / `_APPLY_DBG` / `_MARKER_DBG` / `_INPUT_NAN_DBG` | off | EXL3 probes |
| `VLLM_CONV1D_DEBUG` / `VLLM_MLP_DBG` / `VLLM_RDNA2_MOE_DEBUG_NAN` | off | probes |
| `DBG_VLLM_STEP_TIMING` | off | per-step timing |

### AR / collectives

| Path | Where | Default | hippihx |
|---|---|---|---|
| RCCL | extras fallback | on when custom AR off | — |
| Custom all-reduce (vLLM/ROCm) | `VLLM_FORCE_CUSTOM_ALL_REDUCE` | **on** in dest gfx1030 launcher (`4d25a048`); envs.py still False | serve |
| AITER custom AR | `VLLM_ROCM_USE_AITER_CUSTOM_AR` | on in envs, AITER itself off | CDNA, not gfx1030 dest |
| Quick-reduce | `VLLM_ROCM_QUICK_REDUCE_*` | unset | serve |
| Symm-mem AR | `VLLM_ALLREDUCE_USE_SYMM_MEM` | on | serve |
| leapdragon `rdna_ar` Uncached+push | dest extras (merged PR **#1** @ `a4060647`) | **off** | `comm/pcie` Later. Unique HIP: Aron Hsiao. Occupancy pin closed. Boot self-test. INT8/Q8 wire preferred; no Finegrained; no E4M3 without FP8 HW. Pick unique commits, not the dest squash. |

### Not taken / leave in extras

| Item | Why |
|---|---|
| a17t PR **#3** second AWQ GEMM + `qwen4_exp` unique W4 | duplicate W4 family |
| dest `qwen4_exp/**` / PLE offload / TunableOp CSVs | product serve; Flash-Next dest-landed as extras |
| leapdragon `gemv_f16` as a zoo family | no skinny tile; extras consume only |
| Produce / pack (`-cb 3inst`, AWQ produce) | outside hippihx |
| Cudagraph `torch.zeros` / `new_zeros` / `zeros_like`, persist keepalive, `eager_break_during_capture`, GDN arenas | serve page-commit / dispatcher |
| extras PR **#5** Flash-Next draft | **closed**; dest absorbed via PR **#8** — do not merge the old other-fork |
| extras PR **#11** wvSplitK | **closed**; dest-picked as `c350fa218` — extras dense GEMM |
| Explore PRs **#9/#10** W4A8 sdot4 / W4A4 sdot8 | not dest; no sdot tile |
| `rdna_extras_wip_20260910` TRUE FULL snapshot | not dest tip |
| ConfigH (`K_STEP=64`) | dest-reverted; garbage for `M>256` |
| V1 FULL→PIECEWISE + persist keepalive | extras runner; not a zoo file |
| vLLM custom AR cudagraph + launcher default-on | extras `custom_all_reduce.py` / `serve_gfx1030_full.sh`; not Uncached+push |
| extras `bench_report.py` | extras ops; newest by mtime (`7e70e240`); do not copy tok/s |
| extras Qwen4Exp HIP S6 default-on then revert | dest-reverted (`cb0d4418` / `9c9509b3`); gates still off |
| extras Qwen4Exp `_custom_ops` wrappers | extras Python wrappers (`d0d577f1`); still opt-in; do not dump `.cu` |
| extras Qwen4Exp HC HIP compute | extras product HC (`8960a3bc`); isolated compute dest-fixed; capture still not dest-on; do not dump `.cu` |
