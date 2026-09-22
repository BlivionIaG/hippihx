# extras inventory (unvalidated)

Moved out of the root README so the zoo stays lean. Tracker, not a
claim that any row works. Bodies stay in extras until extras consumes
one V1 op. Review: [`BACKPORT.md`](BACKPORT.md).

## Unvalidated extras inventory

Snapshot of [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `f3dd65fa7063` (2026-09-20 20:29 UTC). Dest default
branch is **`rdna_extras`**. `main` is upstream vLLM `c00091e02670`.
Merged extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
squash is `a4060647cfbb`. Open **#2–#3**. Draft **#17**. Closed **#12**
(superseded by **#15**). Closed **#16** (unmerged). Merged **#13**,
**#14**, **#15**. **No** `torch.ops.hippihx.*`.

**Unvalidated.** Not dest. Not silicon-signed. No tok/s. hippihx still
ships stubs.

Observed SHAs and actions live in [`BACKPORT.md`](BACKPORT.md). Live dest
bugs (do not copy): W4 scale-baked ZP, unaligned K-split, GDN HIP-on-BF16,
ConfigH, GDN batched-decode n≥8. Dest retraces Flash-Next
FULL_AND_PIECEWISE c=8 to probe artifacts (`609c9c0d`).

Status: **extras** = live on dest tip · **Later** = side branch ·
**skip** = do not take · **stub** = hippihx contract only.

### Kernels (HIP)

| Kernel / op | extras | hippihx tile | Notes |
|---|---|---|---|
| FA paged decode / prefill / split-K / short / GQA | `fa_rdna2.cu` | `attention/fa_fdot2` | `fdot2`. Occupancy pin closed. GQA-subgroup default. O register-resident @ `d1b200b1`. Persist O stay extras. **unvalidated** |
| FA INT8 KV writer | `reshape_and_cache_int8_rdna2` | `attention/fa_fdot2` | INT8 cache layout. **unvalidated** |
| FA fp16 flash KV writer | `reshape_and_cache_flash_rdna2` | `attention/fa_fdot2` | Non-native KV. `__launch_bounds__(128, 4)`. **unvalidated** |
| W4A16 dense decode | `q_gemm_rdna2.cu` | `gemm/w4a16_fdot2` | GPTQ + AWQ = pack/zeros, one GEMM. Dest ZP is scale-baked `half` — zoo uses integer `q-zero` then scale. **unvalidated** |
| W4A16 prefill | `q_gemm_rdna2_prefill.cu` | `gemm/w4a16_fdot2` | Unified GPTQ+AWQ. ConfigA for `M>256`. Dest reverted ConfigH. Dest can pick K=640→16×40; zoo requires equal ×32. **unvalidated** |
| W4A16 AWQ high-M prefill | ~~`q_gemm_rdna2_awq_prefill.cu`~~ | `gemm/w4a16_fdot2` | **Dest-deleted** @ `1046782`. Do not reintroduce. |
| W4A16 MoE | `moe_q_gemm_rdna2.cu` | `moe/routed` | Same W4 family. moe_align prealloc is extras. **unvalidated** |
| EXL3 dense / MoE / dequant / Hadamard / trellis decode | `exl3_dot2_*.cu` | `gemm/exl3_3inst` | Consume `-cb 3inst`. Produce outside. UNC-26. **unvalidated** |
| GDN packed decode | `gdn_decode_rdna2.cu` | `attention/gdn_scan` | Dest @ `02adbfd4`: SSM **fp16 or fp32**. Dest @ `388a61b6`: no one-shot `zero_()` wipe. **fp16 act only.** **unvalidated** |
| GDN prefill chain | `gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | **Opt-in** (`VLLM_GDN_HIP_PREFILL=1`; default Triton/FLA @ `cd1231fd`). `o` varlen `i_t_local` dest-fixed. Dispatch still misses dtype. **unvalidated** |
| causal_conv1d update + fwd | `causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA. FIR on pre-shift then shift. BF16: fp32 mul, **not** `fdot2.bf16`. **unvalidated** |
| Paged MQA indexer | `indexer_paged_mqa_rdna2.cu` | `attention/qsa_indexer` | gfx1030 BF16 6h×256: **4 warps**. **unvalidated** |
| Flash-Next QSA store/compress/MQA | `qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | `VLLM_RDNA_QSA_HIP` default **off**. Eager Triton serves (`8cf0dedb`). Dest @ `e45dd5cb`: empty QSA ring prefix hits. Dest @ `f3dd65fa`: live-context prefill scoring bound (Triton). **unvalidated** |
| Sparse MLA decode / prefill | `sparse_mla_rdna2.cu` | `attention/dsa_nope` | **unvalidated** |
| M-RoPE / Flash-Next HC / PLE / fused glue | `mrope_rdna2.cu`, `hc_rdna2.cu`, `ple_short_conv_rdna2.cu`, `rdna_fused_glue.cu` | — | Product HIP. HC/PLE default off (S6 revert `9c9509b3`); wrappers @ `d0d577f1`; isolated HC @ `8960a3bc`; `_contig()` cache @ `50120e13` (gate still off). Stay extras. **unvalidated** |
| W8A16 / FP8 / MXFP4 / gfx1100 WMMA / skinny GEMM / RMSNorm | `moe_w8a16*.cu`, `mxfp4_dot2_*.cu`, `q_gemm_rdna3_wmma.cu`, `skinny_gemms*.cu`, `layernorm.cu` | — | No tile. WMMA is Later overlay. Do not reintroduce leapdragon `gemv_f16`. **unvalidated** |
| GLM-5.3 KDA / DSA | `glm5_*.cu` (PR **#2**) | `kda_scan` / `dsa_nope` / `qsa_indexer` | Later. Drop `glm5_` name. **unvalidated** |
| leapdragon push AR | `rdna_allreduce.{cu,cuh}` (merged PR **#1**) | `comm/pcie` | Dest extras, default **off**. Occupancy pin closed. **unvalidated** |
| a17t extra AWQ GEMM / GEMV | `awq_gemm_rdna2.cu` etc. (PR **#3**) | — | **skip** — second W4 family |
| Explore W4A8 sdot4 / W4A4 sdot8 | extras PRs **#9/#10** | — | **skip** — not dest |
| Resident W4A16 MoE skinny decode | `moe_resident_decode.cu` (draft PR **#17**) | — | **skip** — not dest. Opt-in `VLLM_RDNA_MOE_RESIDENT*`. Closed **#16** unmerged. Do not dump. |

### Modes / dispatch

| Mode | What extras does | Default (extras) | hippihx |
|---|---|---|---|
| W4 decode vs prefill vs Exllama | M/K/N buckets in `rdna2_w4a16.py` | decode `M≤32` & `K≥4096`; AWQ `M>32` → GPTQ prefill (ConfigA if `M>256`); GPTQ `M>256` still Exllama | one `w4a16_fdot2` tile |
| W4 pack | GPTQ `uint4b8` (+1 zeros) vs AWQ `uint4` (literal zeros) | same kernel, `use_v2_format` | pack/zeros, not a second GEMM |
| EXL3 codebook / memory / prefill | `cb==0` 3inst produce; `full` int16 trellis | 3inst dest; `VLLM_EXL3_PREFILL_DECODE=1` | consume only |
| GDN decode HIP | packed HIP vs Triton/FLA | on unless `VLLM_GDN_DECODE_RDNA2=0`; fires on fp16 SSM @ `02adbfd4` | `gdn_scan` |
| GDN prefill HIP | 5-kernel chain | **opt-in** (`=1`); default Triton/FLA (`cd1231fd`) | `gdn_scan` |
| FA backend / GQA prefill | `RDNA_ATTN` when gfx10x | `VLLM_USE_RDNA2_FA`; GQA **`subgroup`** | `fa_fdot2` |
| FA spec/MTP gate | opt-in abort on verify-shaped batches | **off** | serve |
| MLA sparse HIP | indexer + sparse MLA | `VLLM_USE_RDNA2_MLA=1` | indexer / `dsa_nope` |
| causal conv HIP | update + fwd | on unless set `0` | `causal_conv` |
| Flash-Next HC/QSA/PLE HIP | dest scaffolding | **off**; S6 default-on reverted; wrappers @ `d0d577f1`; isolated HC @ `8960a3bc`; `_contig()` cache @ `50120e13` (same-shape clobber still live) | extras until dest-on |
| Hybrid W4A16 gfx10 | extras linear backend | ungated; RDNA2 W4 auto on gfx1030 | not a second W4 family |
| gfx1030 wvSplitK n≤5 | extras dense GEMM (`c350fa218`) | **dest-reverted** (`5c4ab989`); decode stays `gemv_f16_rdna2` `M<=8` | **no tile** |
| V1 FULL_AND_PIECEWISE | dest maps FULL→PIECEWISE + persist keepalive | extras runner (`1ff73596`); dest @ `f3dd65fa` only redirects FULL when compiled + piecewise; Flash-Next launcher FULL_AND_PIECEWISE (`609c9c0d`) | serve |
| Custom AR (vLLM/ROCm) | force custom all-reduce on PCIe | **on** in dest gfx1030 launcher (`4d25a048`); `envs.py` still False; cudagraph-correct @ `849292ec` | serve |
| leapdragon `rdna_ar` | size-gated Uncached+push | **off** (`VLLM_RDNA_AR=0`) | `comm/pcie` Later |
| Resident W4A16 MoE | native shuffled layout + skinny GEMV | **off** (draft PR **#17**) | extras |

### Env (extras-added / extras-used)

Serve knobs. hippihx does not read these. **Unvalidated.** Debug probes
stay extras (no D2H under capture in the zoo).

| Env | Default (as read in extras) | Role |
|---|---|---|
| `VLLM_USE_RDNA2_FA` | envs.py `False`; `rdna_attn` treats missing as `"1"` | FA-RDNA2 / `RDNA_ATTN` |
| `VLLM_USE_RDNA2_MLA` | off unless `"1"` | sparse MLA + paged MQA HIP |
| `VLLM_FARDNA2_ENABLE_SPEC_GATE` | `"0"` | MTP-verify abort (opt-in) |
| `VLLM_FARDNA2_SPEC_VERIFY_Q_LEN` | `"3"` | spec-gate q len |
| `VLLM_GDN_DECODE_RDNA2` | on (`!= "0"`) | GDN decode HIP |
| `VLLM_GDN_HIP_PREFILL` | off unless `"1"` (opt-in; default Triton/FLA @ `cd1231fd`) | GDN prefill HIP chain |
| `VLLM_GDN_HIP_KERNELS` | recipe `1` | recipe umbrella |
| `VLLM_GDN_DECODE_KERNEL` | `"cuda"` | FLA packed-decode path name |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` | `1` | FLA packed decode |
| `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` / `_FWD` | `"1"` | conv HIP |
| `VLLM_RDNA_FORCE_FP16` | recipe `1` | force fp16 (no BF16 emu) |
| `VLLM_EXL3_MEMORY_MODE` | `full` | `full` / `packed` |
| `VLLM_EXL3_PREFILL_DECODE` | `"1"` | trellis-decode prefill |
| `VLLM_EXL3_M_MAX` | `64` (`0` = no CG-PATH) | EXL3 capture cap |
| `VLLM_EXL3_DEQUANT_ALL` | off | dequant leftover / mul1 |
| `VLLM_EXL3_FOLDED_CACHE` | unset | folded-weight cache dir |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | skinny GEMM |
| `VLLM_ROCM_MOE_SKINNY` | `1` | MoE skinny |
| `VLLM_RDNA_MOE_RESIDENT` / `VLLM_RDNA_MOE_RESIDENT_SKINNY` | `"0"` (draft PR **#17**) | resident W4A16 MoE layout / skinny decode |
| `VLLM_FA_RDNA2_GQA_MODE` | `subgroup` | FA GQA-subgroup prefill |
| `VLLM_RDNA_HC_PREFILL_HIP` / `VLLM_RDNA_QSA_HIP` / `VLLM_RDNA_PLE_CONV_HIP` | `"0"` | Flash-Next product HIP (opt-in) |
| `VLLM_RDNA_FUSED_HC` / `VLLM_RDNA_FUSED_SE` | `"1"` (Flash-Next production launcher sets fused HC `0` @ `3bddd3c9`) | fused decode glue |
| `VLLM_ROCM_USE_AITER` | `False` | AITER (CDNA; not dest gfx1030) |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR` | `True` | AITER AR (CDNA) |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | envs.py `False`; dest gfx1030 launcher default `"1"` (`4d25a048`) | force custom AR without full P2P |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | unset | `1stage` / `2stage` |
| `VLLM_ROCM_QUICK_REDUCE_*` | unset | ROCm quick-reduce |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `1` | symmetric-memory AR |
| `VLLM_RDNA_AR` | `"0"` (dest extras; merged PR **#1**) | leapdragon push AR (opt-in; communicator gate, not the stale “enabled by default” docstring) |
| `VLLM_RDNA_AR_BLOCKS` | auto | AR block cap |
| `VLLM_RDNA_AR_PACE` | `0` | AR store pace |
| `VLLM_RDNA_AR_MAX_KB` | dest extras **64** (`3b59ee16`). zoo lock **512** | AR fast-path size cap |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` (auto-on in some configs) | capture dispatcher |
| `VLLM_LOG_GDN_PTRS` / `VLLM_GDN_DBG` / `VLLM_EXL3_*_DBG` / `VLLM_CONV1D_DEBUG` / `DBG_VLLM_STEP_TIMING` | off | probes |

### AR / collectives

| Path | Where | Default | hippihx |
|---|---|---|---|
| RCCL | extras fallback | on when custom AR off | — |
| Custom all-reduce (vLLM/ROCm) | `VLLM_FORCE_CUSTOM_ALL_REDUCE` | **on** in dest gfx1030 launcher (`4d25a048`); envs.py still False | serve |
| AITER custom AR | `VLLM_ROCM_USE_AITER_CUSTOM_AR` | on in envs, AITER itself off | CDNA, not gfx1030 dest |
| Quick-reduce / symm-mem AR | `VLLM_ROCM_QUICK_REDUCE_*` / `VLLM_ALLREDUCE_USE_SYMM_MEM` | unset / on | serve |
| leapdragon `rdna_ar` Uncached+push | dest extras (PR **#1** @ `a4060647`; T44b @ `3b59ee16`) | **off** | `comm/pcie` Later. Unique HIP: Aron Hsiao. Do not pick Cursor squash. |

### Not taken / leave in extras

Capture plumbing (`torch.zeros` / `new_zeros` / `zeros_like`, persist
keepalive, GDN arenas, `eager_break_during_capture`), product serve
(`qwen4_exp/**`, TunableOp, PLE offload), skinny GEMM, ConfigH, V1
FULL→PIECEWISE, vLLM custom AR, `bench_report.py`, Qwen4Exp HIP gates /
wrappers / HC compute / `_contig()` cache, seq cap 6, Flash-Next
launcher knobs, dest-reverted wvSplitK, QSA Triton bounds, recovered
extras ops, mamba spec-decode `req_idx`, dest T44b `rdna_ar` (still
opt-in; do not pick Cursor squash), dest PR **#14** (HIP MoE ignores
the JSON; do not pick), a17t PR **#3**, closed **#5/#11/#13/#14/#15**,
explore **#9/#10**, closed PR **#12** (superseded by **#15**), closed
PR **#16** (unmerged), draft PR **#17** (resident MoE / TP4 serve —
not dest), ROCm platform init,
extras EXL3 docker arch-guard (one `--offload-arch` per fatbin; gfx1150
/ gfx12xx not dest), Qwen4Exp MTP (not dest), amdsmi `get_device_name`
fallback. Produce (`-cb 3inst`, AWQ pack) stays outside hippihx.
