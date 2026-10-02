# extras inventory (unvalidated)

Moved out of the root README so the zoo stays lean. Tracker, not a
claim that any row works. Bodies stay in extras until extras consumes
one V1 op. Review: [`BACKPORT.md`](BACKPORT.md).

## Unvalidated extras inventory

Snapshot of [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `3fb9d43afdb5` (2026-09-30 18:56 UTC). Dest default
branch is **`rdna_extras`**. `main` is upstream vLLM `c00091e02670`.
Merged extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
squash is `a4060647cfbb`. Open **#2**, **#18**. Draft **#10**, **#23**,
**#25**, **#31**, **#32**, **#33**, **#34**. Draft **#30** dest-landed
independently (GitHub still open). Closed **#3/#9/#16** unmerged
(**#9** superseded by dest W4A8); **#21** dest-integrated unmerged;
**#12** superseded by **#15**. Merged **#13**, **#14**, **#15**, **#17**,
**#19**, **#20**, **#22**, **#24**, **#26**, **#27**, **#28**, **#29**.
Dest **#30** @ `3fb9d43a` dest-presence. **No** `torch.ops.hippihx.*`.
Dest **#27** is opt-in `VLLM_HIPPIHX` ctypes consume (default **off**;
plans not ready).

**Unvalidated.** Not dest. Not silicon-signed. No tok/s. hippihx still
ships stubs.

Observed SHAs and actions live in [`BACKPORT.md`](BACKPORT.md). Live dest
bugs (do not copy): W4 scale-baked ZP, GDN HIP-on-BF16,
ConfigH, GDN batched-decode n≥8. Dest @ `3a0786ea` K_STEP-aligns
unusable POT K-splits. Dest retraces Flash-Next FULL_AND_PIECEWISE c=8
to probe artifacts (`609c9c0d`).

Status: **extras** = live on dest tip · **Later** = side branch ·
**skip** = do not take · **stub** = hippihx contract only.

### Kernels (HIP)

| Kernel / op | extras | hippihx tile | Notes |
|---|---|---|---|
| FA paged decode / prefill / split-K / short / GQA | `fa_rdna2.cu` | `attention/fa_fdot2` | `fdot2`. Occupancy pin closed. GQA-subgroup default. O register-resident @ `d1b200b1`. Dest **#28** @ `bfd5286d` tile skip / in-place out / GQA softmax stay extras. Do not dump. Persist O stay extras. **unvalidated** |
| FA INT8 KV writer | `reshape_and_cache_int8_rdna2` | `attention/fa_fdot2` | INT8 cache layout. **unvalidated** |
| FA fp16 flash KV writer | `reshape_and_cache_flash_rdna2` | `attention/fa_fdot2` | Non-native KV. `__launch_bounds__(128, 4)`. **unvalidated** |
| W4A16 dense decode | `q_gemm_rdna2.cu` | `gemm/w4a16_fdot2` | GPTQ + AWQ = pack/zeros, one GEMM. Dest ZP is scale-baked `half` — zoo uses integer `q-zero` then scale. **unvalidated** |
| W4A16 prefill | `q_gemm_rdna2_prefill.cu` | `gemm/w4a16_fdot2` | Unified GPTQ+AWQ. ConfigA for `M>256`. Dest reverted ConfigH. Dest @ `3a0786ea` K_STEP-aligns unusable POT splits. Zoo still refuse 40-wide. **unvalidated** |
| W4A8 sdot4 dense prefill | `w4a8_sdot4_rdna2.{cu,cuh}` (dest @ `3a0786ea`) | — | Stay extras. Opt-in `VLLM_RDNA2_W4A8_SDOT4`. gfx1030 `M≥33`. Internal W4A16 fallback. Not a second W4 zoo family. Do not dump. **unvalidated** |
| MoE W4A8 sdot4 | `moe_w4a8_rdna2.cu` (dest @ `3a0786ea`) | — | Stay extras. Opt-in. Hard-off under resident MoE. Do not dump. **unvalidated** |
| W4A16 AWQ high-M prefill | ~~`q_gemm_rdna2_awq_prefill.cu`~~ | `gemm/w4a16_fdot2` | **Dest-deleted** @ `1046782`. Do not reintroduce. |
| W4A16 MoE | `moe_q_gemm_rdna2.cu` | `moe/routed` | Same W4 family. Dest @ `e1315629` dequant/eight-row stay extras. moe_align prealloc is extras. **unvalidated** |
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
| leapdragon push AR | `rdna_allreduce.{cu,cuh}` (merged PR **#1**; dest **#22** @ `22d6e346`) | `comm/pcie` | Dest extras, default **off**. Dest two-shot then dest-off @ `101a16c8` (library `MAX_KB` **64**). Zoo **512**. Do not dump. **unvalidated** |
| a17t extra AWQ GEMM / GEMV | `awq_gemm_rdna2.cu` etc. (closed PR **#3**) | — | **skip** — closed unmerged, second W4 family |
| GPTQ exllama `BLOCK_KN_SIZE` 256 | `q_gemm.cu` (PR **#18**) | — | **skip** — not dest DOT W4 |
| PCIe P2P KV disagg | `PcieP2pConnector` (draft PR **#23**) | — | **skip** — not dest. HIP IPC SDMA. Zoo `comm.pcie` is AR, not KV. Do not dump |
| RAM KV offload | offloading connector (dest **#24** @ `cd38a1d3`) | — | Stay extras. George/Codex. Serve only. No HIP. Do not pick |
| rdna_ar retained-output | draft PR **#25** | — | **skip** — not dest. George/Codex. Do not dump. Do not pick |
| HIP MoE skinny `MAX_M=16` | dest PR **#26** @ `7434efee` | — | Stay extras. Opt-in `VLLM_ROCM_MOE_SKINNY_MAX_M`. Default still **8**. Do not dump. Do not pick |
| hippihx V1 consume | `hippihx_v1.py` (dest **#27** @ `a3f7e5da`) | — | Stay extras. Opt-in `VLLM_HIPPIHX`. ctypes + `hippihx_v1_load`. Plans not ready. Still no `torch.ops.hippihx`. Do not dump. Do not pick Claude |
| FA-RDNA2 tile skip / in-place / GQA softmax | dest PR **#28** @ `bfd5286d` | `attention/fa_fdot2` | Stay extras. Foreign Claude. Do not dump. Do not pick. **unvalidated** |
| FA spec-decode split | dest PR **#29** @ `d94e2209` / tip `e0112c55` | `attention/fa_fdot2` | Stay extras. Dest rework of foreign Claude (`cu_query_lens` decode + row gate ≤256). Supersedes dest @ `83e6af80` per-position verify-decode. Do not dump. Do not pick Claude. **unvalidated** |
| W4 `torch.compile` M-dispatch | dest PR **#30** @ `3fb9d43a` | — | Stay extras. Opt-in `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH`. Default **off**. Python only. GitHub PR still open draft. Do not dump. Do not copy tok/s. |
| W4 exact-dequant explore | draft PR **#31** | — | **skip** — explore, not dest. Claude. Opt-in `VLLM_RDNA2_W4A16_EXACT_DEQUANT`. Zoo already locks integer `q-zero` then scale. Do not dump |
| EXL3 mul1 decode / K=1..8 trellis | draft PR **#32** | — | **skip** — not dest. Claude. Do not dump |
| FA decode scores / 4 barriers | draft PR **#33** | — | **skip** — not dest. Do not dump `fa_rdna2` |
| FA D=128 prefill register-O GQA | draft PR **#34** | — | **skip** — not dest. Do not dump `fa_rdna2` |
| Explore W4A8 sdot4 / W4A4 sdot8 | extras PRs **#9/#10** | — | **#9** closed-unmerged (superseded by dest W4A8 @ `3a0786ea`). **#10** **skip** — not dest |
| Resident W4A16 MoE skinny decode | `moe_resident_decode.cu` (dest **#17** @ `e1315629`) | `moe/routed` (watch) | Stay extras. Opt-in `VLLM_RDNA_MOE_RESIDENT*`. ATen HIP. Do not dump. Closed **#16** unmerged. **unvalidated** |

### Modes / dispatch

| Mode | What extras does | Default (extras) | hippihx |
|---|---|---|---|
| W4 decode vs prefill vs Exllama | M/K/N buckets in `rdna2_w4a16.py` | decode `M≤32` & `K≥4096`; AWQ `M>32` → GPTQ prefill (ConfigA if `M>256`); GPTQ `M>256` still Exllama | one `w4a16_fdot2` tile |
| W4 pack | GPTQ `uint4b8` (+1 zeros) vs AWQ `uint4` (literal zeros) | same kernel, `use_v2_format` | pack/zeros, not a second GEMM |
| EXL3 codebook / memory / prefill | `cb==0` 3inst produce; `full` int16 trellis | 3inst dest; `VLLM_EXL3_PREFILL_DECODE=1` | consume only |
| GDN decode HIP | packed HIP vs Triton/FLA | on unless `VLLM_GDN_DECODE_RDNA2=0`; fires on fp16 SSM @ `02adbfd4` | `gdn_scan` |
| GDN prefill HIP | 5-kernel chain | **opt-in** (`=1`); default Triton/FLA (`cd1231fd`) | `gdn_scan` |
| FA backend / GQA prefill | `RDNA_ATTN` when gfx10x | `VLLM_USE_RDNA2_FA`; dest @ `48c56ef` pins `AttentionConfig` when env is `1` and backend unset; GQA **`subgroup`** | `fa_fdot2` |
| FA spec/MTP gate | opt-in abort on verify-shaped batches | **off** | serve |
| MLA sparse HIP | indexer + sparse MLA | `VLLM_USE_RDNA2_MLA=1` | indexer / `dsa_nope` |
| causal conv HIP | update + fwd | on unless set `0` | `causal_conv` |
| Flash-Next HC/QSA/PLE HIP | dest scaffolding | **off**; S6 default-on reverted; wrappers @ `d0d577f1`; isolated HC @ `8960a3bc`; `_contig()` cache @ `50120e13` (same-shape clobber still live) | extras until dest-on |
| Hybrid W4A16 gfx10 | extras linear backend | ungated; RDNA2 W4 auto on gfx1030 | not a second W4 family |
| W4A8 sdot4 prefill | dest @ `3a0786ea` int4×int8 `sdot4` | **off** (`VLLM_RDNA2_W4A8_SDOT4` unset = W4A16) | extras; not a second W4 zoo family |
| gfx1030 wvSplitK n≤5 | extras dense GEMM (`c350fa218`) | **dest-reverted** (`5c4ab989`); decode stays `gemv_f16_rdna2` `M<=8` | **no tile** |
| V1 FULL_AND_PIECEWISE | dest captures FULL + piecewise | extras runner (`1ff73596`); dest @ `68a635ed` keeps FULL decode graphs; dest @ `e0112c55` `UNIFORM_BATCH` when split-decode + verify FULL graph | serve |
| FA split decode | dest **#29** decode-first mixed/verify | **on** (`VLLM_FA_RDNA2_SPLIT_DECODE=1`); row gate ≤256 query×head @ `e0112c55` | extras |
| Custom AR (vLLM/ROCm) | force custom all-reduce on PCIe | **on** in dest gfx1030 launcher (`4d25a048`); `envs.py` still False; cudagraph-correct @ `849292ec` | serve |
| leapdragon `rdna_ar` | size-gated Uncached+push; dest **#22** two-shot dest-off @ `101a16c8` | **off** (`VLLM_RDNA_AR=0`) | `comm/pcie` Later |
| Resident W4A16 MoE | native shuffled layout + skinny GEMV | **off** (dest **#17** @ `e1315629`) | extras |
| hippihx V1 consume | dest **#27** ctypes + `hippihx_v1_load` | **off** (`VLLM_HIPPIHX=0`) | extras; plans not ready |
| W4 compile-dispatch | dest **#30** `torch.ops.vllm.rdna2_w4a16_gemm` | **off** (`VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=0`) | extras; dest keeps default off |

### Env (extras-added / extras-used)

Serve knobs. hippihx does not read these. **Unvalidated.** Debug probes
stay extras (no D2H under capture in the zoo).

| Env | Default (as read in extras) | Role |
|---|---|---|
| `VLLM_USE_RDNA2_FA` | envs.py `False`; dest @ `48c56ef` pins `RDNA_ATTN` in `check_and_update_config` when `"1"` | FA-RDNA2 / `RDNA_ATTN` |
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
| `VLLM_ROCM_MOE_SKINNY_MAX_M` | `8` (dest **#26** @ `7434efee`; opt-in `16`) | sequential HIP MoE row cap |
| `VLLM_HIPPIHX` | `False` (dest **#27** @ `a3f7e5da`; only `"1"` enables) | opt-in hippihx V1 consume |
| `VLLM_RDNA2_W4A8_SDOT4` | unset / `"0"` (dest @ `3a0786ea`; only `"1"` enables) | opt-in W4A8 sdot4 prefill |
| `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH` | `"0"` (dest **#30** @ `3fb9d43a`; only `"1"` enables) | opt-in W4 compile-time M-dispatch via custom op |
| `VLLM_HIPPIHX_LIB` / `VLLM_HIPPIHX_CODE_OBJECT` | unset | `libhippihx_v1.so` / `hippihx_<arch>.hsaco` |
| `VLLM_RDNA_MOE_RESIDENT` / `VLLM_RDNA_MOE_RESIDENT_SKINNY` | `"0"` (dest **#17** @ `e1315629`) | resident W4A16 MoE layout / skinny decode |
| `VLLM_FA_RDNA2_GQA_MODE` | `subgroup` | FA GQA-subgroup prefill |
| `VLLM_FA_RDNA2_GQA_DECODE` | `"0"` (dest **#28** @ `bfd5286d`; dest launchers set `1` when FA) | opt-in GQA decode kernel |
| `VLLM_FA_RDNA2_SPLIT_DECODE` | `"1"` (dest **#29** @ `d94e2209`) | decode-first mixed/verify split |
| `VLLM_FA_RDNA2_VERIFY_FULL_GRAPH` | `"1"` (dest **#29** dest-on; PR was opt-in) | `UNIFORM_BATCH` for uniform verify |
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
| `VLLM_RDNA_AR_MAX_KB` | dest extras **64** (`101a16c8`). zoo lock **512** | AR fast-path size cap |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` (auto-on in some configs) | capture dispatcher |
| `VLLM_LOG_GDN_PTRS` / `VLLM_GDN_DBG` / `VLLM_EXL3_*_DBG` / `VLLM_CONV1D_DEBUG` / `DBG_VLLM_STEP_TIMING` | off | probes |

### AR / collectives

| Path | Where | Default | hippihx |
|---|---|---|---|
| RCCL | extras fallback | on when custom AR off | — |
| Custom all-reduce (vLLM/ROCm) | `VLLM_FORCE_CUSTOM_ALL_REDUCE` | **on** in dest gfx1030 launcher (`4d25a048`); envs.py still False | serve |
| AITER custom AR | `VLLM_ROCM_USE_AITER_CUSTOM_AR` | on in envs, AITER itself off | CDNA, not gfx1030 dest |
| Quick-reduce / symm-mem AR | `VLLM_ROCM_QUICK_REDUCE_*` / `VLLM_ALLREDUCE_USE_SYMM_MEM` | unset / on | serve |
| leapdragon `rdna_ar` Uncached+push | dest extras (PR **#1** @ `a4060647`; T44b @ `3b59ee16`; dest **#22** then dest-off @ `101a16c8`) | **off** | `comm/pcie` Later. Unique HIP: Aron Hsiao. Dest `MAX_KB` **64**. Zoo **512**. Do not pick Cursor squash. |

### Not taken / leave in extras

Stay extras: serve, product HIP, dest-landed extras PRs, dest **#29**
split decode / dest row-gate `e0112c55` / dest W4A8 `3a0786ea` / dest
**#30** compile-dispatch `3fb9d43a` (observe, do not pick). Skip: a17t
PR **#3**, PR **#18** GPTQ `BLOCK_KN_SIZE` 256, draft PR **#23** PCIe
P2P KV, draft PR **#25** rdna_ar retained-output, draft PR **#31** W4
exact-dequant explore, draft PR **#32** EXL3 mul1, draft PR **#33** FA
decode scores, draft PR **#34** FA D=128 prefill, closed **#9**
(superseded), explore **#10**, closed PR **#12**, closed PR **#16**,
closed **#5/#11**. Produce stays outside hippihx.
