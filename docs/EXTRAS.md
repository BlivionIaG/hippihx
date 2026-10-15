# extras inventory (unvalidated)

Moved out of the root README so the zoo stays lean. Tracker, not a
claim that any row works. Bodies stay in extras until extras consumes
one V1 op. Review: [`BACKPORT.md`](BACKPORT.md).

## Unvalidated extras inventory

Snapshot of [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `8bb8ff36cdb2` (2026-10-14 09:45 UTC). Dest default
branch is **`rdna_extras`** (v0.31.0 RDNA line). The old 0.28 line is
`rdna_extra/v0.28.0`. `main` is upstream vLLM `c00091e02670`.
Merged extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
squash is `a4060647cfbb`. Open **#2**, **#18**, **#92**. Draft **#10**, **#23**,
**#25**, **#31**, **#33**, **#35**, **#37**, **#39**, **#40**, cleanup
**#41/#42/#44/#48** (leftover 0.28 drafts), **#50** docs, **#51** Glm5Next,
**#53** launcher GQA, **#86** CPU-load, **#87**–**#90** cleanup, **#93** W8A8, **#94** MonoKernel, **#99** QSA/GDN HIP, backlog **#54–#57/#60–#64/#66–#78**. Closed **#3/#9/#16/#30/#32/#34** unmerged
(**#9** superseded by dest W4A8; **#30** dest-landed independently @
`3fb9d43a`; **#32** dest-landed independently @ `a5059cf0`; **#34**
dest-landed independently @ `284fbad0` / dest-presence `ef8f0a28`); **#21**
dest-integrated unmerged; **#12** superseded by **#15**. Merged **#13**,
**#14**, **#15**, **#17**, **#19**, **#20**, **#22**, **#24**, **#26**,
**#27**, **#28**, **#29**, **#36**, **#38**, **#47**, **#49**, **#52**, **#65**, **#79**, **#81**, **#82**, **#83**, **#84**, **#85**, **#91**, **#95**, **#96**, **#97**, **#98**. Dest **#30** @ `3fb9d43a` / dest **#32** @
`a5059cf0` / dest **#34** @ `ef8f0a28` dest-presence / dest QSA HIP
`a6ab43cf` / dest **#36** `121af1b` / dest **#38** `e3bf3d3` / dest HC
stream + MoE tile `c6d99ca` / dest **#47** `7813953` / dest **#49**
`1dc3bac` / dest **#52** `fbe84e94` / dest **#79** `69a166dc` / dest **#65**
`efe9cf5d` / dest **#81** `c057ee54` / dest **#83** `6956c105` / dest **#84**
`64d59b89` / dest **#85** `a77e9e6d` / dest **#91** `2893bdd7` / dest **#82** `e245a469` / dest **#95** `95046194` / dest **#96** `6b77576c` / dest **#97** `15e9e7ad` / dest **#98** `8bb8ff36`. **No**
`torch.ops.hippihx.*`. Dest **#27** is opt-in `VLLM_HIPPIHX` ctypes
consume (default **off**; plans not ready).

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
| FA paged decode / prefill / split-K / short / GQA | `fa_rdna2.cu` | `attention/fa_fdot2` | `fdot2`. Occupancy pin closed. GQA-subgroup default. O register-resident @ `d1b200b1`. Dest **#28** @ `bfd5286d` tile skip / dest **#34** @ `ef8f0a28` D=128 register-O GQA prefill stay extras. Dest tip `8bb8ff36` is dest **#98** W4A16 MoE asymmetric zp. QSA HIP dest-presence stays `a6ab43cf` (default **off**). Do not dump. Persist O stay extras. **unvalidated** |
| FA INT8 KV writer | `reshape_and_cache_int8_rdna2` | `attention/fa_fdot2` | INT8 cache layout. **unvalidated** |
| FA fp16 flash KV writer | `reshape_and_cache_flash_rdna2` | `attention/fa_fdot2` | Non-native KV. `__launch_bounds__(128, 4)`. **unvalidated** |
| W4A16 dense decode | `q_gemm_rdna2.cu` | `gemm/w4a16_fdot2` | GPTQ + AWQ = pack/zeros, one GEMM. Dest ZP is scale-baked `half` — zoo uses integer `q-zero` then scale. **unvalidated** |
| W4A16 prefill | `q_gemm_rdna2_prefill.cu` | `gemm/w4a16_fdot2` | Unified GPTQ+AWQ. ConfigA for `M>256`. Dest reverted ConfigH. Dest @ `3a0786ea` K_STEP-aligns unusable POT splits. Zoo still refuse 40-wide. **unvalidated** |
| W4A8 sdot4 dense prefill | `w4a8_sdot4_rdna2.{cu,cuh}` (dest @ `3a0786ea`) | — | Stay extras. Opt-in `VLLM_RDNA2_W4A8_SDOT4`. gfx1030 `M≥33`. Internal W4A16 fallback. Not a second W4 zoo family. Do not dump. **unvalidated** |
| MoE W4A8 sdot4 | `moe_w4a8_rdna2.cu` (dest @ `3a0786ea`) | — | Stay extras. Opt-in. Hard-off under resident MoE. Dest @ `9cad8792` packed-fp16 CAS default. Do not dump. **unvalidated** |
| W4A16 AWQ high-M prefill | ~~`q_gemm_rdna2_awq_prefill.cu`~~ | `gemm/w4a16_fdot2` | **Dest-deleted** @ `1046782`. Do not reintroduce. |
| W4A16 MoE | `moe_q_gemm_rdna2.cu` + `moe_accum_rdna2.cuh` | `moe/routed` | Same W4 family. Dest @ `e1315629` dequant/eight-row stay extras. Dest @ `9cad8792` packed-fp16 CAS default; `VLLM_RDNA2_MOE_FP32_ACCUM` opt-in. Dest **#98** @ `8bb8ff36` runtime `use_v2_format` zp (4-bit asymmetric). moe_align prealloc is extras. **unvalidated** |
| EXL3 dense / MoE / dequant / Hadamard / trellis decode | `exl3_dot2_*.cu` | `gemm/exl3_3inst` | Consume `-cb 3inst`. Produce outside. Dest **#32** @ `a5059cf0` mul1 / K=1..8 + MoE n-tile bound. Dest @ `7be99d84` fused `exl3_project_rdna2`. UNC-26. **unvalidated** |
| GDN packed decode | `gdn_decode_rdna2.cu` | `attention/gdn_scan` | Dest @ `02adbfd4`: SSM **fp16 or fp32**. Dest @ `388a61b6`: no one-shot `zero_()` wipe. **fp16 act only.** **unvalidated** |
| GDN prefill chain | `gdn_prefill_*_rdna2.cu` | `attention/gdn_scan` | **Opt-in** (`VLLM_GDN_HIP_PREFILL=1`; default Triton/FLA @ `cd1231fd`). `o` varlen `i_t_local` dest-fixed. Dispatch still misses dtype. **unvalidated** |
| causal_conv1d update + fwd | `causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA. FIR on pre-shift then shift. Dest @ `b13effcb` gates HIP when `state_len != width-1`. Dest @ `ab5ccf3d` HIP re-enable no-gain vs Triton. BF16: fp32 mul, **not** `fdot2.bf16`. **unvalidated** |
| Paged MQA indexer | `csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu` (dest **#47**) | `attention/qsa_indexer` | Dest **#47** moved DSV4 indexer. gfx1030 BF16 6h×256: **4 warps**. Stay extras. Do not dump. **unvalidated** |
| Flash-Next QSA store/compress/MQA | `qsa_rdna2.cu` | `attention/qsa_indexer` (watch) | `VLLM_RDNA_QSA_HIP` default **off**. Eager Triton serves (`8cf0dedb`). Dest @ `e45dd5cb`: empty QSA ring prefix hits. Dest @ `f3dd65fa`: live-context prefill scoring bound (Triton). Dest @ `a6ab43cf` v0.30 port + store-stride fix dest-landed independently; still default **off**. Do not dump. **unvalidated** |
| Sparse MLA decode / prefill | `csrc/rocm/rdna/dsv4/sparse_mla_decode.cu` (dest **#47**) | `attention/dsa_nope` | Dest **#47** moved DSV4 sparse MLA. Stay extras. Do not dump. **unvalidated** |
| M-RoPE / Flash-Next HC / PLE / fused glue | `mrope_rdna2.cu`, `hc_rdna2.cu`, `hc_prefill_rdna2.cu`, `ple_short_conv_rdna2.cu`, `rdna_fused_glue.cu` | — | Product HIP. HC/PLE default off (S6 revert `9c9509b3`); wrappers @ `d0d577f1`; isolated HC @ `8960a3bc`; `_contig()` cache @ `50120e13`; dest @ `c6d99ca` current-stream launches + `data_ptr` key (same-shape clobber dest-fixed; `VLLM_RDNA_HC_PREFILL_HIP` still off). Dest **#47** fused HC prefill dest-on gfx1030 (`VLLM_RDNA_HC_PREFILL_FUSED`). Stay extras. Do not dump. **unvalidated** |
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
| W4 `torch.compile` M-dispatch | dest PR **#30** @ `3fb9d43a` | — | Stay extras. Opt-in `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH`. Default **off**. Python only. GitHub PR closed-unmerged. Do not dump. Do not copy tok/s. |
| W4 exact-dequant explore | draft PR **#31** | — | **skip** — explore, not dest. Claude. Opt-in `VLLM_RDNA2_W4A16_EXACT_DEQUANT`. Zoo already locks integer `q-zero` then scale. Do not dump |
| EXL3 mul1 decode / K=1..8 trellis | dest PR **#32** @ `a5059cf0` | `gemm/exl3_3inst` | Stay extras. Unsigned mul1 + K=1..8 + dest-fixed MoE staging bound. GitHub PR closed-unmerged. Do not dump. |
| EXL3 v0.30 loader / fused project / MoE / PLE | dest @ `30b0bd4e` (`exl3.py`, `exl3_moe.py`, `exl3_ple.py`, `exl3_project_rdna2`) | `gemm/exl3_3inst` | Stay extras. Codebook markers, suh grouping, TP slicing, `fold_weight` lm_head only, `Exl3MoEMethod`, inert `Exl3NgramTable`. Unique Blivion. Do not dump. |
| TunableOp unify / EXL3 27B rows / gemma_rms fake | dest @ `ac9dfd8f` (`tunableop/rocblas-f30bb442e9b5`, `gemma_rms_norm_fake`) | — | Stay extras. Serve/ops + inductor fake. Unique Blivion. Do not dump. Do not copy tok/s. |
| EXL3 MTP draft + `serve_rdna` recipes | dest @ `bc5fbee5` (`exl3.py` `mtp.fc`, `serve_rdna.sh`, HIP conv `state_len` gate) | — | Stay extras. Dest serves EXL3 quantized MTP. No zoo MTP tile. Unique Blivion. Do not dump. Do not copy tok/s. |
| FA decode scores / 4 barriers | draft PR **#33** | — | **skip** — not dest. Dest @ `ab5ccf3d` still not merged. Do not dump `fa_rdna2` |
| FA D=128 prefill register-O GQA | dest PR **#34** @ `ef8f0a28` / bench `bb40498c` | `attention/fa_fdot2` | Stay extras. `HEAD_DIM` 128/256; ~14 KiB LDS dest observation. Dest tip `8bb8ff36` is dest **#98** W4A16 MoE asymmetric zp. QSA HIP dest-presence stays `a6ab43cf`. GitHub closed-unmerged. Do not dump `fa_rdna2`. Do not copy tok/s. |
| not-FA/not-HIP sweep + probes | dest @ `ab5ccf3d` (`tools/rdna2_028`) | — | Stay extras. Docs + A/B probes. Unique Blivion. extras **#33** still skip. Do not dump. Do not copy tok/s. |
| QSA HIP v0.30 port + store-stride | dest @ `a6ab43cf` (`qsa_rdna2.cu`) | `attention/qsa_indexer` (watch) | Stay extras. Unique Blivion dest landing. Default **off**. Do not dump. Do not copy tok/s. |
| TunableOp MTP=2 verify GEMM rows | draft PR **#35** | — | **skip** — not dest. Serve/ops. Do not copy tok/s. |
| Adaptive prefill chunk cap + cadence | dest PR **#36** @ `121af1b` (`dynamic_prefill.py`) | — | Stay extras. Python scheduler. Foreign Waldecir / wsantos. `VLLM_RDNA_DYNAMIC_PREFILL` default **off**. Do not dump. Do not pick. Do not copy tok/s. |
| Routed-expert DRAM residency cache | draft PR **#37** | — | **skip** — not dest. Serve/ops. `VLLM_RDNA_MOE_EXPERT_OFFLOAD` default **off**. Do not dump. |
| Restore mixed decode+prefill batching | dest PR **#38** @ `e3bf3d3` | — | Stay extras. Python scheduler. Unique Blivion. Removes `VLLM_ROCM_NO_MIXED_BATCH`. Do not dump. Do not copy tok/s. |
| System One / GLiNER2 decision route | draft PR **#39** | — | **skip** — not dest. Serve/API. No HIP. Do not dump. |
| HC HIP stream + `_contig` `data_ptr` + MoE prefill tile | dest @ `c6d99ca` (`hc_rdna2`, `rdna2_w4a16_moe`) | — | Stay extras. Unique Blivion. Current-stream launches. Same-shape clobber dest-fixed. Prefill `block_size_m=8` for large token counts. `VLLM_RDNA_HC_PREFILL_HIP` still default **off**. Do not dump. Do not copy tok/s. |
| v0.31.0 RDNA port | dest PR **#47** @ `7813953` | — | Stay extras. Unique Blivion dest landing. Serve/port onto upstream vLLM v0.31.0. Old 0.28 line is `rdna_extra/v0.28.0`. Launchers `tools/rdna/`. DSV4 HIP under `csrc/rocm/rdna/dsv4/`. Fused HC prefill dest-on gfx1030. Still no `torch.ops.hippihx`. Do not dump. Do not copy tok/s. |
| INT4 PLE table on upstream PLE path | dest PR **#49** @ `1dc3bac` | — | Stay extras. Unique Blivion dest landing. Opt-in `VLLM_PLE_QUANT_DIR`. Python/Triton UVA. Unset = BF16 path unchanged. Do not dump. Do not copy tok/s. |
| INT4 PLE table from disk / CPU sidecar | dest PR **#52** @ `fbe84e94` | — | Stay extras. Unique Blivion dest landing. Opt-in `PLE=int4-disk` / `VLLM_PLE_CPU_OFFLOAD=1`. Unset = pinned/BF16 path unchanged. Do not dump. Do not copy tok/s. |
| Tuning guide | dest PR **#79** @ `69a166dc` | — | Stay extras. Docs only. Unique Blivion. Do not copy tok/s. |
| W4A8 MoE prefill opt-in | dest PR **#65** @ `efe9cf5d` | — | Stay extras. Unique Blivion dest landing. Opt-in `VLLM_RDNA2_W4A8_SDOT4=1`. Prefill-only ≥512 tokens. Do not dump `moe_w4a8`. Do not copy tok/s. |
| TunableOp 8k prefill chunks | dest PR **#81** @ `c057ee54` | — | Stay extras. Serve/ops. Unique Blivion. Do not copy tok/s. |
| P2P=0 launcher + LD_LIBRARY_PATH | dest PR **#83** @ `6956c105` | — | Stay extras. Serve/tools. Unique Blivion dest landing. Alan/Claude LD_LIBRARY_PATH from **#80** — do not pick. Do not dump. |
| Decline asymmetric MoE checkpoints | dest PR **#84** @ `64d59b89` | — | Stay extras. Python only. Unique Blivion dest landing. Native RDNA2 MoE declined asymmetric weights (Triton WNA16 fallback). Dest **#98** later accepts 4-bit asymmetric. Do not dump. Do not copy tok/s. |
| Bounded-memory PLE short-conv prefill | dest PR **#85** @ `a77e9e6d` | — | Stay extras. Unique Blivion dest landing. Python. Opt-in HIP path untouched. Do not dump. Do not copy tok/s. |
| Qwen4Exp PLE INC (AutoRound) | dest PR **#91** @ `2893bdd7` | — | Stay extras. Python only. Cherry-pick upstream vllm#59990 (Vadim Gimpelson + Claude). Do not pick. Do not dump. Do not copy tok/s. |
| Flash-Next PP + MTP drafter quantizer | dest PR **#82** @ `e245a469` | — | Stay extras. Python/serve/tools. Unique Blivion dest landing of PR **#80** PP=3 + PR **#58** PLE sidecar under PP + PR **#59** MTP drafter W4A16 quantizer. Foreign Alan/alanoo81 — do not pick. Do not dump. Do not copy tok/s. |
| Int8 dense shadows in row chunks | dest PR **#95** @ `95046194` | — | Stay extras. Python/serve. Unique Blivion dest landing. `VLLM_RDNA_DENSE_INT8` opt-in. Approach from alanoo81 — do not pick. Do not dump. Do not copy tok/s. |
| PP3 MTP2 Flash-Next on three cards | dest PR **#96** @ `6b77576c` | — | Stay extras. Python/serve/tools. Unique Blivion dest landing. Drafter CPU placeholders + embed share under PP. Approach from alanoo81 — do not pick. Do not dump. Do not copy tok/s. |
| Prefix caching off launcher | dest PR **#97** @ `15e9e7ad` | — | Stay extras. Serve/tools. Unique Blivion dest landing. `ENABLE_PREFIX_CACHING=0` passes `--no-enable-prefix-caching`. Do not dump. Do not copy tok/s. |
| Native W4A16 MoE asymmetric zp | dest PR **#98** @ `8bb8ff36` | `moe/routed` | Stay extras. ATen HIP. Unique Blivion dest landing. Runtime `use_v2_format` zp offset. Native RDNA2 MoE accepts 4-bit asymmetric compressed-tensors; 8-bit still falls back. Do not dump. Do not copy tok/s. |
| Advertise `serve_rdna` + wiki | draft PR **#50** | — | **skip** — not dest. Docs only. |
| Glm5Next through upstream model | draft PR **#51** | — | **skip** — not dest. Later. |
| Launcher GQA decode default off | draft PR **#53** | — | **skip** — not dest. Serve scripts. |
| Host CPU-load investigation | draft PR **#86** | — | **skip** — not dest. Docs only. Do not dump. Do not copy tok/s. |
| Delete tools/rdna2_028 campaign drivers | draft PR **#87** | — | **skip** — not dest. Cleanup. Do not dump. |
| Delete unimported W8A16-FP8 + EXL3 n-gram | draft PR **#88** | — | **skip** — not dest. Cleanup. Do not dump. |
| W4A8 sdot4 launched-config only | draft PR **#89** | — | **skip** — not dest. Cleanup. Do not dump. |
| W4A8 MoE gate above cudagraph capture size | draft PR **#90** | — | **skip** — not dest. Do not dump. Do not copy tok/s. |
| TunableOp qualify rocm10-rocblas5.6 | open PR **#92** | — | **skip** — not dest. Serve/ops. Foreign wsantos/Waldecir. Do not dump. Do not pick. Do not copy tok/s. |
| W8A8 int8 sdot4 dense / MoE | draft PR **#93** | — | **skip** — not dest. New HIP family. Do not dump `w8a8_sdot4`. Do not copy tok/s. |
| MonoKernel plan gfx1030 Flash-Next FFN-only | draft PR **#94** | — | **skip** — not dest. Docs + standalone probe. Do not dump `tools/monokernel-p0-probe`. Do not copy tok/s. |
| QSA/GDN HIP prefill + TunableOp grid | draft PR **#99** | — | **skip** — not dest. Foreign wsantos/Waldecir. Implements backlog **#64/#69**. Do not dump. Do not pick. Do not copy tok/s. |
| Backlog ports / HIP / sched / AR | draft PRs **#54–#57/#60–#64/#66–#78** | — | **skip** — not dest. **#58/#59** dest-landed via **#82**. Do not dump. |
| Hetero MoE gfx1100/gfx1030 | draft PR **#40** | — | **skip** — not dest. Leftover 0.28 draft. Do not dump. |
| Cleanup W4A8 sweep / tools / probes / prefill hooks | draft PR **#41** / PR **#42** / PR **#44** / PR **#48** | — | **skip** — not dest. Leftover 0.28 drafts. Do not dump `w4a8_sdot4` / tools / `q_gemm_rdna2_prefill`. |
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
| Flash-Next HC/QSA/PLE HIP | dest scaffolding | **off** (`VLLM_RDNA_{HC_PREFILL,QSA,PLE_CONV}_HIP`); dest **#47** fused HC prefill dest-on gfx1030 (`VLLM_RDNA_HC_PREFILL_FUSED`) | extras until dest-on; fused prefill stay extras |
| Adaptive prefill | dest **#36** @ `121af1b` live chunk cap + step cadence | **off** (`VLLM_RDNA_DYNAMIC_PREFILL=0`) | extras; serve/scheduler |
| Mixed decode+prefill batch | dest **#38** @ `e3bf3d3` | **on** (fork-only `VLLM_ROCM_NO_MIXED_BATCH` removed) | extras; serve/scheduler |
| MoE prefill `block_size_m` | dest @ `c6d99ca` large-token tile 8 | decode stays 4; prefill 8 when tokens ≥ `VLLM_RDNA2_MOE_BSM_MIN_TOKENS` | extras |
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
| MoE epilogue accum | dest @ `9cad8792` packed-fp16 CAS | **off** (`VLLM_RDNA2_MOE_FP32_ACCUM=0`); `=1` is fp32 scratch | extras |

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
| `VLLM_RDNA2_MOE_FP32_ACCUM` | `False` (dest @ `9cad8792`; only `"1"` enables) | opt-in MoE fp32 scratch epilogue; dest default is packed-fp16 CAS |
| `VLLM_RDNA_DYNAMIC_PREFILL` | `"0"` (dest **#36** @ `121af1b`; only `"1"` enables) | opt-in adaptive prefill chunk cap + step cadence |
| `VLLM_RDNA2_MOE_BSM_LARGE` / `VLLM_RDNA2_MOE_BSM_MIN_TOKENS` | dest @ `c6d99ca` (prefill tile 8 / token floor) | MoE prefill `block_size_m` for large token counts |
| `VLLM_HIPPIHX_LIB` / `VLLM_HIPPIHX_CODE_OBJECT` | unset | `libhippihx_v1.so` / `hippihx_<arch>.hsaco` |
| `VLLM_RDNA_MOE_RESIDENT` / `VLLM_RDNA_MOE_RESIDENT_SKINNY` | `"0"` (dest **#17** @ `e1315629`) | resident W4A16 MoE layout / skinny decode |
| `VLLM_FA_RDNA2_GQA_MODE` | `subgroup` | FA GQA-subgroup prefill |
| `VLLM_FA_RDNA2_GQA_DECODE` | `"0"` (dest **#28** @ `bfd5286d`; dest launchers set `1` when FA) | opt-in GQA decode kernel |
| `VLLM_FA_RDNA2_SPLIT_DECODE` | `"1"` (dest **#29** @ `d94e2209`) | decode-first mixed/verify split |
| `VLLM_FA_RDNA2_VERIFY_FULL_GRAPH` | `"1"` (dest **#29** dest-on; PR was opt-in) | `UNIFORM_BATCH` for uniform verify |
| `VLLM_RDNA_HC_PREFILL_HIP` / `VLLM_RDNA_QSA_HIP` / `VLLM_RDNA_PLE_CONV_HIP` | `"0"` | Flash-Next product HIP (opt-in) |
| `VLLM_RDNA_HC_PREFILL_FUSED` | dest **#47** dest-on gfx1030 | fused HC prefill (`hc_prefill_rdna2`); stay extras |
| `VLLM_PLE_QUANT_DIR` | unset (dest **#49** @ `1dc3bac`; INT4 table when set) | opt-in INT4 group-16 PLE table on upstream PLE path |
| `VLLM_PLE_CPU_OFFLOAD` / `PLE=int4-disk` | unset (dest **#52** @ `fbe84e94`; disk sidecar when set) | opt-in INT4 PLE from disk via CPU sidecar |
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
**#30** compile-dispatch `3fb9d43a` / dest MoE epilogue `9cad8792` /
dest **#32** EXL3 mul1 `a5059cf0` / dest EXL3 loader `30b0bd4e` / dest
TunableOp `ac9dfd8f` / dest EXL3 MTP + `serve_rdna` `bc5fbee5` / dest
**#34** FA D=128 GQA `ef8f0a28` / bench follow-on `bb40498c` / dest sweep `ab5ccf3d` / dest QSA HIP `a6ab43cf` / dest **#36** adaptive prefill `121af1b` / dest **#38** mixed batch `e3bf3d3` / dest HC stream + MoE tile `c6d99ca` / dest **#47** v0.31 port `7813953` / dest **#49** INT4 PLE `1dc3bac` / dest **#52** INT4 PLE disk `fbe84e94` / dest **#79** tuning docs `69a166dc` / dest **#65** W4A8 MoE prefill `efe9cf5d` / dest **#81** TunableOp 8k `c057ee54` / dest **#83** P2P=0 `6956c105` / dest **#84** decline asymmetric MoE `64d59b89` / dest **#85** PLE short-conv mem `a77e9e6d` / dest **#91** INC AutoRound PLE `2893bdd7` / dest **#82** PP + MTP bundle `e245a469` / dest **#95** int8 shadow chunks `95046194` / dest **#96** PP3 MTP2 `6b77576c` / dest **#97** prefix cache off `15e9e7ad` / dest **#98** asymmetric W4A16 MoE zp `8bb8ff36` (observe, do not pick). Skip: a17t
PR **#3**, PR **#18** GPTQ `BLOCK_KN_SIZE` 256, draft PR **#23** PCIe
P2P KV, draft PR **#25** rdna_ar retained-output, draft PR **#31** W4
exact-dequant explore, draft PR **#33** FA decode scores, draft PR **#35** TunableOp MTP=2 rows, draft PR **#37** MoE expert DRAM residency, draft PR **#39** System One / GLiNER2, draft PR **#40** hetero MoE, cleanup **#41/#42/#44/#48**, draft PR **#50** docs advertise `serve_rdna`, draft PR **#51** Glm5Next, draft PR **#53** launcher GQA decode, draft PR **#86** CPU-load investigation, draft PR **#87** delete `tools/rdna2_028`, draft PR **#88** delete unimported W8A16-FP8 + EXL3 n-gram, draft PR **#89** W4A8 sdot4 launched-config only, draft PR **#90** W4A8 MoE capture-size gate, open PR **#92** TunableOp profile, draft PR **#93** W8A8 sdot4, draft PR **#94** MonoKernel plan, draft PR **#99** QSA/GDN HIP prefill, backlog PR **#54**–PR **#57**/PR **#60**–PR **#64**/PR **#66**–PR **#78**, closed **#9**
(superseded), closed **#30** / **#32** / **#34** (dest-landed
independently), explore **#10**, closed PR **#12**, closed PR **#16**,
closed **#5/#11**. Leftover 0.28 drafts target `rdna_extra/v0.28.0`.
Produce stays outside hippihx.
