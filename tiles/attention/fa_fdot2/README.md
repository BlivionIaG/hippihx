# attention/fa_fdot2

Paged / varlen flash-attention class via `fdot2`.

**Shared DOT source:** the same `kernel.hip` is compiled for **gfx1030** and
**gfx1100** as two fatbins (`--offload-arch` each). Never one multi-arch
object. Never `#ifdef WMMA` on this path — WMMA is a gfx1100-only Later
overlay, optional, never required.

| Lock | Status |
|---|---|
| LDS bytes | extras decode ~33 KiB / prefill ~48 KiB — see below. Occupancy pin closed |
| `__launch_bounds__` | extras 128 / 256 variants — **not dest-locked** |
| Wave | **32 only** (gfx1030 and gfx1100) |
| ISA | `fdot2` / `v_dot2c` only. **No `fdot2.bf16`.** No WMMA/MFMA/FP8 HW. Activations **fp16**; V1 refuses bf16 |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |

## FA LDS pins (before migrate from extras)

Observed on extras `d71721c79547` (`csrc/rocm/fa_rdna2.cu`). File added
`b1b3fa938` **BlivionIaG** `<kev29lt@gmail.com>`. Not dest-locked (FA
occupancy pin stays closed). Do not invent tok/s. Do not copy the `.cu`
body yet — when you do, cherry-pick BlivionIaG, do not re-author
([`docs/BACKPORT.md`](../../docs/BACKPORT.md)).

| Pin | Notes | Number |
|---|---|---|
| Decode smem | `Br=1`, `Bc=64`, `head_dim=128`, `THREADS=128` | ~33 KiB (extras comment) |
| Prefill smem | `Br=16`, `Bc=64`, `head_dim=128`, `THREADS=128` | ~48 KiB; 64 KiB/CU class, 1 block/CU |
| Prefill leftover launch shape | hygiene `(N,1)` leftover / remainder launches | from extras — fill at migrate |
| TopK / LDS budget | ≤48 KiB working set + `attn_stages` guard | extras prefill ~48 KiB sits on this pin |
| `LDS_PAD` | W4 A-tile pad lives on `gemm/w4a16_fdot2`, not FA | `8` (GEMM tile) |
| `__launch_bounds__` | extras @ `a4060647`: decode D=128 `__launch_bounds__(128)`; decode D=256 + GQA-aware D=256 `__launch_bounds__(256)`; prefill `THREADS_PREFILL=128` | **not dest-locked** — occupancy pin closed |
| GQA D=256 decode | extras @ `6c5ff94`: idle waves still shuffle; skip online-softmax when `m_new == -inf` (NaN guard); zero `O_partial` on empty seq. `GQA_MAX_G=8`, `GQA_BC=32`, `GQA_DSK=256+8` | observed; not dest-locked |
| fp16 flash KV writer | extras: `reshape_and_cache_flash_rdna2`, `__launch_bounds__(128, 4)`, used for non-native KV write, fp16 only | extras HIP; do not dump ATen wrapper. Persist workspaces stay extras. |
| GQA-subgroup prefill | extras @ `ecfec4e412ad`: `fa_rdna2_prefill_paged_varlen_gqa`, `HEADS_PER_CTA=2` / `BR=8`, dual-acc fdot2. Env `VLLM_FA_RDNA2_GQA_MODE` default `subgroup`. True-GQA (`HEADS_PER_CTA=6`) dropped. Dest @ `d1b200b1`: GQA prefill O accumulator is register-resident (was smem). Occupancy pin closed. Do not dump `.cu`. | observed; occupancy pin closed. |

Dest extras @ `48c56ef` pins `AttentionConfig.backend = RDNA_ATTN` from
the API-server `VLLM_USE_RDNA2_FA` env so workers actually select
FA-RDNA2. Dest extras **#28** @ `bfd5286d` and dest **#29** @ `d94e2209` /
`e0112c55` (`cu_query_lens` split decode + row gate) stay extras.
Do not dump. Do not pick Claude. No tok/s.

## V1 rev 4 contract (pinned)

Observed extras `csrc/rocm/fa_rdna2.cu` @ `cd38a1d` host signatures
(`fa_rdna2_decode_paged`, `fa_rdna2_prefill_paged_varlen[_splitk]`).
Contract only, no body. Rows live in `hippihx/_lib/catalog.py`. Header
enums are `HIPPIHX_V1_ATTN_FA_FDOT2_P_*` and `_T_*`.

| | Contract |
|---|---|
| Params | `mode` (decode = split-K + combine, prefill = paged varlen), `head_dim` 128 / 256, `num_q_heads` a multiple of `num_kv_heads`, `block_size`, `kv_splits` 1–16 (extras `MAX_SPLITS`), `sliding_window` (0 = off), `causal`, `max_tokens` (the capture bucket), `scale` |
| Tensor slots | `q` fp16 `[≤max_tokens, H_q, D]` contiguous · `k_cache` fp16 `[blocks, H_kv, D/x, block_size, x]` strided · `v_cache` fp16 5-D strided · `block_table` i32 rows (decode reads one row per query token) · `seq_lens` i32 · `cu_query_lens` i32 (prefill only) · `out` fp16 `[≤max_tokens, H_q, D]` contiguous |
| Scratch | fp32 `o_partial [T, H_q, S, D]` and `m_partial` / `l_partial [T, H_q, S]` when `mode == decode` or `kv_splits > 1`. Zeroed once by serve |
| Variant | explore rank: decode 0, prefill 1 |
| Index width | extras uses 32-bit `int` partial indices in the decode per-head kernels, the combine, and the prefill split-K empty-split path. Plan refuses `T·H_q·S·D > 2^31 − 1` (`HIPPIHX_V1_ERR_PARAM`) |

extras allocates `O` and the partials inside the op (`rdna2_persist_zeros`).
The zoo contract moves both out: `out` is a caller-owned slot and the
partials are plan scratch. That keeps `bind` allocation-free.

Scratch is sized by `plan`. C consume id: `HIPPIHX_V1_OP_ATTN_FA_FDOT2`
(`include/hippihx/v1.h`). Serve wraps as one `torch.ops.hippihx.*` — no
Triton→HIP double-fire.
