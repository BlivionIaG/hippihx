# gemm/w4a16_fdot2

W4A16 GEMM class: nibble weights, zero-point, `fdot2` accumulate.

GPTQ and AWQ share this tile; they differ in pack/zeros, not in a second
GEMM family.

**Shared DOT source** with gfx1100 (two fatbins). Never `#ifdef WMMA`.
**wave32 only.** **No `fdot2.bf16`.** Activations are **fp16**; bf16 is not
a DOT path.

| Lock | Status |
|---|---|
| LDS bytes | A-tile `[M][BLOCK_K + LDS_PAD]`; **`LDS_PAD=8`** (extras `q_gemm_rdna2.cu`, add `fabf51493` BlivionIaG) |
| High-M AWQ prefill | extras `q_gemm_rdna2_awq_prefill.cu` add `feb7b457e` **BlivionIaG**: `BLOCK_M=16`, `BLOCK_N=64`, `BLOCK_K=32`, `THREADS=128`, `LDS_PAD=8`. Same GEMM family as decode; AWQ is zeros-mode, not a second tile |
| `__launch_bounds__` | extras prefill uses `__launch_bounds__(THREADS)` (128). Not dest-locked here until migrate |
| Wave | **32 only** |
| Zero-point | **Integer nibble `q` minus integer `zero`, then `* scale`.** GPTQ zeros are `uint4b8` (+1); AWQ zeros are literal. Do **not** copy extras `prep_zero_scale_fp16` scale-baked `half` (`0xE400 \| zero`) — that rounded offset is a dest defect (all-zero weights were not exact zero). |
| Prefill K-split | `K_STEP=32`. Every `k_per_split` must be **equal and a multiple of 32**, inside the LDS budget. Dest `compute_split_k` can pick K=640 → 16×40; the kernel reads 32-value tiles — refuse that split. |
| Scratch | sized by `plan`; **zeroed** for page-commit; **no D2H under capture** |

Do not take a17t `awq_gemm_rdna2.cu` / `qdq_awq_rdna2.cuh` as a second W4
family. GPTQ vs AWQ is pack/zeros on this tile. If dest ever locks that
family, cherry-pick **their** commits (`d53572644` and follow-ups), do
not rewrite them.
