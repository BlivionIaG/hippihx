# gemm/w4a16_fdot2

W4A16 GEMM class: nibble weights, zero-point, `fdot2` accumulate.

GPTQ and AWQ share this tile; they differ in pack/zeros, not in a second
GEMM family.

**Shared DOT source** with gfx1100 (two fatbins). Never `#ifdef WMMA`.
**wave32 only.** **No `fdot2.bf16`.**

| Lock | Status |
|---|---|
| LDS bytes | A-tile `[M][BLOCK_K + LDS_PAD]`; **`LDS_PAD=8`** (extras `q_gemm_rdna2*.cu`, `d71721c79547`) |
| High-M AWQ prefill | extras `q_gemm_rdna2_awq_prefill.cu`: `BLOCK_M=16`, `BLOCK_N=64`, `BLOCK_K=32`, `THREADS=128`, `LDS_PAD=8`. Same GEMM family as decode; AWQ is zeros-mode, not a second tile |
| `__launch_bounds__` | extras prefill uses `__launch_bounds__(THREADS)` (128). Not dest-locked here until migrate |
| Wave | **32 only** |
| Scratch | sized by `plan`; **zeroed** for page-commit; **no D2H under capture** |

Do not take a17t `awq_gemm_rdna2.cu` / `qdq_awq_rdna2.cuh` as a second W4
family. GPTQ vs AWQ is pack/zeros on this tile.
