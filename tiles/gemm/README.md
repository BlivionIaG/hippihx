# gemm

Dense / quantized GEMM classes. `fdot2` is the math unit on the **shared
DOT source** (gfx1030 + gfx1100). Pack formats are *modes*, not extra
directories.

| Directory | Class |
|---|---|
| `w4a16_fdot2` | W4A16 nibble + zero-point via `fdot2`. GPTQ/AWQ are pack/zeros modes of this tile. |
| `exl3_3inst` | HIP consume hook for EXL3 weights produced as `-cb 3inst`. |

Both are **DOT tiles**: same `.hip` compiled for gfx1030 and gfx1100 as
two fatbins. **wave32 only. No `fdot2.bf16`. Never `#ifdef WMMA`.** gfx1100
WMMA is a Later overlay, not these files. gfx900 does not load these
objects (`mad_mix` / `pk_fma` slot). W4 consume: integer ZP then scale;
prefill K-splits equal multiples of 32. Do not copy dest extras'
scale-baked ZP or unaligned `compute_split_k`.

**Produce is not a hippihx directory.** AWQ / `3inst` packers stay in their
own tools.

**Lock per tile:** LDS, `__launch_bounds__`, occupancy lever (often VGPR).
`LDS_PAD=8` on W4 A-tile paths where relevant (from extras — fill at
migrate).
