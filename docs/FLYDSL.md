# FlyDSL on gfx1030 — research, not dest

Sourced from [rdna-hip-wiki `engine/flydsl.md`](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/engine/flydsl.md)
(`ROCm/FlyDSL`). hippihx dest backend is **HIP fatbins**.

## Verdict

FlyDSL may emit ordinary ROCDL/LLVM code for `gfx1030`, but **none of its
current optimized GEMM, MoE, or FlashAttention kernels support RDNA2**.
Those depend on CDNA MFMA or gfx11/gfx12 WMMA. The first useful experiment
is a tiny DOT/GEMV proof, not an MFMA port.

`hippihx._lib.backend.FLYDSL_DEST` is **false**.
`Caps(arch="gfx1030", backend="flydsl")` raises.

## Gates (in order)

0. gfx1030 code object + vector-add correctness (`FLYDSL_GPU_ARCH=gfx1030`).
1. Explicit `fdot2` / `sdot4` wrappers lower to `v_dot2*` / `v_dot4*` ISA.
2. Small-M GEMV correctness and graph-safe warm/cache behavior.
3. Occupancy + performance vs Triton, stock HIP skinny, rocBLAS.
4. Only then a narrow extras consume path. MoE/attention are later rewrites.

## Do not

- Port `preshuffle_gemm`, `moe_gemm_2stage`, `rdna3_f16_gemm`,
  `rdna_f16_gemm`, `flash_attn_generic` (wave64 MFMA).
- Call FlyDSL `WMMA` helpers on gfx1030 (they raise).
- Add `flydsl` as a package dependency.
- Treat a PyPI wheel as proof of gfx1030 kernel support.

Prefer LLVM intrinsics (`llvm.amdgcn.fdot2`, `llvm.amdgcn.sdot4`) over
inline asm. ISA evidence is mandatory.

When a gate passes, flip the matching flag in `hippihx/_lib/backend.py`
in the **same** commit as the proof, not before.
