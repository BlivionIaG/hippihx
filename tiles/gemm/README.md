# gemm

Dense / quantized GEMM classes for gfx1030. `fdot2` is the math unit; pack
formats are *modes*, not extra directories.

| Directory | Class |
|---|---|
| `w4a16_fdot2` | W4A16 nibble + zero-point via `fdot2`. GPTQ/AWQ are pack/zeros modes of this tile. |
| `exl3_3inst` | HIP consume hook for EXL3 weights produced as `-cb 3inst`. |

**Produce is not a hippihx directory.** AWQ / `3inst` packers stay in their
own tools. This tree only *consumes* a documented pack layout.

**Lock per tile:** LDS, `__launch_bounds__`, occupancy lever (often VGPR, not
`waves_per_eu` on DOT kernels). No WMMA/MFMA/FP8 HW on gfx1030.
