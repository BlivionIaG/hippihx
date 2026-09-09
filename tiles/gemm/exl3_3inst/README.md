# gemm/exl3_3inst

HIP consume hook for EXL3 weights whose **produce** used `-cb 3inst`.
Decode is `3inst` → `half2` → `fdot2`.

**Shared DOT source** with gfx1100 (two fatbins). Never `#ifdef WMMA`.
**wave32 only.** **No `fdot2.bf16`.**

| Lock | Status |
|---|---|
| LDS bytes | extras `exl3_dot2_dense.cu` (add `40850e6c5` **BlivionIaG**): A-staging `[M_PER][BLOCK_K + LDS_PAD]`, `BLOCK_K=256`, **`LDS_PAD=8`**. Small vs FA; occupancy is VGPR |
| `__launch_bounds__` | **unset on DOT kernels** (UNC-26). Hadamard wrappers use `__launch_bounds__(32)` — not the K-dot |
| Wave | **32 only** |
| Geometry (prefill / large-M) | extras @ `a4060647`: `THREADS_X=256`, `BLOCK_N=1024`, `BLOCK_K=256`, 4 N-cols/thread, 16×16 window → `decode_3inst` → `half2` → `fdot2`. `M_PER` dispatch: 1 / 2 / 4 / 8 |
| Geometry (decode grain v2) | extras @ `a4060647`: bits=3 and `sm≤8` → grain v2. `V2_THREADS_X=64`, `V2_BLOCK_N=1024`, `V2_BLOCK_K=128`, `V2_COL=16`, `V2_KTILE=16`, **`LDS_PAD=8`**. `sm==1`→`M_PER=1`; `sm==2`→`M_PER=2`; else `M_PER=4` (z-split; 8 lost to VGPR). Prefill `sm>8` keeps the original kernel |
| Graph | scratch **zeroed**; **no D2H under capture** |
| Produce | **outside** hippihx — do not add a `produce/` or `3inst/` packer dir here |
| Consume ABI | `HIPPIHX_V1_OP_GEMM_EXL3_3INST` in `include/hippihx/v1.h` — plan/run stub until body migrate |

`mcg` / `mul1` codebooks are compile slots, not extra zoo roots. UNC-26 is
still In Progress on extras — do not copy bodies until produce/consume
and capture buffers stop moving. Port and later perf commits stay
**BlivionIaG** `<kev29lt@gmail.com>` as Author **and** Committer. Do not
squash them into a Cursor rewrite.
