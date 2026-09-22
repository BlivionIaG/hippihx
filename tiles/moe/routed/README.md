# moe/routed

Routed expert gate / up / down. W4A16 / EXL3 consume reuse `gemm/` tiles;
this directory owns the MoE launch, dispatch, and scratch around them.

| Lock | Status |
|---|---|
| LDS bytes | extras `moe_q_gemm_rdna2.cu`: A-tile `[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD]`, **`LDS_PAD=8`** (same W4 pad as `gemm/w4a16_fdot2`) |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |

Do not take a17t `moe_awq_gemm_rdna2.cu` as a second routed family.
Dest extras @ `b549c229` wires the existing `moe_q_gemm_rdna2` kernel
into the oracle as `RDNA2_W4A16` — extras serve, same W4 family. moe_align
prealloc (`4b799bf4`) stays extras. Closed extras **#16** / draft **#17**
resident skinny (`moe_resident_decode.cu`) is not dest — stay extras.
Do not dump.
