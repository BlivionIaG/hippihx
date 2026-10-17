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
prealloc (`4b799bf4`) stays extras. Dest extras **#17** @ `e1315629`
resident skinny (`moe_resident_decode.cu`) is dest-landed ATen HIP;
`VLLM_RDNA_MOE_RESIDENT*` default **off**. Dest @ `9cad8792` packed-fp16
CAS epilogue is dest default; `VLLM_RDNA2_MOE_FP32_ACCUM` is opt-in.
Dest @ `c6d99ca` selects prefill `block_size_m=8` for large token
counts (`VLLM_RDNA2_MOE_BSM_LARGE` / `_MIN_TOKENS`). Dest **#65** @
`efe9cf5d` opt-in W4A8 MoE prefill (`VLLM_RDNA2_W4A8_SDOT4=1`, ≥512
tokens). Dest **#84** @ `64d59b89` declined asymmetric MoE
(Python). Dest **#98** @ `8bb8ff36` runtime `use_v2_format` zp
accepts 4-bit asymmetric compressed-tensors; 8-bit still falls back.
Dest tip `36f456d2` is dest **#117** EXL3 Flash-Next serve (fused
EXL3 MoE). Dest **#106** test cleanup stays dest-landed. Dest
**#94** MonoKernel decode stays dest-landed opt-in
(`--monokernel`; k2 routed W4A16). Stay extras. Do
not dump `moe_accum_rdna2.cuh`. Closed **#16** unmerged.
