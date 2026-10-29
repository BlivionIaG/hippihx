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
Dest **#117** @ `36f456d2` EXL3 Flash-Next serve (fused EXL3 MoE). Dest
tip `b7402893` is dest **#147** batched expert copies / cache planning. Dest **#146** TP2 AWQ MoE expert offload stays dest-landed. Dest **#144** @ `a3140284` human quick start stays dest-landed. Dest rdna_hint_flex @ `58babef9` stays dest-landed. Dest **#141** @ `8015ea48` stale docs / MD060 stays dest-landed. Dest **#140** @ `0b4533ee` fp16 SSM / derived prefill budget / W4A8 prefill stays dest-landed. Dest **#139** @ `005db032` V620 CU-count comments stays dest-landed. Dest **#138** @ `2d461709` INT4 KV native QSA read stays dest-landed (never NVFP4).
Dest **#134** @ `8cc4e2d5` INT4+FP16 KV block-scale stays dest-landed.
Dest **#137** @ `12a46f40` MTP2 MonoKernel stays dest-landed (opt-in
`--monokernel`). Dest **#132** @ `30d6e82d` DFlash2 + FA BLHNC stays
dest-landed. Dest
**#131** @ `3804d495` no shared-experts aux-stream fork in CUDA graphs
on gfx10x. Dest **#106** test cleanup stays dest-landed. Dest **#94**
MonoKernel decode stays dest-landed opt-in (`--monokernel`; k2 routed
W4A16). Stay extras. Do not dump `moe_accum_rdna2.cuh`. Closed **#16**
unmerged.
