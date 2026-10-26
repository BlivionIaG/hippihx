# attention/qsa_indexer

QSA / group-select indexer class. Selects groups; exact attention over the
selected set is a different launch (`fa_fdot2` or a later QSA-attn tile).

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Occupancy (gfx1030 BF16, 6h×256 prefill) | **4 warps** (128 threads). A 2-warp profile spills on V620. Keep the tile/split/arithmetic; do not drop to 64 threads for that shape. |
| Not | V4 C4A compressed Lightning indexer |

One V1 entry. No Triton→HIP double-fire from this library.

extras relatives: dest `indexer_paged_mqa_rdna2.cu` (DeepSeek class) and
`qsa_rdna2.cu` (Flash-Next store/compress/MQA; dest **#130** dest-on
`VLLM_RDNA_QSA_HIP` in Flash-Next recipes). PR #2
`glm5_dsa_indexer_rdna2.cu` is Later. Confirm class before a body migrate.

Dest @ `820465` recorded a Flash-Next **Triton** `forward_qsa` GPU trap
(`SIGABRT`). `VLLM_RDNA_QSA_WARPS=2` did **not** fix it — do not treat
the 2-warp prefill spill as that abort. Dest since then: eager QSA
break (`8cf0dedb`) serves; dest V1 maps FULL→PIECEWISE (`1ff73596`).
HIP `qsa_rdna2.cu` is dest-present @ `a6ab43cf` (v0.30 port +
store-stride fix). Dest **#126** @ `6854b2bc` MonoKernel QSA indexer HIP
(fused store / one-launch epilogue / HIP scoring). Dest **#130** dest-on
QSA HIP in Flash-Next recipes. Dest tip `58babef9` is dest
rdna_hint_flex stacked layout (direct push; no extras PR). Dest **#141**
@ `8015ea48` stale docs / MD060 stays dest-landed. Dest **#140** @ `0b4533ee` fp16 SSM /
derived prefill budget / W4A8 prefill stays dest-landed. Dest **#139**
@ `005db032` V620 CU-count comments stays dest-landed. Dest **#138**
@ `2d461709` INT4 KV native QSA read stays dest-landed (opt-in `int4_fp16_block`; never NVFP4). Dest
**#134** @ `8cc4e2d5` INT4+FP16 KV block-scale stays dest-landed. Dest
**#137** @ `12a46f40` MTP2
MonoKernel stays dest-landed (opt-in `--monokernel`). Dest **#132** @
`30d6e82d` DFlash2 + FA BLHNC stays dest-landed. Dest **#117** EXL3 Flash-Next serve stays dest-landed
(QSA indexer quantized under EXL3). Dest **#106** test cleanup stays dest-landed. Dest **#94** MonoKernel decode stays dest-landed opt-in (`--monokernel`; includes HIP QSA decode part). Dest **#99** dest-landed via **#130** (foreign Waldecir — do not pick). Dest @ `e45dd5cb`
fixed extras prefix hits when the QSA compression ring is empty at a
group boundary — stay extras. Dest extras **#15** @ `f3dd65fa` bounds
Triton prefill scoring to live context — Python only. Do not dump
`qsa_rdna2`. Do not copy tok/s. Do not pick the merge. The 4-warp
occupancy row is a gfx1030 observation, not a tok/s claim.
