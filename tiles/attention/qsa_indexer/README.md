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
opt-in `qsa_rdna2.cu` (Flash-Next store/compress/MQA; `VLLM_RDNA_QSA_HIP`
default **off**). PR #2 `glm5_dsa_indexer_rdna2.cu` is Later. Confirm
class before a body migrate.

Dest @ `820465` recorded a Flash-Next **Triton** `forward_qsa` GPU trap
(`SIGABRT`). `VLLM_RDNA_QSA_WARPS=2` did **not** fix it — do not treat
the 2-warp prefill spill as that abort. Dest since then: eager QSA
break (`8cf0dedb`) serves; dest V1 maps FULL→PIECEWISE (`1ff73596`).
HIP `qsa_rdna2.cu` is still scaffolding, not dest-on. Dest @ `e45dd5cb`
fixed extras prefix hits when the QSA compression ring is empty at a
group boundary — stay extras. extras draft PR **#15** bounds Triton
prefill scoring to live context — Python only, **not dest**, no HIP
body. That PR also folds extras **#12** CPU PLE / MTP / graph-redirect
(serve, not this tile). Do not dump Triton QSA. Do not copy tok/s. The
4-warp occupancy row is a gfx1030 observation, not a tok/s claim.
