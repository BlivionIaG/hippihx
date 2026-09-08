# attn/qsa_indexer

QSA / group-select indexer class. Selects groups; exact attention over the
selected set is a different launch (`fa_fdot2` or a later QSA-attn tile).

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Not | V4 C4A compressed Lightning indexer |

One V1 entry. No Triton→HIP double-fire from this library.

extras relatives: `indexer_paged_mqa_rdna2.cu` (dest) and PR #2
`glm5_dsa_indexer_rdna2.cu` (Later). Confirm class before a body migrate.
