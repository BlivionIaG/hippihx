# attn/dsa_nope

DSA MLA-NoPE class (LoRA-compressed Q/KV, NoPE heads). Same *class* as a
sparse MLA, not the same launch as a V4 FP8 Lightning tile.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Not | `fp8_ds_mla` / 448+64 RoPE Lightning |

Indexer work for DSA lives in `qsa_indexer` or a later sibling — do not
collapse classes to save a directory.

extras `sparse_mla_rdna2.cu` is the dest sparse-MLA relative. PR #2
`glm5_dsa_mla_decode_rdna2.cu` is Later — same rename rule as `kda_scan`.
