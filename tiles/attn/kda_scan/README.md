# attn/kda_scan

KDA linear-state scan class. State shape is a *class* parameter (e.g. 64h×128),
not a product-named 128×128 fuse.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Not | GDN scan; not a Kimi fused KDA decode kernel |

GEMM leftovers around this tile stay in `gemm/` or `moe/leftover_bf16`.
