# attention/kda_scan

KDA linear-state scan class. State shape is a *class* parameter (e.g. 64h×128),
not a product-named 128×128 fuse.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Not | GDN scan; not a Kimi fused KDA decode kernel |

GEMM leftovers around this tile stay in `gemm/` or `moe/leftover_bf16`.

extras PR #2 (`later/glm53-flash-awq-glm5next`) has product-named
`glm5_kda_*.cu`. When those land as dest, extract into **this** class
directory — drop the `glm5_` prefix. Do not retarget GDN layouts onto KDA.
