# moe/routed

Routed expert gate / up / down. W4A16 / EXL3 consume reuse `gemm/` tiles;
this directory owns the MoE launch, dispatch, and scratch around them.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
