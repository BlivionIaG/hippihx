# moe/shared

Shared (non-routed) expert path. Same consume math as routed GEMM when the
pack matches; different launch and occupancy.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
