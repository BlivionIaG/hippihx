# moe/shared

Shared (non-routed) expert path. Same consume math as routed GEMM when the
pack matches; different launch and occupancy.

**Shared DOT source** with gfx1030 + gfx1100 (two fatbins). Never
`#ifdef WMMA`. **wave32 only.** **No `fdot2.bf16`.**

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | **32 only** |
| Graph | scratch **zeroed**; **no D2H under capture** |
