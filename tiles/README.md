# Tile contracts

Class directories (not product names). Each child README will lock LDS and
`__launch_bounds__` when the production kernel lands.

| Group | Contracts |
|---|---|
| [`attention/`](attention/README.md) | `fa_fdot2` (DOT), `gdn_scan`, `kda_scan`, `qsa_indexer`, `dsa_nope` |
| [`gemm/`](gemm/README.md) | `w4a16_fdot2` (DOT), `exl3_3inst` (DOT, consume only) |
| [`moe/`](moe/README.md) | `routed`, `shared` (DOT), `leftover_bf16` |
| [`sequence/`](sequence/README.md) | `causal_conv` (scalar FMA, not a scan) |
| [`comm/`](comm/README.md) | `pcie` (stub) |

**DOT tiles** (`fa_fdot2`, `w4a16_fdot2`, `exl3_3inst`, `moe/shared`) share
one source compiled per `--offload-arch` (gfx1030, gfx110x, gfx1151,
gfx103x, gfx1013). gfx1151 / Deck / BC-250 are portable — can run, not
dest-tuned. Steam Deck gfx1033 is **wave32** RDNA2. gfx1013 is **RDNA2**
Skillfish (same generation, different GFX); do not force
`-mwavefrontsize32`. gfx900 does not load them.

`smoke.hip` is a CMake link stub only.

FlyDSL kernel contracts live in `hippihx/flydsl/`, not under `tiles/`.
