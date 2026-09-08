# Tile contracts

Class directories (not product names). Each child README will lock LDS and
`__launch_bounds__` when the production kernel lands.

| Group | Contracts |
|---|---|
| [`attn/`](attn/README.md) | `fa_fdot2` (DOT), `gdn_scan`, `kda_scan`, `qsa_indexer`, `dsa_nope` |
| [`gemm/`](gemm/README.md) | `w4a16_fdot2` (DOT), `exl3_3inst` (DOT, consume only) |
| [`moe/`](moe/README.md) | `routed`, `shared` (DOT), `leftover_bf16` |
| [`sequence/`](sequence/README.md) | `causal_conv` (scalar FMA, not a scan) |
| [`comm/`](comm/README.md) | `pcie` (stub) |

**DOT tiles** (`fa_fdot2`, `w4a16_fdot2`, `exl3_3inst`, `moe/shared`) share
one source compiled per `--offload-arch` (gfx1030, gfx110x, gfx1151,
gfx103x). gfx1151 / Deck are portable — can run, not dest-tuned.
gfx1013 (BC-250) is Later VERIFY (RADV reports warp 64; do not share
wave32 DOT until measured). gfx900 does not load them.

`smoke.hip` is a CMake link stub only.
