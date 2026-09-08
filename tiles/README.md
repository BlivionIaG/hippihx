# Tile contracts

Class directories (not product names). Each child README will lock LDS and
`__launch_bounds__` when the production kernel lands.

| Group | Contracts |
|---|---|
| [`attn/`](attn/README.md) | `fa_fdot2`, `gdn_scan`, `kda_scan`, `qsa_indexer`, `dsa_nope` |
| [`gemm/`](gemm/README.md) | `w4a16_fdot2`, `exl3_3inst` (consume only) |
| [`moe/`](moe/README.md) | `routed`, `shared`, `leftover_bf16` |
| [`comm/`](comm/README.md) | `pcie` (stub) |

`smoke.hip` is a CMake link stub only.
