# moe/leftover_bf16

Leftover dense BF16: layers produce left unquantized (shared experts, mHC,
heads). Prefer cvt→fp16 `fdot2` if the work is a GEMM; otherwise a no-ISA
scan/norm tile.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
