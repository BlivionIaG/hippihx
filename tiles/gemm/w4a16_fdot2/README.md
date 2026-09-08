# gemm/w4a16_fdot2

W4A16 GEMM class: nibble weights, zero-point, `fdot2` accumulate.

GPTQ and AWQ share this tile; they differ in pack/zeros, not in a second
GEMM family.

**Shared DOT source** with gfx1100 (two fatbins). Never `#ifdef WMMA`.
**wave32 only.** **No `fdot2.bf16`.**

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel; `LDS_PAD=8` on W4 A-tile paths where relevant (from extras — fill at migrate) |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | **32 only** |
| Scratch | sized by `plan`; **zeroed** for page-commit; **no D2H under capture** |
