# gemm/exl3_3inst

HIP consume hook for EXL3 weights whose **produce** used `-cb 3inst`.
Decode is `3inst` → `half2` → `fdot2`.

**Shared DOT source** with gfx1100 (two fatbins). Never `#ifdef WMMA`.
**wave32 only.** **No `fdot2.bf16`.**

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — DOT kernels often leave this unset; document the choice |
| Wave | **32 only** |
| Graph | scratch **zeroed**; **no D2H under capture** |
| Produce | **outside** hippihx — do not add a `produce/` or `3inst/` packer dir here |

`mcg` / `mul1` codebooks are compile slots, not extra zoo roots.
