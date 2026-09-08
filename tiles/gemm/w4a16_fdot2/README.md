# gemm/w4a16_fdot2

W4A16 GEMM class: nibble weights, zero-point, `fdot2` accumulate.

GPTQ and AWQ share this tile; they differ in pack/zeros, not in a second
GEMM family.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Scratch | sized by `plan`; prefer caller `empty`/`zeros`, not hipMallocAsync pools under capture |
