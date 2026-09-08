# attn/fa_fdot2

Paged / varlen flash-attention class for gfx1030 via `fdot2`.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| ISA extras | `fdot2` / `v_dot2c` only; no WMMA/MFMA/FP8 HW |

Scratch is sized by `plan`, zeroed for cudagraph page-commit. No D2H under
capture. One `torch.ops` / V1 entry when the engine bind lands.
