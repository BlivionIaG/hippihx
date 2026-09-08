# attn/fa_fdot2

Paged / varlen flash-attention class via `fdot2`.

**Shared DOT source:** the same `kernel.hip` is compiled for **gfx1030** and
**gfx1100** as two fatbins (`--offload-arch` each). Never one multi-arch
object. Never `#ifdef WMMA` on this path — WMMA is a gfx1100-only Later
overlay, optional, never required.

| Lock | Status |
|---|---|
| LDS bytes | pin at migrate — see below |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | **32 only** (gfx1030 and gfx1100) |
| ISA | `fdot2` / `v_dot2c` only. **No `fdot2.bf16`.** No WMMA/MFMA/FP8 HW |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |

## FA LDS pins (before migrate from extras)

Docs only — fill measured numbers at migrate. Do not invent tok/s.

| Pin | Notes | Number |
|---|---|---|
| Prefill leftover launch shape | hygiene `(N,1)` leftover / remainder launches | from extras — fill at migrate |
| TopK / LDS budget | ≤48 KiB working set + `attn_stages` guard (64 KiB class) | from extras — fill at migrate |
| `LDS_PAD` | `LDS_PAD=8` on W4 A-tile paths where relevant | from extras — fill at migrate |

Scratch is sized by `plan`. One `torch.ops` / V1 entry when the engine bind
lands. No Triton→HIP double-fire.
