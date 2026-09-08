# attn/fa_fdot2

Paged / varlen flash-attention class via `fdot2`.

**Shared DOT source:** the same `kernel.hip` is compiled for **gfx1030** and
**gfx1100** as two fatbins (`--offload-arch` each). Never one multi-arch
object. Never `#ifdef WMMA` on this path — WMMA is a gfx1100-only Later
overlay, optional, never required.

| Lock | Status |
|---|---|
| LDS bytes | extras decode ~33 KiB / prefill ~48 KiB — see below. Occupancy pin closed |
| `__launch_bounds__` | extras 128 / 256 variants — **not dest-locked** |
| Wave | **32 only** (gfx1030 and gfx1100) |
| ISA | `fdot2` / `v_dot2c` only. **No `fdot2.bf16`.** No WMMA/MFMA/FP8 HW |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |

## FA LDS pins (before migrate from extras)

Observed on extras `d71721c79547` (`csrc/rocm/fa_rdna2.cu`). File added
`b1b3fa938` **BlivionIaG** `<kev29lt@gmail.com>`. Not dest-locked (FA
occupancy pin stays closed). Do not invent tok/s. Do not copy the `.cu`
body yet — when you do, cherry-pick BlivionIaG, do not re-author
([`docs/BACKPORT.md`](../../docs/BACKPORT.md)).

| Pin | Notes | Number |
|---|---|---|
| Decode smem | `Br=1`, `Bc=64`, `head_dim=128`, `THREADS=128` | ~33 KiB (extras comment) |
| Prefill smem | `Br=16`, `Bc=64`, `head_dim=128`, `THREADS=128` | ~48 KiB; 64 KiB/CU class, 1 block/CU |
| Prefill leftover launch shape | hygiene `(N,1)` leftover / remainder launches | from extras — fill at migrate |
| TopK / LDS budget | ≤48 KiB working set + `attn_stages` guard | extras prefill ~48 KiB sits on this pin |
| `LDS_PAD` | W4 A-tile pad lives on `gemm/w4a16_fdot2`, not FA | `8` (GEMM tile) |
| `__launch_bounds__` | extras uses 128 / 256 variants | **not dest-locked** — occupancy pin closed |

Scratch is sized by `plan`. One `torch.ops` / V1 entry when the engine bind
lands. No Triton→HIP double-fire.
