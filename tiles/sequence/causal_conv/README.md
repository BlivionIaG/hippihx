# sequence/causal_conv

Short-window causal convolution. **Scalar FMA**, `state_len≈4`.

This is not `gdn_scan` and not `kda_scan`. GDN hybrid-state layouts (e.g.
16 QK / 48 V class) and KDA linear-state layouts (e.g. 64h×128) differ;
do not retarget one onto the other. This stub is the conv contract only.

| Lock | Status |
|---|---|
| LDS bytes | extras `causal_conv1d_rdna2.cu` add `5eb84b4fa` **agent@opencode.local**: **0** — register-only, no global scratch. Keep that Author/Committer if picked; do not re-author as Cursor Agent |
| `__launch_bounds__` | unset in extras (one warp per dim-block) |
| Wave | 32 on gfx1030 / gfx1100; extras: 32 threads / dim-block, dim multiple of 32 |
| Math | scalar FMA, fp16 in/out, fp32 acc — not `fdot2`, not `fdot2.bf16`, not WMMA |
| State | `state_len = width-1`, typically **3 or 4** for GDN |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |

extras decode kernel is cudagraph-safe *as ISA* (no device alloc). Dest
still captures it on a per-step tensor in GDN hybrid piecewise graphs —
that stays extras. Do not copy the ATen wrapper.
