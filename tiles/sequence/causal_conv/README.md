# sequence/causal_conv

Short-window causal convolution. **Scalar FMA**, `state_len≈4`.

This is not `gdn_scan` and not `kda_scan`. GDN hybrid-state layouts (e.g.
16 QK / 48 V class) and KDA linear-state layouts (e.g. 64h×128) differ;
do not retarget one onto the other. This stub is the conv contract only.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 / gfx1100 |
| Math | scalar FMA — not `fdot2`, not `fdot2.bf16`, not WMMA |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |
