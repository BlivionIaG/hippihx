# sequence/causal_conv

Short-window causal convolution. **Scalar FMA**, `state_len≈4`.

This is not `gdn_scan` and not `kda_scan`. GDN hybrid-state layouts (e.g.
16 QK / 48 V class) and KDA linear-state layouts (e.g. 64h×128) differ;
do not retarget one onto the other. This stub is the conv contract only.

| Lock | Status |
|---|---|
| LDS bytes | extras `causal_conv1d_rdna2.cu` add `5eb84b4fa` **BlivionIaG**: **0** — register-only, no global scratch |
| `__launch_bounds__` | unset in extras (one warp per dim-block) |
| Wave | 32 on gfx1030 / gfx1100; extras: 32 threads / dim-block, dim multiple of 32 |
| Math | scalar FMA, fp16 or bf16 in/out, **fp32 acc / mul**. **Not** `fdot2`. **Not** `fdot2.bf16`. gfx1030 LLVM aborts on `llvm.amdgcn.fdot2.bf16.bf16` — BF16 must promote the multiply, not retarget DOT. |
| Decode FIR order | extras @ `cafe95ef8` / tip `1046782`: FIR on **pre-shift** state, then shift (matches fwd). Do not shift-first. |
| State | `state_len = width-1`, typically **3 or 4** for GDN |
| Graph | scratch sized by `plan`, **zeroed** for page-commit; **no D2H under capture** |

extras decode kernel is cudagraph-safe *as ISA* for **fp16**. Dest GDN
capture now uses permanent state arenas (extras, not this tile). BF16
was rolled back off the Triton/`fdot2.bf16` path on the Flash-Next V620
integration (fp32-promoted scalar FMA). Do not copy the ATen wrapper.

C consume id: `HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV`. `hippihx_v1_plan`
returns a **0-byte** zeroed workspace (register-only). `hippihx_v1_run`
returns `HIPPIHX_V1_ERR_NOT_READY` until the body migrates. bf16 caps are
accepted; FA/GDN/DOT are not.
