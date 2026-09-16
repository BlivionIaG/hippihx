# Consume hippihx from `rdna_extras`

hippihx is a library. Serve wiring stays in
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras`. Do not edit that fork from this tree.

## One V1 entry per op

C ABI: `include/hippihx/v1.h` (rev **3**).

| Call | When |
|---|---|
| `hippihx_v1_plan(op, caps, specs, nspecs)` | Host. Sizes scratch. Always `zeroed=1`. |
| `hippihx_v1_run(op, caps, scratch, nbytes)` | Capture-safe enqueue. Stub returns `NOT_READY` until a body migrates. |

Python (torch-free, same contracts):

```python
from hippihx.attention import fa_fdot2

caps = fa_fdot2.Caps(arch="gfx1030", wave=32, dtype="fp16")
plan = fa_fdot2.plan(caps)
# extras: scratch = torch.zeros(plan.scratch_specs()[0].nbytes, device="cuda")
binding = fa_fdot2.bind(plan, scratch=scratch, q=q, k=k, v=v)
fa_fdot2.run(binding)
```

Register **one** `torch.ops.hippihx.<op>` that calls `hippihx_v1_run`.
No Triton→HIP double-fire. Bind keys on **arch + wave**, not GFX name
alone. extras V1 consume is the HIP fatbin. FlyDSL extras consume waits
on `FLYDSL_V1_CONSUME` (graph-safe JIT) — see [`FLYDSL.md`](FLYDSL.md).

## Host contracts already here

Import these instead of reimplementing extras bugs:

- `hippihx.gemm.w4a16_fdot2.pack` — integer ZP, `K_STEP=32`, refuse ConfigH
  and 40-wide K-splits.
- `Caps.dtype` — DOT and GDN HIP refuse bf16 (`HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`).

## Fatbin

One `libhippihx_<arch>.a` per configure tree. Never load a gfx1030
object on gfx1013. Never `HSA_OVERRIDE_GFX_VERSION`.

## After extras rewires

Delete the extras `.cu` copy. ISA lives here. Capture arenas,
keepalive, env flags stay extras.
