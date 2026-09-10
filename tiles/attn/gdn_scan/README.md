# attn/gdn_scan

GDN / hybrid-state scan class (Qwen-style linear attention state). Distinct
from `kda_scan`.

| Lock | Status |
|---|---|
| Decode LDS | extras `gdn_decode_rdna2.cu` add `55527010c` **BlivionIaG**: **0**. State is 16 fp32 VGPR/thread. `GDN_THREADS=256`, `GDN_BV=32`, `GDN_K=128` |
| Prefill `o` LDS | extras `gdn_prefill_o_rdna2.cu`: **45312 B** (`s_q`/`s_k` 16 KiB each, `s_h` 8 KiB, `s_v` 4 KiB, `s_g` 256 B; `s_bA` reuses `s_q`) |
| `__launch_bounds__` | extras decode: `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)` |
| Wave | 32 on gfx1030 |
| Invalid slot | extras zeros output and skips state when the cache index is the vLLM `NULL_BLOCK_ID=0` sentinel. Zoo contract: **invalid slot → zero out, do not touch state**. Do not bake the vLLM constant name into hippihx |
| Graph | allocate zeroed state once; never per-step D2H under capture |

Not a product fuse. Not `gdn_decode_rdna2` by another name — this is the
hippihx contract the serve layer will call.

**Do not migrate bodies while dest GDN hybrid is capture-unsafe.** Prefill
HIP is default-off on extras (chunk-boundary corruption). 2026-09-08 dest
profiling still attributes piecewise-graph garbage to the GDN hybrid
state path, with `causal_conv1d_update` captured as a non-splitting op.
That is extras wiring. See [`docs/BACKPORT.md`](../../docs/BACKPORT.md).

**fp16 activations only.** extras HIP (`gdn_prefill_*_rdna2`) rejects
`mixed_qkv` that is not fp16. Dest dispatch selected the HIP prefill
chain on BF16 and only failed inside the kernel — zoo `plan` / V1
must refuse bf16 (`HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`). Do not emit
`fdot2.bf16` for a “BF16 GDN” shortcut.
