# moe

Split MoE by **role**, not by model name.

| Directory | Class |
|---|---|
| `routed` | Routed expert **gate / up / down** |
| `shared` | Shared (always-on) expert path — **DOT** (same source as FA/EXL3/AWQ) |
| `leftover_bf16` | Leftover dense BF16 (unquantized produce, mHC-style leftovers, cvt→fp16 `fdot2`) |

A fused “one kernel does the whole MoE” launch can still live behind these
contracts; the directories stay split so serve wiring can bind roles
independently.

**Lock per tile:** LDS and `__launch_bounds__` in the child README. DOT
`shared` is wave32, no `fdot2.bf16`, never `#ifdef WMMA`.
