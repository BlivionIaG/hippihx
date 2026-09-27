# ISA contracts

The durable asset in hippihx is the ISA contract, not the language that
emits it. HIP fatbins (`tiles/`), FlyDSL atoms (`hippihx.flydsl`), and the
Mojo/MAX authoring tree (`mojo/`) all read the same locks:

| Mirror | Path |
|---|---|
| Python | `hippihx.isa` (`hippihx/_lib/isa.py`) |
| C++ | `include/hippihx/isa.hpp` (included from `arch.hpp`) |
| Mojo | `mojo/isa/contracts.mojo` |

`include/hippihx/dot.hpp` remains the compile-time DOT lock (`#error` on
Vega, gfx1013, wave64, and `fdot2.bf16`). The table below is what that
header, the Mojo arch switch, and `plan` are not allowed to weaken.

## Packed DOT (gfx1030 primary)

| Contract | Value |
|---|---|
| `fdot2` | `llvm.amdgcn.fdot2` → `v_dot2_f32_f16` / `v_dot2c_f32_f16` |
| `sdot4` | `llvm.amdgcn.sdot4` → `v_dot4_i32_i8` / `v_dot4c_i32_i8` |
| Activations on DOT and GDN | fp16. V1 refuses bf16 |
| `fdot2.bf16` | never (gfx1030 LLVM ISel abort) |
| WMMA gate on gfx1030 | **off**. No `#ifdef WMMA` on the shared DOT source |
| MFMA / `sudot4` | not this zoo |

gfx110x may grow an optional WMMA overlay later. That overlay is not a
gate: DOT still builds and runs without it. gfx1030 has no matrix core
to gate on.

## LDS banks and the wave32 occupancy gate

RDNA LDS is **32 banks × 4 bytes**. The W4 A-tile pad is **`LDS_PAD=8`**
(`gemm/w4a16_fdot2`, `moe/routed`). That pad is the bank-conflict contract.
Do not invent a second pad per emit language.

DOT tiles are **wave32 only**. `wave_occupancy_ok(dot=True, wave=64)` is
false. QSA on gfx1030 keeps the **4-warp / 128-thread** occupancy gate
(6h×256 prefill). The 2-warp profile spills; ranked explore records it
so a search does not promote it. These are occupancy locks, not tok/s.

## Arch switch and fatbin policy

`hippihx.isa.arch_switch` returns one class per arch:

| Switch | Arches | DOT objects |
|---|---|---|
| `dot` | **gfx1030** primary; gfx1100/1101/1102; **gfx1200**; portable gfx1151 and gfx1031–1036 | yes, one `--offload-arch` each |
| `mad_mix` | gfx900 (built stub); gfx906 and gfx1013 (Later, not built) | **no** |

`refuse_dot_on_mad_mix` is the placement check. gfx1030 packed-DOT objects
do not ship onto gfx900/906/1013. Never `HSA_OVERRIDE`. Never one
multi-arch `.so`. CMake still refuses Later slots and omits the DOT list
from the gfx900 archive.

`plan(...).meta["arch_switch"]` carries the same class. Ranked explore
shapes for a DOT op are empty on `mad_mix`.

## What is not an ISA contract

Family names (`qwen.qsa`, `qwen.gdn`, `qwen.ple`, `moe.routed`,
`moe.leftover_bf16`, `hybrid.heap`) are bind hooks plus tensor views.
They select a catalog op and a caller-owned buffer. They do not add a
packed-DOT opcode. GLM KDA and the DeepSeek transplant are not new hooks.
See [`MOJO.md`](MOJO.md).

The Mojo mirror of this table is authoring text. It is not the dest
produce ABI. Dest produce stays `hippihx_v1_*` loaded by `hipModuleLoad`
until a documented HIP re-emit or MAX-serve soak exists.
