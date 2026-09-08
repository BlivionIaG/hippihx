# hippihx

HIP kernel / op zoo. **gfx1030** (Radeon Pro **V620**, RDNA2, wave32, ROCm
7.14) and **gfx1100/1101/1102** are first-class **DOT** consumers of the
**same tile source**. **gfx1151** and Deck **gfx103x** build the same DOT
stubs — they can run, not dest-tuned. **gfx1013** (BC-250 / Cyan Skillfish)
is Later VERIFY: RADV reports warp 64 / no matrix cores — do not share
wave32 `dot.hpp` until `hipDeviceProp.warpSize` is measured. **gfx900** is
a Vega stub (`mad_mix` / `pk_fma` — not DOT). **gfx906** is Later
Vega20/MI50, not BC-250.

hippihx is the place tile contracts live so Blivion’s
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` can stay a **thin serve wiring layer**.

This is the HIP/RDNA analogue of the
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
→ serve split: **plan / bind / run** in the zoo, engine binds one `torch.ops`
entry. It is **not** a b12x clone and must not import CUDA, CuTe, CE, WMMA,
or NVFP4 objects from that tree.

## Zoo vs serve

| Tree | Owns | Does not own |
|---|---|---|
| **hippihx** (this repo) | Tile contracts, HIP ISA, scratch *layout*, `plan`/`bind`/`run`, per-arch fatbins | vLLM scheduler, model registry, PagedAttention wrappers, produce/packers |
| **`vllm-rdna` `rdna_extras`** | Serve wiring: load hippihx, register one V1 op, hand caller scratch, capture-safe launch | Kernel bodies, LDS locks, pack formats |

hippihx is a **library**. It is not a mini-vLLM. Do not add an engine, a
model loader, or a plugin that reimplements serve. Consume it from
`rdna_extras`.

Do **not** open PRs against upstream vLLM. Do **not** land kernels by
editing `opengfx1030/vllm-rdna` from this repository.

## Layering (`plan` / `bind` / `run`)

Copied as *methodology* only (same verbs as b12x, HIP runtime):

1. **`plan(caps)`** — host-side. May allocate. Returns scratch specs (sizes,
   zeroing) and launch policy. Never creates a serve workspace.
2. **`bind(plan, scratch=…, …)`** — views only. `narrow` / `view` / stride.
   **Never allocates.** Caller owns scratch (vLLM workspace manager).
3. **`run(binding)`** — enqueue. Capture-safe. No device-to-host under
   cudagraph capture.

Engine bind rules (also in `docs/ARCHITECTURE.md`):

- One `torch.ops` / V1 entry per op. No Triton→HIP double-fire.
- Scratch sized from `plan`, **zeroed** for cudagraph page-commit.
- No D2H (`.item()`, prints, probes) under capture.

```python
import hippihx
from hippihx.attn import fa_fdot2

print(hippihx.list_ops())

caps = fa_fdot2.Caps(arch="gfx1100")  # or gfx1030 — same DOT source
plan = fa_fdot2.plan(caps)
# caller: scratch = zeros(plan.scratch_specs()[0].nbytes)  # serve layer
binding = fa_fdot2.bind(plan, scratch=None)
fa_fdot2.run(binding)  # stub: no device work yet
```

## Arch targets

| Target | Built? | Shared DOT with gfx1030? | Notes |
|---|---|---|---|
| **gfx1030** | yes (primary) | — | V620 dest. ROCm 7.14. wave32. |
| **gfx1100/1101/1102** | yes | **yes** (same `dot.hpp` source, no WMMA gate) | separate fatbin each |
| **gfx1151** Strix Halo | yes (portable) | **yes** (same `dot.hpp`) | can run, not dest-tuned; not WMMA-gated |
| **gfx1031/1032/1033/1035/1036** Deck/mobile | yes (portable) | **yes** (RDNA2 DOT class) | can run, not dest-tuned; separate objects |
| **gfx1013** BC-250 | Later VERIFY | **no** until wave measured | Cyan Skillfish. RADV/llama.cpp: warp 64, no matrix cores. Skillfish-only if HIP is wave64; never force `-mwavefrontsize32` / `HSA_OVERRIDE`. Serve bind keys on **arch + wave**. |
| **gfx900** | yes stub / Later mad_mix | **no** | never load FA/EXL3 DOT |
| **gfx906** (real Vega20/MI50) | Later non-DOT if ever | **no** | **not** BC-250 |

**Separate fatbins, no shared objects.** DOT source may be shared; objects
are not. One configure tree → one `libhippihx_<arch>.a`. Never
`--offload-arch=gfx1030,gfx1100` in one artifact. **Never
`HSA_OVERRIDE_GFX_VERSION` / never load a foreign ISA** (especially never
run gfx1030 objects on gfx1013).

## Shared DOT source

FA (`attn/fa_fdot2`), EXL3 (`gemm/exl3_3inst`), AWQ/W4A16
(`gemm/w4a16_fdot2`), and `moe/shared` are **one source tree** compiled
once per `--offload-arch`:

```bash
# same tiles/*.hip, one --offload-arch per tree
cmake -S . -B build-gfx1030 -DHIPPIHX_ARCH=gfx1030
cmake -S . -B build-gfx1100 -DHIPPIHX_ARCH=gfx1100
# gfx1101 / gfx1102 first-class; gfx1151 / gfx103x portable; gfx1013 Later VERIFY
```

Craft locks: **wave32 only**, **no `fdot2.bf16`**, **never `#ifdef WMMA`**.
See `include/hippihx/dot.hpp`.

## Build fatbins

Requires ROCm **7.14** `hipcc` on the V620 box. This cloud/CI tree also
configures a **host compile stub** when `hipcc` is missing so the layout
stays buildable.

```bash
# DOT slots — same source, one tree per arch
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030
cmake --build build
# archive: build/fatbin/gfx1030/libhippihx_gfx1030.a

cmake -S . -B build-gfx1100 -DHIPPIHX_ARCH=gfx1100
cmake -S . -B build-gfx1151 -DHIPPIHX_ARCH=gfx1151  # portable / unoptimized
cmake -S . -B build-gfx900  -DHIPPIHX_ARCH=gfx900   # no DOT objects
# gfx1013 is Later VERIFY — CMake refuses until warpSize is measured
```

Without ROCm (layout check only):

```bash
cmake -S . -B build -DHIPPIHX_FORCE_HOST_STUB=ON -DHIPPIHX_ARCH=gfx1030 \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build
./build/fatbin/gfx1030/hippihx_smoke_host
```

Use `g++` for the host stub if the default `c++` is clang without a
working `-lstdc++` (common on slim images). V620 builds use `hipcc`.

Raw `hipcc` (same policy: one arch per invocation):

```bash
hipcc --offload-arch=gfx1030 -std=c++17 -I include -c tiles/smoke.hip -o smoke.gfx1030.o
hipcc --offload-arch=gfx1100 -std=c++17 -I include -c tiles/smoke.hip -o smoke.gfx1100.o
```

Python package (protocol stubs, no torch required):

```bash
pip install -e ".[dev]"
pytest
```

## Tile layout

Directories are **classes**. See each group README: LDS and
`__launch_bounds__` will be locked per tile.

```
tiles/attn/fa_fdot2           # DOT (shared gfx1030+gfx1100)
tiles/attn/gdn_scan
tiles/attn/kda_scan
tiles/attn/qsa_indexer
tiles/attn/dsa_nope
tiles/gemm/w4a16_fdot2        # DOT (AWQ/GPTQ pack modes)
tiles/gemm/exl3_3inst         # DOT consume hook; produce stays outside
tiles/moe/routed              # gate / up / down
tiles/moe/shared              # DOT
tiles/moe/leftover_bf16
tiles/sequence/causal_conv    # scalar FMA, state_len≈4; not under gdn_scan
tiles/comm/pcie               # Uncached+push Later; INT8/Q8 wire; stub
```

## Packs stay outside

`3inst` / AWQ **produce** is not a hippihx directory. Packers, Viterbi, and
checkpoint rewrite live in their own tools. hippihx only documents the
consume layout and ships HIP that reads it.

## Non-goals

- Not a fork or copy of `local-inference-lab/b12x` CUDA / CuTe / CE / NVFP4.
- Not a vLLM tree. Not a Triton zoo. Not a produce/packer.
- No production GEMM or attention ISA in this skeleton (stubs exist so
  CMake is real).
- No multi-arch `.so`. No `#ifdef WMMA` on shared DOT tiles. No
  `fdot2.bf16`. No DOT objects on gfx900 / gfx906. No HSA_OVERRIDE.
- BC-250 is **gfx1013** (Cyan Skillfish), not gfx906. Later VERIFY:
  measure `hipDeviceProp.warpSize` before sharing wave32 DOT/FA.
  Serve bind keys on **arch + wave size**.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fatbin policy, DOT
  source, bind rules, zoo vs serve.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ROCm pin, how to land a tile.
