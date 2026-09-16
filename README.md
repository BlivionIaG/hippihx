# hippihx

HIP kernel / op zoo for RDNA. **gfx1030** (Radeon Pro **V620**, RDNA2,
wave32, ROCm **7.14**) and **gfx1100/1101/1102** are first-class **DOT**
consumers of the **same tile source**. gfx1151, Deck **gfx103x**
(wave32, including Steam Deck gfx1033), and **gfx1013** (BC-250 / Cyan
Skillfish) build the same DOT stubs — they can run, not dest-tuned.
**gfx900** is a Vega stub. **gfx906** is Later Vega20/MI50, not BC-250.

This is the HIP/FlyDSL analogue of
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x):
**plan / bind / run** in the zoo, engine binds one `torch.ops` entry. It
is **not** a b12x clone and must not import CUDA, CuTe, CE, WMMA, or
NVFP4. FlyDSL is a dest **compiler** backend; extras V1 consume is still
HIP fatbins — see [`docs/FLYDSL.md`](docs/FLYDSL.md).

hippihx is the place tile contracts live so
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` can stay a **thin serve wiring layer**.

Consume surface: `include/hippihx/v1.h` — `hippihx_v1_plan` /
`hippihx_v1_run` (ABI rev **3**: group `attn` → `attention`). Serve
wraps one V1 id as one `torch.ops.hippihx.*`. Bodies stay in extras
until that rewire lands.

## Zoo vs serve

| Tree | Owns | Does not own |
|---|---|---|
| **hippihx** (this repo) | Tile contracts, HIP ISA, FlyDSL atoms, scratch *layout*, `plan`/`bind`/`run`, per-arch fatbins | vLLM scheduler, model registry, PagedAttention wrappers, produce/packers |
| **`vllm-rdna` `rdna_extras`** | Serve wiring: load hippihx, register one V1 op, hand caller scratch, capture-safe launch | Kernel bodies, LDS locks, pack formats |

hippihx is a **library**. It is not a mini-vLLM. Do **not** open PRs
against upstream vLLM. Do **not** land kernels by editing
`opengfx1030/vllm-rdna` from this repository.

## Layering (`plan` / `bind` / `run`)

1. **`plan(caps)`** — host-side. May allocate. Returns scratch specs (sizes,
   zeroing) and launch policy. Never creates a serve workspace.
2. **`bind(plan, scratch=…, …)`** — views only. **Never allocates.** Caller
   owns scratch (vLLM workspace manager).
3. **`run(binding)`** — enqueue. Capture-safe. No device-to-host under
   capture.

```python
import hippihx
from hippihx.attention import fa_fdot2

print(hippihx.list_ops())

caps = fa_fdot2.Caps(arch="gfx1100")  # or gfx1030; backend="flydsl" is valid
plan = fa_fdot2.plan(caps)
# caller: scratch = zeros(plan.scratch_specs()[0].nbytes)
binding = fa_fdot2.bind(plan, scratch=None)
fa_fdot2.run(binding)  # stub: no device work yet
```

Engine bind rules: one `torch.ops` / V1 entry per op; scratch from `plan`
and **zeroed** for cudagraph page-commit; no D2H under capture; bind
keys on **arch + wave**.

## Arch targets

| Target | Built? | Shared DOT with gfx1030? | Notes |
|---|---|---|---|
| **gfx1030** | yes (primary) | — | V620 dest. ROCm 7.14. wave32. |
| **gfx1100/1101/1102** | yes | **yes** | separate fatbin each |
| **gfx1151** Strix Halo | yes (portable) | **yes** | can run, not dest-tuned |
| **gfx1031/1032/1033/1035/1036** | yes (portable) | **yes** (wave32) | Steam Deck is gfx1033 |
| **gfx1013** BC-250 | yes (portable) | **yes** | RDNA2 Skillfish; RADV warp 64 — never force `-mwavefrontsize32` |
| **gfx900** | yes stub | **no** | never load FA/EXL3 DOT |
| **gfx906** | Later | **no** | Vega20/MI50, **not** BC-250 |

**Separate fatbins, no shared objects.** Never
`--offload-arch=gfx1030,gfx1100`. **Never `HSA_OVERRIDE_GFX_VERSION`.**

## Shared DOT source

FA (`attention/fa_fdot2`), EXL3 (`gemm/exl3_3inst`), AWQ/W4A16
(`gemm/w4a16_fdot2`), and `moe/shared` are **one source tree** compiled
once per `--offload-arch`. Craft locks: **wave32 only**, **no
`fdot2.bf16`**, **never `#ifdef WMMA`**. See `include/hippihx/dot.hpp`.

## Build

Requires ROCm **7.14** `hipcc` on the V620 box. This tree also configures
a **host compile stub** when `hipcc` is missing.

```bash
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030
cmake --build build

# without ROCm (layout check)
cmake -S . -B build -DHIPPIHX_FORCE_HOST_STUB=ON -DHIPPIHX_ARCH=gfx1030 \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build
./build/fatbin/gfx1030/hippihx_smoke_host

pip install -e ".[dev]"
pytest
```

## Tile layout

Directories are **classes**. See each group README.

```
tiles/attention/fa_fdot2        # DOT (shared gfx1030+gfx1100)
tiles/attention/gdn_scan
tiles/attention/kda_scan
tiles/attention/qsa_indexer
tiles/attention/dsa_nope
tiles/gemm/w4a16_fdot2           # DOT (AWQ/GPTQ pack modes)
tiles/gemm/exl3_3inst            # DOT consume hook; produce stays outside
tiles/moe/routed                 # gate / up / down
tiles/moe/shared                 # DOT
tiles/moe/leftover_bf16
tiles/sequence/causal_conv       # scalar FMA; not under gdn_scan
tiles/comm/pcie                  # Uncached+push Later
```

Host consume contracts that extras should import (not reimplement):
`hippihx.gemm.w4a16_fdot2.pack` (integer ZP, `K_STEP=32`).

FlyDSL dest kernel: `hippihx.flydsl.launch_vec_add`. Atoms: `FDOT2`,
`SDOT4`. Optional extra: `pip install hippihx[flydsl]`.

## Packs stay outside

`3inst` / AWQ **produce** is not a hippihx directory.

## Non-goals

- Not a fork of `b12x` CUDA / CuTe / CE / NVFP4.
- Not a vLLM tree. Not a Triton zoo. Not a produce/packer.
- Not a port of ROCm/FlyDSL MFMA/WMMA GEMM/MoE/FA kernels.
- No multi-arch `.so`. No `#ifdef WMMA` on shared DOT tiles. No
  `fdot2.bf16`. No HSA_OVERRIDE.
- BC-250 is **gfx1013** (Cyan Skillfish), **RDNA2** — not gfx906.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fatbin, DOT, bind, layout
- [`docs/CONSUME.md`](docs/CONSUME.md) — how extras wraps V1
- [`docs/FLYDSL.md`](docs/FLYDSL.md) — FlyDSL compiler backend
- [`docs/BACKPORT.md`](docs/BACKPORT.md) — extras → zoo review
- [`docs/EXTRAS.md`](docs/EXTRAS.md) — unvalidated extras inventory
- [`AGENTS.md`](AGENTS.md) — room locks for agents
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ROCm pin, how to land a tile
