# hippihx

HIP kernel / op zoo for **gfx1030 first** (Radeon Pro **V620**, RDNA2,
wave32), with separate fatbin slots for **gfx1100** and **gfx900**.

hippihx is the place tile contracts live so Blivion’s
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` can stay a **thin serve wiring layer**.

This is the HIP/RDNA analogue of the
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
→ serve split: **plan / bind / run** in the zoo, engine binds one `torch.ops`
entry. It is **not** a b12x clone and must not import CUDA, CuTe, CE, or NVFP4
objects from that tree.

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

caps = fa_fdot2.Caps(arch="gfx1030")
plan = fa_fdot2.plan(caps)
# caller: scratch = zeros(plan.scratch_specs()[0].nbytes)  # serve layer
binding = fa_fdot2.bind(plan, scratch=None)
fa_fdot2.run(binding)  # stub: no device work yet
```

## Arch targets

| Slot | Role | Wave | Hardware assumptions |
|---|---|---|---|
| **gfx1030** | First. V620. **ROCm 7.14** pin. | 32 | No WMMA, no MFMA, no FP8 HW. `fdot2` (`v_dot2c`) is the DOT unit. |
| **gfx1100** | Separate fatbin. | 32 | Do not reuse the gfx1030 object. |
| **gfx900** | Separate fatbin. | 64 | Do not reuse the gfx1030 object. |

**Three fatbins, no shared objects.** One configure tree → one
`libhippihx_<arch>.a` under `build/fatbin/<arch>/`. Never
`--offload-arch=gfx1030,gfx1100` in a single artifact.

## Build fatbins

Requires ROCm **7.14** `hipcc` on the V620 box. This cloud/CI tree also
configures a **host compile stub** when `hipcc` is missing so the layout
stays buildable.

```bash
# Default slot (V620)
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030
cmake --build build
# archive: build/fatbin/gfx1030/libhippihx_gfx1030.a

# Other slots — new build trees, never mixed
cmake -S . -B build-gfx1100 -DHIPPIHX_ARCH=gfx1100
cmake -S . -B build-gfx900  -DHIPPIHX_ARCH=gfx900
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
tiles/attn/fa_fdot2
tiles/attn/gdn_scan
tiles/attn/kda_scan
tiles/attn/qsa_indexer
tiles/attn/dsa_nope
tiles/gemm/w4a16_fdot2
tiles/gemm/exl3_3inst      # consume hook; produce stays outside
tiles/moe/routed           # gate / up / down
tiles/moe/shared
tiles/moe/leftover_bf16
tiles/comm/pcie            # Uncached+push AR later; stub only
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
- No multi-arch `.so`. No WMMA/MFMA/FP8 kernels aimed at gfx1030.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fatbin policy, bind
  rules, zoo vs serve.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ROCm pin, how to land a tile.
