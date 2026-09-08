# Contributing

## Where work belongs

| Change | Land in |
|---|---|
| Tile contract, HIP ISA, LDS / `__launch_bounds__` lock | **this repo** (`tiles/…`) |
| `plan` / `bind` / `run`, scratch specs | **this repo** (`hippihx/`) |
| Fatbin / CMake / ROCm pin | **this repo** |
| vLLM registration, flags, model hooks | `opengfx1030/vllm-rdna` `rdna_extras` (not from this tree) |
| AWQ / `3inst` produce | packer tools — **not** a hippihx directory |
| Upstream vLLM | **do not open PRs** |

Do not copy CUDA, CuTe, CE, or NVFP4 objects from
`local-inference-lab/b12x`. Copy the **plan / bind / run** verbs only.

## ROCm / arch

- V620 = **gfx1030**, **wave32**, **ROCm 7.14**.
- **gfx1100/1101/1102** are first-class DOT consumers of the **same** FA /
  EXL3 / AWQ / `moe.shared` source. One `--offload-arch` per fatbin.
- gfx900 is a Vega stub (`mad_mix` / `pk_fma`). It does **not** load DOT
  tiles.
- **BC-250 is gfx1013** (Cyan Skillfish): Later VERIFY. RADV/llama.cpp
  report warp size 64 and no matrix cores. Do not share wave32 `dot.hpp`
  / FA with gfx1030 until `hipDeviceProp.warpSize` and
  `hipcc --offload-arch=gfx1013` ISA are measured. If HIP is wave64, use
  a Skillfish-only path (`-mwavefrontsize64` or leave wave32 off). Never
  force `-mwavefrontsize32` or `HSA_OVERRIDE` to fake Navi21. Serve bind
  keys on **arch + wave size**. **gfx906 is Vega20/MI50**, Later non-DOT,
  **not** BC-250.
- gfx1151 and Deck gfx103x are built portable DOT fatbins (same stubs,
  can run, not dest-tuned).
- CMake must refuse multi-arch lists, gfx906, and gfx1013 (until VERIFY),
  and never imply `HSA_OVERRIDE` / foreign ISA load.
- Shared DOT tiles: **wave32 only**, **no `fdot2.bf16`**, **never
  `#ifdef WMMA`**. WMMA is a gfx110x-only Later overlay.

## Landing a tile

1. Directory already exists under `tiles/<class>/…`. Do not name it after a
   product (no `qwen38/`, no `glm53/`).
2. Fill the tile README **locks** (LDS bytes, `__launch_bounds__`, wave,
   DOT unit) *before* claiming the kernel is real.
3. One HIP entry that will become one `torch.ops` / V1 symbol. No second
   Triton path in this library.
4. Scratch sized in `plan`. Serve zeros it for page-commit. `bind` views
   only. `run` is capture-safe (no D2H).
5. Host tests for the protocol stay torch-free until a HIP extension exists.
6. If the tile is DOT, include `hippihx/dot.hpp` (not a WMMA header) and
   mark `make_op(..., dot=True)`. Do not add a per-SKU copy of the file.
   gfx1151 / gfx103x compile the same header (portable, not dest-tuned).
   gfx1013 is Later VERIFY — do not compile this header for it until
   wave size is measured. Never HSA_OVERRIDE a dest object onto those
   chips.

## Build checks

```bash
# Layout / host stub (no ROCm). Prefer g++ if clang cannot find libstdc++.
cmake -S . -B build -DHIPPIHX_FORCE_HOST_STUB=ON -DCMAKE_CXX_COMPILER=g++
cmake --build build
./build/fatbin/gfx1030/hippihx_smoke_host

# Silicon (V620)
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030
cmake --build build

pip install -e ".[dev]"
pytest
```

Dump `hipcc -Rpass-analysis=kernel-resource-usage` (VGPR / LDS /
`waves_per_eu`, `v_dot2c`) in the tile README when ISA lands. Spill or a
bad ISel is a drop, not a “tune later”.

## Python

Keep `hippihx` importable without torch and without a built fatbin. Optional
HIP extensions can load lazily later. `list_ops()` must stay in lockstep
with `tiles/` class directories.
