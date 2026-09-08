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
- gfx1100 and gfx900 are **separate fatbins**. New build tree per arch.
- CMake must keep failing if someone passes multiple `--offload-arch` values
  into one target.
- gfx1030 tiles must not assume WMMA, MFMA, or FP8 hardware.

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
