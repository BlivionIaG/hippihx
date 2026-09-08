# Contributing

## Where work belongs

| Change | Land in |
|---|---|
| Tile contract, HIP ISA, LDS / `__launch_bounds__` lock | **this repo** (`tiles/…`) |
| `plan` / `bind` / `run`, scratch specs | **this repo** (`hippihx/`) |
| Fatbin / CMake / ROCm pin | **this repo** |
| Observed extras LDS / launch numbers (no body dump) | **this repo** (tile READMEs) |
| extras HIP body migrate | **this repo**, only after extras can consume one V1 op |
| Foreign HIP (leapdragon / kletorch / a17t / extras agent) | cherry-pick `-x`, **their** Author **and** Committer — never a Cursor rewrite |
| vLLM registration, flags, capture, model hooks | `opengfx1030/vllm-rdna` `rdna_extras` (not from this tree) |
| AWQ / `3inst` produce | packer tools — **not** a hippihx directory |
| Upstream vLLM | **do not open PRs** |

Do not copy CUDA, CuTe, CE, or NVFP4 objects from
`local-inference-lab/b12x`. Copy the **plan / bind / run** verbs only.

Do not dump `opengfx1030/vllm-rdna` `csrc/rocm/*.cu` into `tiles/` until
the extras consume path exists. Those files are ATen wrappers + paged
layouts + capture probes. ISA migrate is extract-then-rewire in extras
(see [`docs/BACKPORT.md`](docs/BACKPORT.md)). Until then, extras
observations belong in the tile README locks — not a second kernel body.

## Attribution (foreign HIP)

If the HIP / commit is **not** new hippihx work, **do not rewrite it**.
Keep only the original authors' commits. Same rule as extras
[PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1) (leapdragon
`rdna_ar`).

| Rule | Meaning |
|---|---|
| Cherry-pick `-x` | One source commit → one hippihx commit. Not a squash. Not a re-authored file dump. |
| Author **and** Committer | Copy both from the source commit (`GIT_COMMITTER_NAME` / `EMAIL` = source Author). Do not leave Committer as Cursor / Blivion / `cursoragent`. |
| Trailers | Keep the source `Co-authored-by` / `Signed-off-by`. **Do not add** Cursor, Cursor Agent, or BlivionIaG trailers to *their* commits. |
| Mixed commits | Skip. Do not take a commit that mixes their kernel with serve wiring, PLE, tok/s docs, or a second op family. |
| Wiring after pick | Conflict resolution and plan/bind wrappers are **new** hippihx commits (agent/Blivion). They must not rewrite kernel bodies. |
| Verify | `git log --format='%an <%ae> | %cn <%ce> | %s'` on unique imported commits matches the source. |

```bash
# Example: leapdragon commit onto a hippihx side branch (not dest).
# Author is preserved by cherry-pick; Committer must be forced or it
# becomes whoever ran the pick.
src=<sha>
name=$(git -C "$EXTRAS" log -1 --format='%an' "$src")
email=$(git -C "$EXTRAS" log -1 --format='%ae' "$src")
GIT_COMMITTER_NAME="$name" GIT_COMMITTER_EMAIL="$email" \
  git cherry-pick -x "$src"
```

Do not `git commit --amend --reset-author`. That replaces Author with
the picker.

## ROCm / arch

- V620 = **gfx1030**, **wave32**, **ROCm 7.14**.
- **gfx1100/1101/1102** are first-class DOT consumers of the **same** FA /
  EXL3 / AWQ / `moe.shared` source. One `--offload-arch` per fatbin.
- gfx900 is a Vega stub (`mad_mix` / `pk_fma`). It does **not** load DOT
  tiles.
- **BC-250 is gfx1013** (Cyan Skillfish): **RDNA2**, same generation as
  V620/Deck, built portable/unoptimized. Different GFX than Navi21 or
  Van Gogh — not Steam Deck. RADV/llama.cpp report Skillfish warp size
  64 — do not force `-mwavefrontsize32` or `HSA_OVERRIDE`. Serve bind
  keys on **arch + wave size**. **gfx906 is Vega20/MI50**, Later non-DOT,
  **not** BC-250.
- gfx1151 and Deck gfx103x (including Steam Deck **gfx1033 / wave32**)
  are built portable DOT fatbins (same stubs, can run, not dest-tuned).
- CMake must refuse multi-arch lists and gfx906, and never imply
  `HSA_OVERRIDE` / foreign ISA load.
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
   gfx1151 / gfx103x / gfx1013 compile the same header (portable, not
   dest-tuned). Deck is wave32; gfx1013 does not force wave32. Never
   HSA_OVERRIDE a dest object onto those chips.

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
