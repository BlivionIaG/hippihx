# Contributing

## Where work belongs

| Change | Land in |
|---|---|
| Tile contract, HIP ISA, LDS / `__launch_bounds__` lock | **this repo** (`tiles/…`) |
| `plan` / `bind` / `run`, scratch specs | **this repo** (`hippihx/`) |
| Fatbin / CMake / ROCm pin | **this repo** |
| Observed extras LDS / launch numbers (no body dump) | **this repo** (tile READMEs) |
| extras HIP body migrate | **this repo**, only after extras can consume one V1 op |
| Dest extras HIP (this lab only) | **BlivionIaG** `<kev29lt@gmail.com>` — Author **and** Committer |
| Foreign HIP (leapdragon / a17t) | cherry-pick `-x`, **their** Author **and** Committer — never Cursor, never Blivion |
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

## Attribution

**Foreign HIP is inviolable.** leapdragon (`rdna_ar`, Aron Hsiao
`<leapdragon@gmail.com>`) and a17t (Simon Siebert
`<Mail@simonsiebert.de>` and the Author on their unique HIP commits)
keep **their** Author **and** Committer. Location on dest extras does
**not** make them this lab. Never rewrite them as Blivion or Cursor.
Keep their `Co-authored-by` / `Signed-off-by`. Add none of ours.
Same bar as extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1).

**This lab's dest HIP** (FA, EXL3, W4A16, GDN, causal_conv, …) is
**BlivionIaG** `<kev29lt@gmail.com>`. Author **and** Committer. Do not
leave Cursor / `cursoragent` on those commits.

| Rule | Meaning |
|---|---|
| Cherry-pick `-x` | One source commit → one hippihx commit. Not a squash. Not a re-authored file dump. |
| Foreign first | If source `%an`/`%ae` is leapdragon / Aron Hsiao / a17t / Simon Siebert / their HIP commit Author, **stop**. Use the foreign recipe. Never `GIT_AUTHOR_NAME=BlivionIaG`. |
| This lab only | Author **and** Committer = BlivionIaG `<kev29lt@gmail.com>` — **only** when the source is this lab's HIP |
| Trailers | Foreign: keep theirs; **do not add** Cursor, Cursor Agent, or BlivionIaG. This lab: BlivionIaG only. |
| Mixed commits | Skip. Do not take a commit that mixes their kernel with serve wiring, PLE, tok/s docs, or a second op family. |
| Wiring after pick | Conflict resolution and plan/bind wrappers are **new** hippihx commits. They must not rewrite kernel bodies. |
| Verify foreign | unique leapdragon / a17t commits: Author **and** Committer match the source (Aron Hsiao / a17t). Zero Blivion/Cursor trailers. |
| Verify this lab | this-lab dest-HIP commits show `BlivionIaG <kev29lt@gmail.com>` for Author and Committer |

```bash
# Classify from SOURCE Author. Location on dest extras is not a classifier.
name=$(git -C "$FOREIGN" log -1 --format='%an' "$src")
email=$(git -C "$FOREIGN" log -1 --format='%ae' "$src")

foreign=0
case "$email" in
  leapdragon@gmail.com|Mail@simonsiebert.de) foreign=1 ;;
esac
case "$name" in
  *leapdragon*|*"Aron Hsiao"*|a17t|*"Simon Siebert"*) foreign=1 ;;
esac

if [ "$foreign" = 1 ]; then
  # Foreign FIRST (leapdragon / a17t). Author is preserved by cherry-pick.
  # Committer must be forced or it becomes whoever ran the pick.
  # NEVER GIT_AUTHOR_NAME=BlivionIaG / Cursor. NEVER --reset-author.
  GIT_COMMITTER_NAME="$name" GIT_COMMITTER_EMAIL="$email" \
    git cherry-pick -x "$src"
else
  # This lab's dest HIP only — not leapdragon, not a17t, not anyone else.
  GIT_AUTHOR_NAME='BlivionIaG' GIT_AUTHOR_EMAIL='kev29lt@gmail.com' \
  GIT_COMMITTER_NAME='BlivionIaG' GIT_COMMITTER_EMAIL='kev29lt@gmail.com' \
    git cherry-pick -x "$src"
fi
```

Do not `git commit --amend --reset-author` on **foreign** picks. That
replaces Author with the picker. Do not set `GIT_AUTHOR_*` on those
picks at all.

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
