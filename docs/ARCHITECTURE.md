# Architecture

hippihx is a HIP **op zoo**. `opengfx1030/vllm-rdna` `rdna_extras` is the
**serve wiring**. The split is the same *shape* as
`local-inference-lab/b12x` → a serving engine, but the ISA is HIP/RDNA, not
CUDA/CuTe.

## Why this repo exists

Kernels that live inside a vLLM fork become undeletable serve bugs: capture
probes, Triton fallbacks, and pack logic accrete next to `torch.ops`
registration. hippihx keeps tile contracts, LDS locks, and fatbins here.
`rdna_extras` should load a library, size scratch from `plan`, bind views,
and launch one V1 op.

```
                    plan(caps) ── scratch specs
                         │
 caller (rdna_extras)    │  zeros(nbytes)  # page-commit
                         ▼
                    bind(plan, scratch, tensors)  # views only
                         │
                         ▼
                    run(binding)  # no D2H under capture
```

## Shared DOT source (gfx1030 + gfx110x)

FA, EXL3, AWQ/W4A16, and `moe/shared` are **the same tile source**.
gfx1100/1101/1102 are **first-class DOT consumers**, not a later port.

| Rule | Meaning |
|---|---|
| One source | `tiles/attn/fa_fdot2`, `tiles/gemm/w4a16_fdot2`, `tiles/gemm/exl3_3inst`, `tiles/moe/shared` |
| Separate fatbins | one `--offload-arch` per CMake tree (every built DOT slot) |
| No multi-arch object | Never `--offload-arch=gfx1030,gfx1100` in one `.a` / `.so` |
| No foreign ISA load | **Never** `HSA_OVERRIDE_GFX_VERSION` or load gfx1030 objects on another GFX |
| No WMMA gate | **Never** `#ifdef WMMA` (or WMMA-only paths) in those files |
| WMMA Later | gfx110x overlay, optional, never required for DOT |
| wave32 only | DOT tiles do not ship a wave64 path |
| No `fdot2.bf16` | `fdot2` / `v_dot2c` only |

`include/hippihx/dot.hpp` is the compile-time lock. CMake omits the DOT
list from the gfx900 archive.

## Fatbin policy

| Target | Built? | Shared DOT with gfx1030? | Notes |
|---|---|---|---|
| **gfx1030** | yes (primary) | — | V620 dest |
| **gfx1100/1101/1102** | yes | **yes** (same `dot.hpp`, no WMMA gate) | separate fatbin |
| **gfx1151** Strix Halo | yes (portable) | **yes** (same `dot.hpp`) | can run, not dest-tuned; not WMMA-gated |
| **gfx1031/1032/1033/1035/1036** Deck/mobile | yes (portable) | **yes** (RDNA2 DOT, **wave32**) | Steam Deck is **gfx1033 / wave32**. Same gen as gfx1030; not dest-tuned |
| **gfx1013** BC-250 | yes (portable) | **yes** (RDNA2 DOT stubs; not dest-tuned) | **RDNA2** Cyan Skillfish — same generation as V620/Deck, different GFX (not Navi21, not Van Gogh). Can run, unoptimized. See wave note. |
| **gfx900** | yes stub / Later mad_mix | **no** | never load FA/EXL3 DOT |
| **gfx906** (real Vega20/MI50) | Later non-DOT if ever | **no** | **not** BC-250 |

One configure tree → one `libhippihx_<arch>.a` in `build/fatbin/<arch>/`.
CMake rejects multi-arch lists and refuses **gfx906** only. Portable DOT
slots (1151 / 103x / 1013) configure and compile the same stubs.

**BC-250 is RDNA2.** It is Cyan Skillfish **`gfx1013`** (Oberon cut-down
APU) — same generation as V620 **gfx1030** and Steam Deck **gfx1033**,
not a different ISA family. It is **not** Vega20/`gfx906`, **not** Navi21,
and **not** Van Gogh. Own `--offload-arch=gfx1013` only. `gfx906` is
MI50/Vega20 only.

### gfx1013 wave note (BC-250 / Cyan Skillfish)

gfx1013 **builds** (RDNA2, portable / unoptimized; CMake does not refuse
it). RADV / llama.cpp dumps on this Skillfish part report **warp size 64**
and **no matrix cores**. That is a **SKU wave** note — not “not RDNA2”.
Do not apply it to Deck gfx103x (those are wave32).

On real BC-250, still measure `hipDeviceProp.warpSize` and inspect
`hipcc --offload-arch=gfx1013` ISA. If HIP is **wave64**, keep a
Skillfish-only path (`-mwavefrontsize64` or leave the wave32 flag off).
**Do not** force `-mwavefrontsize32` or `HSA_OVERRIDE` to load a Navi21
or Deck object. Serve bind keys on **arch + wave size**. Never load a
gfx1030 / gfx1033 fatbin on gfx1013.

## ROCm pin (V620)

V620 work is pinned to **ROCm 7.14**. That pin is documented and carried as
`HIPPIHX_ROCM_PIN` in CMake. Do not silently retarget gfx1030 tiles to a
newer toolchain without an explicit contract change.

## Engine bind rules

These are room-locked for future Python / `torch.ops` and for anyone wiring
`rdna_extras`:

1. **One V1 entry per op.** `torch.ops.hippihx.<op>` (name TBD at bind time)
   is the only launch path hippihx exposes. The zoo must not also fire a
   Triton implementation of the same op.
2. **Scratch from `plan`.** `Plan.scratch_specs()` is authoritative. The
   serve layer allocates; `bind` only views.
3. **Zero for cudagraph page-commit.** Use `torch.zeros` / `torch::zeros`
   (or equivalent) so RDNA2 page-commits under capture. Do not assume
   `empty` + first write is enough on V620.
4. **No D2H under capture.** No `.item()`, host prints of device scalars,
   or debug probes on the hot path while the stream is capturing.
5. **`bind` never allocates.** No `empty` / `zeros` / `arange` inside
   `bind`. Fresh binding every call is fine; a cached workspace is not
   required and must not be introduced for the vLLM path.
6. **Key on arch + wave size.** Serve must not select a fatbin from the
   GFX name alone. All of gfx1030 / Deck gfx1033 / gfx1013 are RDNA2;
   Deck is wave32; gfx1013 wave is unverified (RADV reports 64) — still
   a separate object.

## Tile classes

Directories are classes, not SKUs:

- `attn/fa_fdot2` (DOT), `attn/gdn_scan`, `attn/kda_scan`,
  `attn/qsa_indexer`, `attn/dsa_nope`
- `gemm/w4a16_fdot2` (DOT), `gemm/exl3_3inst` (DOT consume hook)
- `moe/routed` (gate/up/down), `moe/shared` (DOT), `moe/leftover_bf16`
- `sequence/causal_conv` — scalar FMA, `state_len≈4`; **not** under
  `gdn_scan`. GDN vs KDA layouts differ (do not retarget GDN 16/48 onto
  KDA 64×128).
- `comm/pcie` — Uncached+push **Later**; INT8/Q8 wire class preferred;
  Leave E4M3 / `f8_dma` without FP8 HW

Each tile README will lock **LDS** and **`__launch_bounds__`** before ISA
lands. Occupancy notes (VGPR vs `waves_per_eu`) belong there too.

### FA LDS pins (before extras migrate)

Recorded on `tiles/attn/fa_fdot2/README.md`. Fill numbers at migrate; do
not invent tok/s:

- prefill leftover launch shape hygiene `(N,1)`
- ≤48 KiB TopK / LDS budget + `attn_stages` guard (64 KiB class)
- `LDS_PAD=8` on W4 A-tile paths where relevant

## Produce vs consume

| Lives here | Lives elsewhere |
|---|---|
| HIP that *reads* W4A16 / `3inst` | AWQ produce, EXL3 `-cb 3inst` pack, Viterbi |
| Wire-class codecs used at run | Checkpoint rewrite tools |

Do not add `produce/`, `awq/`, or `3inst/` packer trees to hippihx.

## Python surface

`hippihx.list_ops()` enumerates contracts. Each op is
`hippihx.<group>.<op>` with `Caps`, `plan`, `bind`, `run`, `is_supported`.
DOT ops report `META.dot is True` and `is_supported` on every built DOT
slot (gfx1030, gfx110x, gfx1151, gfx103x, gfx1013). The skeleton is
host-side and torch-free.

## Non-goals (room lock)

- Importing or forking b12x CUDA / CuTe / CE / NVFP4 / WMMA sources
- A serve stack, model registry, or vLLM plugin inside this repo
- PRs against upstream vLLM
- Editing `opengfx1030/vllm-rdna` from this tree
- Dumping extras `csrc/rocm/*.cu` here before extras can consume hippihx
  (see [`BACKPORT.md`](BACKPORT.md))
- Re-authoring foreign HIP as Cursor / Blivion commits (cherry-pick `-x`,
  keep source Author **and** Committer)
