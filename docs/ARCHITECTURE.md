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

## Shared DOT source (gfx1030 + gfx1100)

FA, EXL3, AWQ/W4A16, and `moe/shared` are **the same tile source**.
gfx1100 is a **first-class DOT consumer**, not a later port.

| Rule | Meaning |
|---|---|
| One source | `tiles/attn/fa_fdot2`, `tiles/gemm/w4a16_fdot2`, `tiles/gemm/exl3_3inst`, `tiles/moe/shared` |
| Two fatbins | `--offload-arch=gfx1030` and `--offload-arch=gfx1100` as **two CMake trees** |
| No multi-arch object | Never `--offload-arch=gfx1030,gfx1100` in one `.a` / `.so` |
| No WMMA gate | **Never** `#ifdef WMMA` (or WMMA-only paths) in those files |
| WMMA Later | gfx1100-only overlay, optional, never required for DOT |
| wave32 only | DOT tiles do not ship a wave64 path |
| No `fdot2.bf16` | `fdot2` / `v_dot2c` only |

`include/hippihx/dot.hpp` is the compile-time lock. CMake omits the DOT
list from the gfx900 archive so Vega cannot pick up a DOT object.

## Fatbin policy

| Slot | Built? | Loads DOT? | Notes |
|---|---|---|---|
| gfx1030 | yes | yes | V620, ROCm 7.14, wave32 |
| gfx1100 | yes | yes | same source as gfx1030 |
| gfx900 | yes | **no** | `mad_mix` / `pk_fma` — do not conflate with gfx1030 DOT |
| gfx906 | **Later** | **no** | fourth Vega-variant slot; enum + docs only; CMake refuses to configure |

One configure tree → one `libhippihx_<arch>.a` in `build/fatbin/<arch>/`.
CMake rejects a multi-arch `HIPPIHX_ARCH` or `CMAKE_HIP_ARCHITECTURES`
list, and rejects `gfx906` as “Later — not built yet.”

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
DOT ops report `META.dot is True` and `is_supported` only on gfx1030 /
gfx1100. The skeleton is host-side and torch-free.

## Non-goals (room lock)

- Importing or forking b12x CUDA / CuTe / CE / NVFP4 / WMMA sources
- A serve stack, model registry, or vLLM plugin inside this repo
- PRs against upstream vLLM
- Editing `opengfx1030/vllm-rdna` from this tree
- Migrating real `fa_rdna2` / EXL3 / AWQ bodies in this change
