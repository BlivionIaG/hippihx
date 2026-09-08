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

## Fatbin policy

| Rule | Meaning |
|---|---|
| Three slots | `gfx1030`, `gfx1100`, `gfx900` |
| One arch per artifact | `libhippihx_<arch>.a` in `build/fatbin/<arch>/` |
| No shared objects across arches | Do not ship one `.so` / `.a` with multiple `--offload-arch` |
| No object reuse | A gfx1030 code object is invalid on gfx1100 / gfx900 |
| Default | `HIPPIHX_ARCH=gfx1030` |

CMake rejects a multi-arch `HIPPIHX_ARCH` or `CMAKE_HIP_ARCHITECTURES` list.

Configure **three build trees** if you need all slots. A convenience target
`hippihx_list_fatbin_slots` only prints the policy; it does not merge
objects.

## ROCm pin (V620)

V620 work is pinned to **ROCm 7.14**. That pin is documented and carried as
`HIPPIHX_ROCM_PIN` in CMake. Do not silently retarget gfx1030 tiles to a
newer toolchain without an explicit contract change.

gfx1030: **wave32**. No WMMA, no MFMA, no FP8 hardware. DOT math is
`fdot2` / `v_dot2c` unless a tile README records a different unit.

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

- `attn/fa_fdot2`, `attn/gdn_scan`, `attn/kda_scan`, `attn/qsa_indexer`,
  `attn/dsa_nope`
- `gemm/w4a16_fdot2`, `gemm/exl3_3inst` (consume hook)
- `moe/routed` (gate/up/down), `moe/shared`, `moe/leftover_bf16`
- `comm/pcie` (Uncached + push AR, INT8/Q8 wire class — stub)

Each tile README will lock **LDS** and **`__launch_bounds__`** before ISA
lands. Occupancy notes (VGPR vs `waves_per_eu`) belong there too.

## Produce vs consume

| Lives here | Lives elsewhere |
|---|---|
| HIP that *reads* W4A16 / `3inst` | AWQ produce, EXL3 `-cb 3inst` pack, Viterbi |
| Wire-class codecs used at run | Checkpoint rewrite tools |

Do not add `produce/`, `awq/`, or `3inst/` packer trees to hippihx.

## Python surface

`hippihx.list_ops()` enumerates contracts. Each op is
`hippihx.<group>.<op>` with `Caps`, `plan`, `bind`, `run`, `is_supported`.
The skeleton is host-side and torch-free. Device launches wait for real
tiles + a HIP extension.

## Non-goals (room lock)

- Importing or forking b12x CUDA / CuTe / CE / NVFP4 sources
- A serve stack, model registry, or vLLM plugin inside this repo
- PRs against upstream vLLM
- Editing `opengfx1030/vllm-rdna` from this tree
