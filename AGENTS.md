# hippihx guidance

HIP/RDNA op zoo. Methodology from
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
(plan / bind / run). Silicon and dest policy from
[`BlivionIaG/rdna-hip-wiki`](https://github.com/BlivionIaG/rdna-hip-wiki).

## Split

| Layer | Owns |
|---|---|
| **this repo** | Tile contracts, HIP fatbins, FlyDSL atoms, catalog, V1 ABI |
| **`opengfx1030/vllm-rdna` `rdna_extras`** | `torch.ops`, envs, graphs, model hooks |
| **produce** | AWQ / EXL3 `-cb 3inst` packers |

Do not open PRs against upstream vLLM. Do not edit the extras fork from
this tree. Do not dump `csrc/rocm/*.cu` here until extras consumes V1.

## Like b12x, not a clone

- Ops live at `hippihx.<group>.<op>` with `api.py` + `META`.
- Group is `attention` (not `attn`). ISA names stay (`fa_fdot2`, not `paged`).
- Catalog is `hippihx/_lib/catalog.py` — one table for Python, C, tiles, tests.
- Never import `b12x`, CuTe, CUTLASS, CUDA, WMMA, NVFP4.

## HIP fatbins / FlyDSL compiler

Both are dest zoo backends. extras V1 consume is HIP (`FLYDSL_V1_CONSUME`
is false). FlyDSL is an optional extra, not a required dep. Do not port
MFMA/WMMA FlyDSL GEMM/MoE/FA into `tiles/` or `hippihx.flydsl`.
`Caps(backend="flydsl")` is valid. See `docs/FLYDSL.md`.

## Craft locks

- One `--offload-arch` per artifact. Never `HSA_OVERRIDE`.
- Shared DOT: wave32, `fdot2` / `v_dot2c`, never `fdot2.bf16`, never `#ifdef WMMA`.
- DOT + GDN HIP: fp16 activations. V1 refuses bf16.
- W4: integer `q - zero` then scale. `K_STEP=32`. No ConfigH. One GEMM family.
- `bind` never allocates. `run` never D2H under capture. Scratch zeroed by serve.
- Live request counts are runtime args, not compile keys (b12x rule, same here).

## Attribution

Dest extras HIP: Author **and** Committer `BlivionIaG <kev29lt@gmail.com>`.
Foreign HIP: cherry-pick `-x`, keep **their** Author and Committer. See
`CONTRIBUTING.md`.

## Tests before claims

Host protocol tests are dest for this skeleton. No tok/s. Spill or a bad
ISel is a drop. Fill tile README LDS / `__launch_bounds__` before claiming
a kernel is real.
