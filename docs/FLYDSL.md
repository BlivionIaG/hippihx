# FlyDSL on gfx1030 — dest compiler backend

Sourced from [rdna-hip-wiki `engine/flydsl.md`](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/engine/flydsl.md)
(`ROCm/FlyDSL`). hippihx is a **HIP + FlyDSL** zoo. The compiler substrate
is dest here. extras V1 consume is still the HIP fatbin.

## Split

| Path | Owns |
|---|---|
| `tiles/` | HIP ISA / fatbins (extras `hippihx_v1_*`) |
| `hippihx.flydsl` | FlyDSL atoms + kernel contracts |
| `hippihx.<group>.<op>` | `plan` / `bind` / `run` (backend-agnostic) |
| `rdna_extras` | one `torch.ops` wrap; graphs; envs |

`Caps(arch="gfx1030", backend="flydsl")` is valid. `FLYDSL_DEST` is
**true**. `FLYDSL_V1_CONSUME` stays **false** until a graph-safe JIT path
exists (warm cache, no compile / alloc / D2H under capture).

FlyDSL is an **optional extra** (`pip install hippihx[flydsl]`). Host
protocol tests do not require the wheel. A wheel existing is not proof of
a dest GEMM.

## What is dest

- Compiler pipeline: Python → Fly/ROCDL → LLVM → gfx1030 HSACO, wave32.
- DOT atoms in `hippihx.flydsl.atoms`: `fly_fdot2` → `v_dot2*`,
  `fly_sdot4` → `v_dot4*`. Prefer `llvm.amdgcn.fdot2` / `sdot4`.
- First kernel contract: `hippihx.flydsl.VEC_ADD` (vector-add, wave32,
  no gfx11/12 opcodes).

## What is not dest

Do **not** port these ROCm/FlyDSL kernels (CDNA MFMA or gfx11/12 WMMA):

- `preshuffle_gemm`, `moe_gemm_2stage`
- `rdna3_f16_gemm`, `rdna_f16_gemm`, `rdna_fp8_preshuffle_gemm`
- `flash_attn_generic` (wave64 MFMA — not RDNA2-generic)

Do not call FlyDSL `WMMA` helpers on gfx1030. Do not use `sudot4` or
`fdot2.bf16`. `refuse_shipped_kernel(name)` encodes that lock.

## Kernel landing (in this tree)

Flip the matching `FLYDSL_GATE*` flag in `hippihx/_lib/backend.py` in the
**same** commit as the proof, not before.

0. gfx1030 code object + vector-add correctness (`FLYDSL_GPU_ARCH=gfx1030`).
1. `fdot2` / `sdot4` wrappers lower to `v_dot2*` / `v_dot4*`.
2. Small-M GEMV correctness and graph-safe warm/cache behavior.
3. Occupancy + performance vs HIP skinny, Triton, rocBLAS.
4. Then a narrow extras consume path. MoE/attention are later rewrites.

ISA evidence is mandatory. Spill or a bad ISel is a drop.
