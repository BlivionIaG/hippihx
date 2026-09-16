# hippihx

HIP / FlyDSL op zoo for RDNA. **plan / bind / run** live here so
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` stays thin serve wiring. Shape from
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x);
ISA is HIP/RDNA, not CUDA/CuTe.

**Dest:** gfx1030 (V620, wave32, ROCm **7.14**). gfx110x share the same
DOT source, one fatbin per `--offload-arch`. Consume:
`include/hippihx/v1.h` (`hippihx_v1_plan` / `hippihx_v1_run`, ABI rev
**3**, group `attention`). Kernel bodies stay in extras until extras
wraps those symbols.

```python
import hippihx
from hippihx import Caps, Fabric
from hippihx.attention import fa_fdot2
from hippihx.comm import pcie

print(hippihx.list_ops())

caps = fa_fdot2.Caps(arch="gfx1030")  # or gfx1100; backend="flydsl" is valid
plan = fa_fdot2.plan(caps)
fa_fdot2.run(fa_fdot2.bind(plan, scratch=None))  # stub until HIP migrates

pcie.plan(Caps(arch="gfx1030", fabric=Fabric(hop="pix", switch="88096")))
```

`plan` may allocate metadata. `bind` never allocates. `run` is
capture-safe (no D2H). Bind keys: **arch + wave**; comm also **hop +
switch**. Never `HSA_OVERRIDE_GFX_VERSION`. Never a multi-arch `.so`.

## Split

| | Owns |
|---|---|
| **hippihx** | tiles, FlyDSL kernels, catalog, V1 ABI, pack/fabric contracts |
| **`rdna_extras`** | `torch.ops`, graphs, envs, ACS/`lspci` |
| **outside** | AWQ / EXL3 produce, vLLM scheduler |

No PRs to upstream vLLM. Do not edit the extras fork from this tree.

## Arch

| Target | DOT w/ 1030 | Notes |
|---|---|---|
| **gfx1030** | primary | V620 dest |
| **gfx1100/1101/1102** | yes | first-class, separate fatbin |
| **gfx1151** | portable | can run, not dest-tuned |
| **gfx1031–1036** | portable | Deck **gfx1033** is wave32 |
| **gfx1013** | portable | BC-250 / Cyan Skillfish, **RDNA2**, **not** gfx906 |
| **gfx900** | stub | Vega; never load FA/EXL3 DOT |
| **gfx906** | Later | Vega20/MI50, not BC-250 |

## Tiles

```
tiles/attention/fa_fdot2        # DOT
tiles/attention/{gdn_scan,kda_scan,qsa_indexer,dsa_nope}
tiles/gemm/w4a16_fdot2          # DOT; import hippihx.gemm.w4a16_fdot2.pack
tiles/gemm/exl3_3inst           # DOT consume; produce stays out
tiles/moe/{routed,shared,leftover_bf16}
tiles/sequence/causal_conv
tiles/comm/pcie                 # Uncached+push; PIX on PEX88096
```

DOT tiles: wave32, no `fdot2.bf16`, no `#ifdef WMMA` (`include/hippihx/dot.hpp`).

Host contracts extras should import: `hippihx.gemm.w4a16_fdot2.pack`,
`hippihx.comm.fabric`, `hippihx.comm.pcie.policy`. FlyDSL:
`hippihx.flydsl.launch_vec_add` (`pip install hippihx[flydsl]`).

## Build

```bash
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030 && cmake --build build

cmake -S . -B build -DHIPPIHX_FORCE_HOST_STUB=ON -DHIPPIHX_ARCH=gfx1030 \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build && ./build/fatbin/gfx1030/hippihx_smoke_host

pip install -e ".[dev]" && pytest
```

ROCm **7.14** `hipcc` on the V620 box. Host stub is for layout when
`hipcc` is missing.

## Docs

[`ARCHITECTURE`](docs/ARCHITECTURE.md) · [`CONSUME`](docs/CONSUME.md) ·
[`FLYDSL`](docs/FLYDSL.md) · [`BACKPORT`](docs/BACKPORT.md) ·
[`EXTRAS`](docs/EXTRAS.md) · [`AGENTS`](AGENTS.md) ·
[`CONTRIBUTING`](CONTRIBUTING.md)
