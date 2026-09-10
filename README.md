# hippihx

HIP kernel / op zoo. **gfx1030** (Radeon Pro **V620**, RDNA2, wave32, ROCm
7.14) and **gfx1100/1101/1102** are first-class **DOT** consumers of the
**same tile source**. **gfx1151**, Deck **gfx103x** (**wave32**, including
Steam Deck gfx1033), and **gfx1013** (BC-250 / Cyan Skillfish, **also
RDNA2**) build the same DOT stubs — they can run, not dest-tuned.
gfx1013 is the same generation as V620/Deck, a different GFX; RADV
reports Skillfish warp 64 — do not force `-mwavefrontsize32`.
**gfx900** is a Vega stub (`mad_mix` / `pk_fma` — not DOT). **gfx906** is
Later Vega20/MI50, not BC-250.

hippihx is the place tile contracts live so Blivion’s
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` can stay a **thin serve wiring layer**.

Consume surface (in progress): `include/hippihx/v1.h` —
`hippihx_v1_plan` / `hippihx_v1_run`. Serve wraps one V1 id as one
`torch.ops.hippihx.*`. Bodies stay in extras until that rewire lands.

This is the HIP/RDNA analogue of the
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
→ serve split: **plan / bind / run** in the zoo, engine binds one `torch.ops`
entry. It is **not** a b12x clone and must not import CUDA, CuTe, CE, WMMA,
or NVFP4 objects from that tree.

## Zoo vs serve

| Tree | Owns | Does not own |
|---|---|---|
| **hippihx** (this repo) | Tile contracts, HIP ISA, scratch *layout*, `plan`/`bind`/`run`, per-arch fatbins | vLLM scheduler, model registry, PagedAttention wrappers, produce/packers |
| **`vllm-rdna` `rdna_extras`** | Serve wiring: load hippihx, register one V1 op, hand caller scratch, capture-safe launch | Kernel bodies, LDS locks, pack formats |

hippihx is a **library**. It is not a mini-vLLM. Do not add an engine, a
model loader, or a plugin that reimplements serve. Consume it from
`rdna_extras`.

Do **not** open PRs against upstream vLLM. Do **not** land kernels by
editing `opengfx1030/vllm-rdna` from this repository.

## Layering (`plan` / `bind` / `run`)

Copied as *methodology* only (same verbs as b12x, HIP runtime):

1. **`plan(caps)`** — host-side. May allocate. Returns scratch specs (sizes,
   zeroing) and launch policy. Never creates a serve workspace.
2. **`bind(plan, scratch=…, …)`** — views only. `narrow` / `view` / stride.
   **Never allocates.** Caller owns scratch (vLLM workspace manager).
3. **`run(binding)`** — enqueue. Capture-safe. No device-to-host under
   cudagraph capture.

Engine bind rules (also in `docs/ARCHITECTURE.md`):

- One `torch.ops` / V1 entry per op. No Triton→HIP double-fire.
- Scratch sized from `plan`, **zeroed** for cudagraph page-commit.
- No D2H (`.item()`, prints, probes) under capture.

```python
import hippihx
from hippihx.attn import fa_fdot2

print(hippihx.list_ops())

caps = fa_fdot2.Caps(arch="gfx1100")  # or gfx1030 — same DOT source
plan = fa_fdot2.plan(caps)
# caller: scratch = zeros(plan.scratch_specs()[0].nbytes)  # serve layer
binding = fa_fdot2.bind(plan, scratch=None)
fa_fdot2.run(binding)  # stub: no device work yet
```

## Arch targets

| Target | Built? | Shared DOT with gfx1030? | Notes |
|---|---|---|---|
| **gfx1030** | yes (primary) | — | V620 dest. ROCm 7.14. wave32. |
| **gfx1100/1101/1102** | yes | **yes** (same `dot.hpp` source, no WMMA gate) | separate fatbin each |
| **gfx1151** Strix Halo | yes (portable) | **yes** (same `dot.hpp`) | can run, not dest-tuned; not WMMA-gated |
| **gfx1031/1032/1033/1035/1036** Deck/mobile | yes (portable) | **yes** (RDNA2 DOT, **wave32**) | Steam Deck is gfx1033 / wave32. Not dest-tuned |
| **gfx1013** BC-250 | yes (portable) | **yes** (RDNA2 DOT stubs; not dest-tuned) | **RDNA2** Cyan Skillfish (same gen as V620/Deck, not Navi21). Can run, unoptimized. RADV: warp 64 — never force `-mwavefrontsize32` / `HSA_OVERRIDE`. Bind on **arch + wave**. |
| **gfx900** | yes stub / Later mad_mix | **no** | never load FA/EXL3 DOT |
| **gfx906** (real Vega20/MI50) | Later non-DOT if ever | **no** | **not** BC-250 |

**Separate fatbins, no shared objects.** DOT source may be shared; objects
are not. One configure tree → one `libhippihx_<arch>.a`. Never
`--offload-arch=gfx1030,gfx1100` in one artifact. **Never
`HSA_OVERRIDE_GFX_VERSION` / never load a foreign ISA** (especially never
run gfx1030 objects on gfx1013).

## Shared DOT source

FA (`attn/fa_fdot2`), EXL3 (`gemm/exl3_3inst`), AWQ/W4A16
(`gemm/w4a16_fdot2`), and `moe/shared` are **one source tree** compiled
once per `--offload-arch`:

```bash
# same tiles/*.hip, one --offload-arch per tree
cmake -S . -B build-gfx1030 -DHIPPIHX_ARCH=gfx1030
cmake -S . -B build-gfx1100 -DHIPPIHX_ARCH=gfx1100
# gfx1101 / gfx1102 first-class; gfx1151 / gfx103x (wave32) / gfx1013 portable
```

Craft locks: **wave32 only**, **no `fdot2.bf16`**, **never `#ifdef WMMA`**.
See `include/hippihx/dot.hpp`.

## Build fatbins

Requires ROCm **7.14** `hipcc` on the V620 box. This cloud/CI tree also
configures a **host compile stub** when `hipcc` is missing so the layout
stays buildable.

```bash
# DOT slots — same source, one tree per arch
cmake -S . -B build -DHIPPIHX_ARCH=gfx1030
cmake --build build
# archive: build/fatbin/gfx1030/libhippihx_gfx1030.a

cmake -S . -B build-gfx1100 -DHIPPIHX_ARCH=gfx1100
cmake -S . -B build-gfx1151 -DHIPPIHX_ARCH=gfx1151  # portable / unoptimized
cmake -S . -B build-gfx1033 -DHIPPIHX_ARCH=gfx1033  # Steam Deck, wave32
cmake -S . -B build-gfx1013 -DHIPPIHX_ARCH=gfx1013  # BC-250; never HSA_OVERRIDE
cmake -S . -B build-gfx900  -DHIPPIHX_ARCH=gfx900   # no DOT objects
```

Without ROCm (layout check only):

```bash
cmake -S . -B build -DHIPPIHX_FORCE_HOST_STUB=ON -DHIPPIHX_ARCH=gfx1030 \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build
./build/fatbin/gfx1030/hippihx_smoke_host
```

Use `g++` for the host stub if the default `c++` is clang without a
working `-lstdc++` (common on slim images). V620 builds use `hipcc`.

Raw `hipcc` (same policy: one arch per invocation):

```bash
hipcc --offload-arch=gfx1030 -std=c++17 -I include -c tiles/smoke.hip -o smoke.gfx1030.o
hipcc --offload-arch=gfx1100 -std=c++17 -I include -c tiles/smoke.hip -o smoke.gfx1100.o
```

Python package (protocol stubs, no torch required):

```bash
pip install -e ".[dev]"
pytest
```

## Tile layout

Directories are **classes**. See each group README: LDS and
`__launch_bounds__` will be locked per tile.

```
tiles/attn/fa_fdot2           # DOT (shared gfx1030+gfx1100)
tiles/attn/gdn_scan
tiles/attn/kda_scan
tiles/attn/qsa_indexer
tiles/attn/dsa_nope
tiles/gemm/w4a16_fdot2        # DOT (AWQ/GPTQ pack modes)
tiles/gemm/exl3_3inst         # DOT consume hook; produce stays outside
tiles/moe/routed              # gate / up / down
tiles/moe/shared              # DOT
tiles/moe/leftover_bf16
tiles/sequence/causal_conv    # scalar FMA, state_len≈4; not under gdn_scan
tiles/comm/pcie               # Uncached+push Later; INT8/Q8 wire; stub
```

## Packs stay outside

`3inst` / AWQ **produce** is not a hippihx directory. Packers, Viterbi, and
checkpoint rewrite live in their own tools. hippihx only documents the
consume layout and ships HIP that reads it.

## Non-goals

- Not a fork or copy of `local-inference-lab/b12x` CUDA / CuTe / CE / NVFP4.
- Not a vLLM tree. Not a Triton zoo. Not a produce/packer.
- No production GEMM or attention ISA in this skeleton (stubs exist so
  CMake is real).
- No multi-arch `.so`. No `#ifdef WMMA` on shared DOT tiles. No
  `fdot2.bf16` (gfx1030 LLVM abort). No DOT objects on gfx900 / gfx906.
  No HSA_OVERRIDE. DOT/GDN HIP refuse bf16 activations.
- BC-250 is **gfx1013** (Cyan Skillfish), **RDNA2** — same generation as
  V620/Deck, not gfx906 and not Steam Deck. Deck gfx1033 is wave32.
  gfx1013 builds unoptimized; do not force `-mwavefrontsize32`. Serve
  bind keys on **arch + wave size**.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fatbin policy, DOT
  source, bind rules, zoo vs serve.
- [`docs/BACKPORT.md`](docs/BACKPORT.md) — extras review: what is zoo vs
  serve, dest defects not to copy, why bodies are not copied yet.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ROCm pin, how to land a tile.
  Kernel / mode / env / AR tracker: **unvalidated** list below.

## Unvalidated extras inventory

Snapshot of [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` @ `6c5ff94efb8e` (2026-09-10 19:24 UTC; seven commits past
`a4060647cfbb`, which merged extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1))
plus open extras PRs **#2–#3**. Dest default branch is **`rdna_extras`**.
`main` is still upstream vLLM `c00091e02670` — no dest HIP there.

**Unvalidated.** Not dest. Not silicon-signed. No tok/s. hippihx still
ships stubs; bodies stay in extras until a consume bind exists. This
list is a tracker, not a claim that any row works.

Dest extras journals GDN FPP13 16k c=8 capture **green** via serve
arenas. Do **not** copy dest W4 scale-baked ZP, unaligned prefill
K-splits, `fdot2.bf16`, persist keepalive, or GDN HIP-on-BF16 dispatch.
V1 refuses bf16 on DOT and GDN HIP. Details:
[`docs/BACKPORT.md`](docs/BACKPORT.md#dest-extras-defects-do-not-copy).

Status key: **extras** = live on dest tip (unvalidated here) ·
**Later** = side branch / review-only · **skip** = do not take ·
**stub** = hippihx contract only.

### Kernels (HIP)

| Kernel / op | extras | hippihx tile | Notes |
|---|---|---|---|
| FA paged decode / prefill / split-K / short | `fa_rdna2.cu` | `attn/fa_fdot2` | `fdot2`. Occupancy pin closed. @ `6c5ff94`: GQA-256 NaN/idle-wave; persist O stay extras. **unvalidated** |
| FA INT8 KV writer | `reshape_and_cache_int8_rdna2` | `attn/fa_fdot2` | INT8 cache layout for RDNA_ATTN. **unvalidated** |
| FA fp16 flash KV writer | `reshape_and_cache_flash_rdna2` | `attn/fa_fdot2` | Stride-aware hybrid GDN pages. `__launch_bounds__(128, 4)`. fp16 only. **unvalidated** |
| W4A16 dense decode | `q_gemm_rdna2.cu` | `gemm/w4a16_fdot2` | GPTQ + AWQ = pack/zeros, one GEMM. Dest ZP is scale-baked `half` — zoo uses integer `q-zero` then scale. **unvalidated** |
| W4A16 prefill | `q_gemm_rdna2_prefill.cu` | `gemm/w4a16_fdot2` | Multi-config. Dest `compute_split_k` can pick K=640→16×40; zoo requires equal ×32. **unvalidated** |
| W4A16 AWQ high-M prefill | `q_gemm_rdna2_awq_prefill.cu` | `gemm/w4a16_fdot2` | Exllama-clone tile, AWQ zeros (no GPTQ +1). **unvalidated** |
| W4A16 MoE | `moe_q_gemm_rdna2.cu` | `moe/routed` | **unvalidated** |
| EXL3 dense / MoE / dequant / Hadamard / trellis decode | `exl3_dot2_*.cu` | `gemm/exl3_3inst` | Consume `-cb 3inst`. Produce outside. UNC-26. **unvalidated** |
| GDN packed decode | `gdn_decode_rdna2.cu` | `attn/gdn_scan` | Register-resident. Dest journals FPP13 16k c=8 green via **serve arenas**. **fp16 act only.** **unvalidated** |
| GDN prefill chain (prep / kkt / solve_wy / delta_h / o) | `gdn_prefill_*_rdna2.cu` | `attn/gdn_scan` | Opt-out (`VLLM_GDN_HIP_PREFILL=0`). `o` varlen `i_t_local` dest-fixed. Dispatch still misses dtype. **unvalidated** |
| causal_conv1d update + fwd | `causal_conv1d_rdna2.cu` | `sequence/causal_conv` | Scalar FMA, `state_len≈3–4`. FIR on pre-shift then shift. BF16: fp32-promoted mul, **not** `fdot2.bf16`. **unvalidated** |
| Paged MQA indexer | `indexer_paged_mqa_rdna2.cu` | `attn/qsa_indexer` | DeepSeek V4 Lightning class. gfx1030 BF16 6h×256 QSA prefill: **4 warps**. **unvalidated** |
| Sparse MLA decode / prefill | `sparse_mla_rdna2.cu` | `attn/dsa_nope` | **unvalidated** |
| W8A16 / W8A16-FP8 / W8A8-FP8 dense+MoE | `moe_w8a16*.cu`, `gemm_w8a8_fp8_dense_rdna2.cu`, `q_gemm_w8a16_fp8_rdna2.cu` | — | No hippihx tile yet. **unvalidated** |
| MXFP4 dense + MoE | `mxfp4_dot2_*.cu` | — | No hippihx tile yet. **unvalidated** |
| gfx1100 W4 WMMA | `q_gemm_rdna3_wmma.cu` | — | WMMA Later overlay, not shared DOT. **unvalidated** |
| Skinny GEMM / INT4 skinny | `skinny_gemms*.cu` | — | `VLLM_ROCM_USE_SKINNY_GEMM`. Dest extras has it; Flash-Next fork used BF16 decode. **No tile** until dest locks the family. Do not reintroduce leapdragon `gemv_f16`. **unvalidated** |
| RMSNorm HIP AOT | `layernorm.cu` | — | Serve fused-norm; no tile. **unvalidated** |
| GLM-5.3 KDA decode + prefill | `glm5_kda_*.cu` (PR **#2**) | `attn/kda_scan` | Later. Drop `glm5_` name. **unvalidated** |
| GLM-5.3 DSA indexer + MLA-NoPE | `glm5_dsa_*.cu` (PR **#2**) | `attn/dsa_nope`, `qsa_indexer` | Later. **unvalidated** |
| leapdragon push AR | `rdna_allreduce.{cu,cuh}` (merged PR **#1**) | `comm/pcie` | Dest extras, default **off**. Uncached+push. Occupancy pin closed. Aron Hsiao unique commits. **unvalidated** |
| a17t extra AWQ GEMM / GEMV | `awq_gemm_rdna2.cu`, `moe_awq_gemm_rdna2.cu`, `gemv_w4_kpack_rdna2.cu` (PR **#3**) | — | **skip** — second W4 family |

### Modes / dispatch

| Mode | What extras does | Default (extras) | hippihx |
|---|---|---|---|
| W4 decode vs prefill vs AWQ-prefill vs Exllama | M/K/N buckets in `rdna2_w4a16.py` | decode `M≤32` & `K≥4096`; AWQ high-M → `awq_prefill`; `M>256` Exllama (GPTQ) | one `w4a16_fdot2` tile |
| W4 pack | GPTQ `uint4b8` (+1 zeros) vs AWQ `uint4` (literal zeros) | same kernel, `use_v2_format` | pack/zeros, not a second GEMM |
| EXL3 codebook | `cb==0` 3inst produce, `cb==1` mcg compile, `cb==2` mul1 not produced | 3inst dest | consume only |
| EXL3 memory | `full` int16 trellis vs `packed` stub | `full` | — |
| EXL3 prefill | decode-trellis prefill vs fused GEMM | `VLLM_EXL3_PREFILL_DECODE=1` | — |
| GDN decode HIP | packed HIP vs Triton/FLA | on unless `VLLM_GDN_DECODE_RDNA2=0` | `gdn_scan` |
| GDN prefill HIP | 5-kernel chain | **opt-out** (`== "0"`) | `gdn_scan` |
| FA backend | `RDNA_ATTN` when gfx10x | `VLLM_USE_RDNA2_FA` (envs.py default false; backend reads `"1"`) | `fa_fdot2` |
| FA spec/MTP gate | opt-in abort on verify-shaped batches | **off** | serve, not a tile |
| MLA sparse HIP | indexer + sparse MLA | `VLLM_USE_RDNA2_MLA=1` | indexer / `dsa_nope` |
| causal conv HIP | update (decode) + fwd (prefill) | on unless set `0` | `causal_conv` |
| Custom AR (dest) | force custom all-reduce on PCIe-only | **off** | serve |
| leapdragon `rdna_ar` | size-gated Uncached+push | **off** (`VLLM_RDNA_AR=0`; dest extras after merged #1) | `comm/pcie` Later |

### Env (extras-added / extras-used)

Serve knobs. hippihx does not read these. **Unvalidated.** Debug probes stay extras (no D2H under capture in the zoo).

| Env | Default (as read in extras) | Role |
|---|---|---|
| `VLLM_USE_RDNA2_FA` | envs.py `False`; `rdna_attn` treats missing as `"1"` | FA-RDNA2 / `RDNA_ATTN` |
| `VLLM_USE_RDNA2_MLA` | off unless `"1"` | sparse MLA + paged MQA HIP |
| `VLLM_FARDNA2_ENABLE_SPEC_GATE` | `"0"` | MTP-verify abort (opt-in) |
| `VLLM_FARDNA2_SPEC_VERIFY_Q_LEN` | `"3"` | spec-gate q len |
| `VLLM_GDN_DECODE_RDNA2` | on (`!= "0"`) | GDN decode HIP |
| `VLLM_GDN_HIP_PREFILL` | off if `"0"` (opt-out; not a code default-off) | GDN prefill HIP chain |
| `VLLM_GDN_HIP_KERNELS` | recipe `1` | recipe umbrella; confirm vs the two gates above |
| `VLLM_GDN_DECODE_KERNEL` | `"cuda"` | FLA packed-decode path name (`cuda`/`triton`) |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` | `1` | FLA packed decode |
| `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` | `"1"` | conv update HIP |
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD` | `"1"` | conv fwd HIP |
| `VLLM_RDNA_FORCE_FP16` | recipe `1` | force fp16 (no BF16 emu) |
| `VLLM_EXL3_MEMORY_MODE` | `full` | `full` / `packed` |
| `VLLM_EXL3_PREFILL_DECODE` | `"1"` | trellis-decode prefill |
| `VLLM_EXL3_M_MAX` | `64` (`0` = no CG-PATH) | EXL3 capture cap |
| `VLLM_EXL3_DEQUANT_ALL` | off | dequant leftover / mul1 |
| `VLLM_EXL3_FOLDED_CACHE` | unset | folded-weight cache dir |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | skinny GEMM |
| `VLLM_ROCM_USE_AITER` | `False` | AITER (CDNA; not dest gfx1030) |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR` | `True` | AITER AR (CDNA) |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | `False` | force custom AR without full P2P |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | unset | `1stage` / `2stage` |
| `VLLM_ROCM_QUICK_REDUCE_*` | unset | ROCm quick-reduce size/quant knobs |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `1` | symmetric-memory AR |
| `VLLM_RDNA_AR` | `"0"` (dest extras; merged PR **#1**) | leapdragon push AR (opt-in; communicator gate, not the stale “enabled by default” docstring) |
| `VLLM_RDNA_AR_BLOCKS` | auto (dest extras; merged PR **#1**) | AR block cap |
| `VLLM_RDNA_AR_PACE` | `0` (dest extras; merged PR **#1**) | AR store pace |
| `VLLM_RDNA_AR_MAX_KB` | `512` (dest extras; merged PR **#1**) | AR fast-path size cap |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` (auto-on in some configs) | capture dispatcher |
| `VLLM_LOG_GDN_PTRS` | off | GDN pointer probe |
| `VLLM_GDN_DBG` | off | GDN debug print |
| `VLLM_EXL3_DEBUG` / `_HADAMARD_DBG` / `_APPLY_DBG` / `_MARKER_DBG` / `_INPUT_NAN_DBG` | off | EXL3 probes |
| `VLLM_CONV1D_DEBUG` / `VLLM_MLP_DBG` / `VLLM_RDNA2_MOE_DEBUG_NAN` | off | probes |
| `DBG_VLLM_STEP_TIMING` | off | per-step timing |

### AR / collectives

| Path | Where | Default | hippihx |
|---|---|---|---|
| RCCL | extras fallback | on when custom AR off | — |
| Custom all-reduce (vLLM/ROCm) | `VLLM_FORCE_CUSTOM_ALL_REDUCE` | **off** | serve |
| AITER custom AR | `VLLM_ROCM_USE_AITER_CUSTOM_AR` | on in envs, AITER itself off | CDNA, not gfx1030 dest |
| Quick-reduce | `VLLM_ROCM_QUICK_REDUCE_*` | unset | serve |
| Symm-mem AR | `VLLM_ALLREDUCE_USE_SYMM_MEM` | on | serve |
| leapdragon `rdna_ar` Uncached+push | dest extras (merged PR **#1** @ `a4060647`) | **off** | `comm/pcie` Later. Unique HIP: Aron Hsiao. Occupancy pin closed. Boot self-test. INT8/Q8 wire preferred; no Finegrained; no E4M3 without FP8 HW. Pick unique commits, not the dest squash. |

### Not taken / leave in extras

| Item | Why |
|---|---|
| a17t PR **#3** second AWQ GEMM + `qwen4_exp` | duplicate W4 family + serve |
| leapdragon `gemv_f16` / `moe_skinny_int4_decode` | dropped at extras PR #1 conflict resolution |
| Produce / pack (`-cb 3inst`, AWQ produce) | outside hippihx |
| Cudagraph `torch.zeros`, persist keepalive, `eager_break_during_capture`, GDN arenas | serve page-commit / dispatcher |
| extras PR **#5** Flash-Next draft | mixed other-fork vs dest extras |
| `rdna_extras_wip_20260910` TRUE FULL snapshot | not dest tip |
