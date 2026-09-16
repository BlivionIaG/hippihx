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

## Package layout

Same *shape* as b12x (`<group>.<op>` + `api.py`), HIP objects:

| Path | Owns |
|---|---|
| `hippihx/_lib/catalog.py` | One op table (qualname, V1 id, DOT, tile path) |
| `hippihx/<group>/<op>/api.py` | `plan` / `bind` / `run` |
| `tiles/<group>/<op>/kernel.hip` | HIP ISA (dest). Torch-free |
| `include/hippihx/v1.h` | C consume ABI extras wraps as `torch.ops` |
| `hippihx/flydsl/` | FlyDSL compiler atoms + kernel contracts |
| `hippihx/comm/fabric.py` | PCIe/PLX hop class (PIX/PXB/PHB × 88096/8749) |

Group rename **`attn` → `attention`** (V1 ABI rev **3**; ids unchanged). ISA
class names stay (`fa_fdot2`, not b12x `paged`). Map:

| hippihx | b12x analogue |
|---|---|
| `attention.fa_fdot2` | `attention.paged` |
| `attention.gdn_scan` | `sequence.gdn_decode` |
| `attention.kda_scan` | `sequence.kda_prefill` |
| `attention.qsa_indexer` | `attention.dsa_indexer` |
| `attention.dsa_nope` | `attention.sparse_mla` |
| `gemm.w4a16_fdot2` | W4 class (b12x folds W4 into MoE) |
| `gemm.exl3_3inst` | `gemm.trellis_linear` |
| `sequence.causal_conv` | `sequence.ple` (short conv) |
| `comm.pcie` | `comm.pcie` |

Do not dual-export aliases. Do not import b12x.

## HIP fatbins / FlyDSL compiler

HIP fatbins (`tiles/`) are the extras V1 consume path. FlyDSL
(`hippihx.flydsl`) is a dest **compiler** backend: `Caps(backend="flydsl")`
is valid. Do not port ROCm/FlyDSL MFMA/WMMA GEMM/MoE/FA into `tiles/` or
`hippihx.flydsl`. FlyDSL is an optional extra, not a required dependency.
extras FlyDSL consume waits on `FLYDSL_V1_CONSUME` (graph-safe JIT). See
[`FLYDSL.md`](FLYDSL.md).


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
| One source | `tiles/attention/fa_fdot2`, `tiles/gemm/w4a16_fdot2`, `tiles/gemm/exl3_3inst`, `tiles/moe/shared` |
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
| **gfx1013** BC-250 | Later | **no** | Cyan Skillfish — **not true RDNA2**, not dest, not portable DOT. Not gfx906. Never `HSA_OVERRIDE` dest ISA onto it. |
| **gfx900** | yes stub / Later mad_mix | **no** | never load FA/EXL3 DOT |
| **gfx906** (real Vega20/MI50) | Later non-DOT if ever | **no** | **not** BC-250 |

One configure tree → one `libhippihx_<arch>.a` in `build/fatbin/<arch>/`.
CMake rejects multi-arch lists and refuses Later slots (**gfx1013**,
**gfx906**). Portable DOT slots (1151 / Deck 103x) configure and compile
the same stubs.

**BC-250 is Later.** It is Cyan Skillfish **`gfx1013`** — not true RDNA2,
not dest, not a portable DOT fatbin with gfx1030. It is **not**
Vega20/`gfx906`. Never `HSA_OVERRIDE` a gfx1030 / gfx1033 object onto
it. Own dest work stays gfx1030 / Deck gfx103x / gfx110x.

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
   GFX name alone. gfx1030 and Deck gfx1033 are wave32 RDNA2. gfx1013
   is Later (not true RDNA2) — never load a dest fatbin on it. **Comm
   also keys on hop + switch** (`pix` /
   `pxb` / `phb` × `pex88096` / `pex8749` / `generic`). Do not load a PIX
   AR on a PHB job.
7. **Activations: fp16 for DOT and GDN HIP.** Never `fdot2.bf16`
   (`llvm.amdgcn.fdot2.bf16.bf16` aborts on gfx1030). V1 returns
   `HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE`. BF16 leftover / causal conv use
   scalar FMA with fp32 mul. W4 consume is integer ZP then scale; prefill
   K-splits are equal multiples of 32.

## Tile classes

Directories are classes, not SKUs:

- `attention/fa_fdot2` (DOT), `attention/gdn_scan`, `attention/kda_scan`,
  `attention/qsa_indexer`, `attention/dsa_nope`
- `gemm/w4a16_fdot2` (DOT), `gemm/exl3_3inst` (DOT consume hook)
- `moe/routed` (gate/up/down), `moe/shared` (DOT), `moe/leftover_bf16`
- `sequence/causal_conv` — scalar FMA, `state_len≈4`; **not** under
  `gdn_scan`. GDN vs KDA layouts differ (do not retarget GDN 16/48 onto
  KDA 64×128).
- `comm/pcie` — Uncached+push; INT8/Q8 wire. Plan keys on
  `Fabric(hop, switch)`: PIX+ACS on **88096** may custom-AR; PHB/PXB /
  8749 stay RCCL. Leave E4M3 / NTB / switch DMA.

Each tile README will lock **LDS** and **`__launch_bounds__`** before ISA
lands. Occupancy notes (VGPR vs `waves_per_eu`) belong there too.

### FA LDS pins (before extras migrate)

Recorded on `tiles/attention/fa_fdot2/README.md`. Fill numbers at migrate; do
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
slot (gfx1030, gfx110x, gfx1151, gfx103x). The skeleton is
host-side and torch-free.

## C consume ABI (V1)

`include/hippihx/v1.h` is the stable entry family extras will load from
the fatbin:

| Symbol | Role |
|---|---|
| `hippihx_v1_plan` | host-only scratch specs (always `zeroed=1`) |
| `hippihx_v1_run` | capture-safe enqueue; stub returns `NOT_READY` until migrate |
| `hippihx_v1_op_name` / `_is_dot` / `_fp16_act` | id ↔ qualname / DOT / fp16-act flags |

One V1 id per tile (`HIPPIHX_V1_OP_*`). Serve wraps as
`torch.ops.hippihx.<op>` — never a second Triton path in this library.
Python mirror: `hippihx.v1` (`V1OpId`, `ABI_REVISION`). Caps include
optional activation `dtype` (revision **2**). Qualnames `attention.*`
(revision **3**; ids unchanged). Dest extras tip
`8960a3bcbb17` still has no `torch.ops.hippihx.*` rewire. Persist
keepalive / GDN arenas / Flash-Next `qwen4_exp` / V1 FULL→PIECEWISE /
vLLM custom AR stay extras. Dest deleted the second W4 prefill `.cu`
and reverted ConfigH. GDN decode HIP now accepts fp16 SSM state
(`02adbfd4`) — observe, do not dump. Do not edit
`opengfx1030/vllm-rdna` from this tree to rewire; that lands in extras.

## Non-goals (room lock)

- Importing or forking b12x CUDA / CuTe / CE / NVFP4 / WMMA sources
- Porting ROCm/FlyDSL MFMA/WMMA GEMM/MoE/FA (see [`FLYDSL.md`](FLYDSL.md))
- A serve stack, model registry, or vLLM plugin inside this repo
- PRs against upstream vLLM
- Editing `opengfx1030/vllm-rdna` from this tree
- Dumping extras `csrc/rocm/*.cu` here before extras can consume hippihx
  (see [`BACKPORT.md`](BACKPORT.md))
- Re-authoring dest extras HIP as Cursor commits (dest Author **and**
  Committer is BlivionIaG). Re-authoring foreign HIP (leapdragon / a17t)
  as Cursor **or** Blivion (cherry-pick `-x`, keep **their** Author and
  Committer)
