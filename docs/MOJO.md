# Mojo/MAX authoring

hippihx is moving its **authoring** surface to Mojo/MAX so tile contracts
stay in one language that can specialize on `arch` without a pile of
`#ifdef`s. The durable asset is still the ISA contract
([`ISA.md`](ISA.md)). Emit language is a backend.

This is a maintainability conversion. It is not a produce-pin change.
Mojo objects are **not dest-ready**.

## Dest produce ABI

Dest produce for vllm-rdna stays **hipcc 7.14 / `hipModuleLoad` /
`libamdhip64`** until a documented HIP re-emit soak or a documented
MAX-serve soak exists. Neither soak exists in this tree. Do not edit
`opengfx1030/vllm-rdna` or `rdna_extras` to invent one.

| | Dest produce (now) | Mojo authoring (this tree) |
|---|---|---|
| Compiler | ROCm **7.14** `hipcc`, one `--offload-arch` | Mojo/MAX, not invoked by host tests |
| Loader | `hipModuleLoad` from `libamdhip64` | MAX `custom_extensions` / `InferenceSession` |
| Symbols | `hippihx_v1_plan` / `hippihx_v1_run`, ABI rev **4** | `@extensibility.register` `execute(OutputTensor, InputTensor, DeviceContext)` |
| Object | `libhippihx_<arch>.a` | not an AMDGPU code object from `hipcc` |

**ABI gap.** A Mojo `execute` registration does not export `hippihx_v1_*`
and is not a module `hipModuleLoad` can open. `Caps(backend="mojo")` may
**plan** and **bind**. `run` raises `MojoNotProduce`. `MOJO_AUTHORING` is
true. `MOJO_PRODUCE`, `MOJO_V1_CONSUME`, and `MOJO_DEST_READY` are false.
`plan.meta["produce"]` is false on this backend. `mojo/` is not in the
CMake fatbin. Do not add `max` as a required dependency.

Rev 4 gives `hippihx_v1_run` a tensor list in catalog slot order, the same
kind of list a MAX `execute` takes. A Mojo registration should take those
slots in that order. `FaFdot2Gfx1030` still takes a single input, and
lifting it is an authoring step. The object and loader gap is unchanged.

A later soak has to be one of these, written down as a soak, not assumed
from a green authoring test:

1. **HIP re-emit.** The winning schedule is lowered again with `hipcc`
   7.14 into `build/fatbin/<arch>/libhippihx_<arch>.a`, exporting the V1
   symbols, then loaded with `hipModuleLoad`. The Mojo file remains the
   schedule source. The object that ran is the HIP object.
2. **MAX-serve soak.** Serve runs a MAX `InferenceSession` / custom op on
   purpose. That is not a silent swap behind `hippihx_v1_run`.

Host tests read the Mojo sources as text. They do not invoke the Mojo
compiler and they are not that soak.

FlyDSL stays a dest compiler backend ([`FLYDSL.md`](FLYDSL.md)). It is not
replaced by this scaffold. Do not port Modular CDNA MFMA attention
(gfx942 MHA and the same class) or FlyDSL WMMA/MFMA GEMM into `mojo/` or
`tiles/`.

## Layout

```
hippihx/isa.py                 # Python ISA API
hippihx/mojo/                  # authoring flags, explore, family hooks
include/hippihx/isa.hpp        # C++ mirror
mojo/                          # MAX custom_extensions package
  isa/contracts.mojo           # arch switch, packed DOT, LDS, wave32
  zoo/fa_fdot2.mojo            # first registration; execute() refuses
```

`mojo/` is a MAX package (`__init__.mojo` present) so a later graph can
pass it as `custom_extensions`. The first op is
`hippihx.attention.fa_fdot2`:

| Registration | `arch` | What `execute` does |
|---|---|---|
| `FaFdot2Gfx1030` | `gfx1030` | raises not-produce. wave32. no WMMA gate. no enqueue |
| `FaFdot2MadMix` | `gfx900` | raises mad_mix. the DOT object does not ship here |

Device fields match current MAX `@extensibility.register`
(`type="gpu"`, `api="hip"`, `arch=`). gfx906 and gfx1013 are Later and
are not registered; `refuse_dot_on_mad_mix` still rejects them.

## Plan / bind / run

Same verbs as the HIP and FlyDSL paths.

```python
from hippihx.attention import fa_fdot2, gdn_scan
from hippihx.protocol import Caps

caps = Caps(arch="gfx1030", wave=32, backend="mojo")
plan = fa_fdot2.plan(caps)
plan.meta["arch_switch"]   # "dot"
plan.meta["explore"]       # ("decode", "prefill") — rank order, no tok/s
plan.meta["produce"]       # False

gdn = gdn_scan.plan(Caps(arch="gfx1030"))
gdn.meta["families"]       # ("qwen.gdn", "hybrid.heap")
gdn_scan.bind(gdn, family="hybrid.heap")  # views only; no heap alloc
```

`bind` never allocates. `family=` is optional and must be a hook for that
op. `run` on a Mojo plan does not enqueue and does not D2H.

Ranked explore shapes live in `hippihx.mojo.ranked_explore`. Rank 0 is
the first candidate. Spill rows (QSA 2-warp) stay in the list at a worse
rank. DOT explore on gfx900/906/1013 is empty.

## Family binds and tensor contracts

A Mojo lift keeps these hooks and the tensor views on them. They are not
new V1 ids. Each op module exports `FAMILY_HOOKS` next to `bind`.
`FamilyHook.tensors` is the view list.

| Hook | Catalog op | Bind keys | Tensor views |
|---|---|---|---|
| `qwen.qsa` | `attention.qsa_indexer` | arch, wave, groups | `groups` (selection). q/k/v of the selected set stay on `fa_fdot2` |
| `qwen.gdn` | `attention.gdn_scan` | arch, wave, state_dtype | `mixed_qkv`, `a`, `b`, `out` (fp16); `state` (fp16 or fp32) |
| `qwen.ple` | `sequence.causal_conv` | arch, wave, state_len | `input`, `out`, `state` (fp16 or bf16, fp32 mul, state_len 3 or 4) |
| `moe.routed` | `moe.routed` | arch, wave, expert_path | `gate`, `up`, `down` (fp16 act on the one W4 family) |
| `moe.leftover_bf16` | `moe.leftover_bf16` | arch, wave, expert_path | `dense` (bf16). Never `fdot2.bf16` |
| `hybrid.heap` | `gdn_scan` and `causal_conv` | arch, wave, heap | `heap` (caller-owned GDN state + PLE/conv state) |

`expert_path` keeps routed experts off the leftover-BF16 tensor. Leftover
accepts bf16 caps. Routed stays on the one W4 family (integer `q - zero`,
then scale). `hybrid.heap` is sized by `plan`, zeroed once by serve, and
only viewed by `bind`. Do not wipe it after prefill. extras PLE / QSA HIP
gates stay extras.

`qwen.qsa` is the Qwen group-select contract. It does not absorb the
DeepSeek-class indexer.

## Left without a new brief

`attention.kda_scan` (GLM KDA) and `attention.dsa_nope` (DeepSeek / sparse
MLA transplant) stay the catalog stubs they already are.
`LEFT_WITHOUT_NEW_BRIEF` is that pair. This conversion does not add a
family hook, a tensor schema, a Mojo registration, or a transplant plan
for them. Existing tile READMEs still apply. Do not retarget GDN 16/48
onto KDA 64×128.

## Next authoring steps

1. Register the remaining catalog ops the same way as `fa_fdot2`: one
   gfx1030 struct that refuses enqueue, and a mad_mix struct only where
   a DOT object must fail closed. No kernel body in that step.
2. Lift one family at a time (QSA, GDN, PLE, then routed versus leftover,
   then the hybrid heap). Each lift keeps that hook's tensor contracts.
   Occupancy and LDS pins stay in the tile README. Spill or a bad ISel is
   a drop. Do not open a GLM KDA or DeepSeek brief in that step.
3. Dest produce stays hipcc 7.14 / `hipModuleLoad` / `libamdhip64` until
   a documented HIP re-emit soak or a documented MAX-serve soak exists.
   `hippihx_v1_run` stays `NOT_READY` until the HIP object is that soak.
   A MAX-serve soak is a serve change, not a fatbin of `.mojo` files, and
   it is not done from this tree.

Do not edit `opengfx1030/vllm-rdna` or `rdna_extras` from this tree.
