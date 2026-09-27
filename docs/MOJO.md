# Mojo/MAX authoring

hippihx is moving its **authoring** surface to Mojo/MAX so tile contracts
stay in one language that can specialize on `arch` without a pile of
`#ifdef`s. The durable asset is still the ISA contract
([`ISA.md`](ISA.md)). Emit language is a backend.

This is a maintainability conversion. It is not a produce-pin change.

## Not produce

`Caps(backend="mojo")` may **plan** and **bind**. `run` raises
`MojoNotProduce`.

A Mojo object is not a HIP fatbin. `hipModuleLoad` / `libamdhip64` will
not load it. extras V1 consume stays the HIP archive (`hippihx_v1_*`,
ROCm **7.14** `hipcc`, one `--offload-arch`).

Sandbox or Mojo winners soak under vllm-rdna only after one of these is
true:

1. **Re-emit HIP.** The winning schedule is lowered again with `hipcc`
   7.14 into `build/fatbin/<arch>/libhippihx_<arch>.a`, then serve loads
   it with `hipModuleLoad`. The Mojo file is the source of the schedule,
   not the object that ran.
2. **An explicit MAX serve path** is chosen in serve (a MAX
   `InferenceSession` / custom op, not a silent swap behind
   `hippihx_v1_run`). That choice is not this repo and not the default.

`MOJO_AUTHORING` is true. `MOJO_PRODUCE` and `MOJO_V1_CONSUME` are false.
`mojo/` is not in the CMake fatbin. Do not add `max` as a required
dependency. Host tests read the Mojo sources as text; they do not invoke
the Mojo compiler.

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

## Family bind hooks

These are not new V1 ids. They are names on the bind surface so Qwen and
MoE dispatch stay documentable while bodies are still stubs.

| Hook | Catalog op | Bind keys |
|---|---|---|
| `qwen.qsa` | `attention.qsa_indexer` | arch, wave, groups |
| `qwen.gdn` | `attention.gdn_scan` | arch, wave, state_dtype |
| `qwen.ple` | `sequence.causal_conv` | arch, wave, state_len |
| `moe.routed` | `moe.routed` | arch, wave, expert_path |
| `moe.leftover_bf16` | `moe.leftover_bf16` | arch, wave, expert_path |
| `hybrid.heap` | `gdn_scan` and `causal_conv` | arch, wave, heap |

`expert_path` distinguishes routed experts from leftover BF16. Leftover
accepts bf16 and must not emit `fdot2.bf16`. Routed stays on the one W4
family. `hybrid.heap` is the caller-owned GDN + PLE/conv state: `plan`
sizes it, serve zeros it once, `bind` only views it. Do not wipe it after
prefill. extras PLE / QSA HIP gates stay extras.

Each op module exports `FAMILY_HOOKS` next to `bind`.

## Next authoring steps

1. Register the remaining catalog ops the same way as `fa_fdot2`: one
   gfx1030 struct that refuses enqueue, and a mad_mix struct only where
   a DOT object must fail closed. No kernel body in that step.
2. Move one family at a time (QSA, GDN, PLE, then routed versus leftover,
   then the hybrid heap keys) onto real Mojo schedules. Occupancy and LDS
   pins stay in the tile README. Spill or a bad ISel is a drop.
3. Re-emit HIP for any winner before `hippihx_v1_run` returns anything
   other than `NOT_READY`. A MAX serve path, if it is ever chosen, is a
   serve change in `rdna_extras`, not a fatbin of `.mojo` files.

Do not edit `opengfx1030/vllm-rdna` from this tree.
