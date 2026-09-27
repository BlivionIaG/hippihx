"""Family bind hooks. Documented on ``bind``, not new V1 ops.

Qwen QSA / GDN / PLE, leftover BF16 versus routed experts, and hybrid
state heaps stay addressable from the bind surface. extras product gates
(``VLLM_RDNA_QSA_HIP``, PLE HIP, hybrid capture arenas) stay in serve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FamilyHook:
    """A bind-time family name. ``qualnames`` are catalog ops, not aliases."""

    name: str
    qualnames: tuple[str, ...]
    summary: str
    bind_keys: tuple[str, ...]


HOOKS: tuple[FamilyHook, ...] = (
    FamilyHook(
        name="qwen.qsa",
        qualnames=("attention.qsa_indexer",),
        summary=(
            "Qwen QSA / group-select indexer. Not V4 Lightning. "
            "gfx1030 occupancy gate is 4 warps (128 threads); a 2-warp "
            "profile spills. Exact attention over the selected set is a "
            "different launch (fa_fdot2)."
        ),
        bind_keys=("arch", "wave", "groups"),
    ),
    FamilyHook(
        name="qwen.gdn",
        qualnames=("attention.gdn_scan",),
        summary=(
            "Qwen GDN / hybrid-state scan. Activations stay fp16. "
            "State load/store may be fp16 or fp32; recurrence stays fp32. "
            "Invalid slot zeros the output and does not touch state."
        ),
        bind_keys=("arch", "wave", "state_dtype"),
    ),
    FamilyHook(
        name="qwen.ple",
        qualnames=("sequence.causal_conv",),
        summary=(
            "Qwen PLE short conv is the sequence.causal_conv contract: "
            "scalar FMA, state_len 3 or 4, not a GDN or KDA scan. "
            "extras PLE HIP stays extras until a body migrates."
        ),
        bind_keys=("arch", "wave", "state_len"),
    ),
    FamilyHook(
        name="moe.routed",
        qualnames=("moe.routed",),
        summary=(
            "Routed expert gate/up/down. W4 consume stays the one "
            "gemm.w4a16_fdot2 family (integer q - zero, then scale). "
            "Not a second AWQ GEMM and not the leftover-BF16 path."
        ),
        bind_keys=("arch", "wave", "expert_path"),
    ),
    FamilyHook(
        name="moe.leftover_bf16",
        qualnames=("moe.leftover_bf16",),
        summary=(
            "Unquantized leftover dense BF16 (shared experts, heads). "
            "Accepts bf16 caps. GEMM leftovers cvt to fp16 then fdot2; "
            "otherwise scalar FMA. Never fdot2.bf16. Distinct from moe.routed."
        ),
        bind_keys=("arch", "wave", "expert_path"),
    ),
    FamilyHook(
        name="hybrid.heap",
        qualnames=("attention.gdn_scan", "sequence.causal_conv"),
        summary=(
            "Caller-owned hybrid state heap for GDN and the PLE/conv "
            "state. plan sizes it; serve zeros it once; bind only views. "
            "Do not wipe the heap after prefill has written state."
        ),
        bind_keys=("arch", "wave", "heap"),
    ),
)


def hooks_for(qualname: str) -> tuple[FamilyHook, ...]:
    return tuple(hook for hook in HOOKS if qualname in hook.qualnames)


def require_family(qualname: str, name: str) -> FamilyHook:
    for hook in hooks_for(qualname):
        if hook.name == name:
            return hook
    known = tuple(hook.name for hook in hooks_for(qualname))
    raise ValueError(
        f"family {name!r} is not a bind hook for {qualname}; known {known}"
    )
