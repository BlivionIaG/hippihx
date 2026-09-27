"""Family bind hooks and the tensor views those hooks carry.

Qwen QSA / GDN / PLE, leftover BF16 versus routed experts, and hybrid
state heaps stay addressable from the bind surface. A Mojo lift keeps
these hooks and their tensor contracts. GLM KDA (``attention.kda_scan``)
and the DeepSeek transplant (``attention.dsa_nope``) stay catalog stubs:
no new family hook and no new brief. extras product gates stay in serve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorContract:
    """A bind view the family must still accept after a Mojo lift.

    ``dtype`` is the tile-README lock (``fp16``, ``bf16``, ``fp16|fp32``,
    ``fp16|bf16``) or ``unpinned`` when the README has not named a dtype.
    """

    name: str
    dtype: str
    note: str


@dataclass(frozen=True, slots=True)
class FamilyHook:
    """A bind-time family name. ``qualnames`` are catalog ops, not aliases."""

    name: str
    qualnames: tuple[str, ...]
    summary: str
    bind_keys: tuple[str, ...]
    tensors: tuple[TensorContract, ...]


def _t(name: str, dtype: str, note: str) -> TensorContract:
    return TensorContract(name=name, dtype=dtype, note=note)


HOOKS: tuple[FamilyHook, ...] = (
    FamilyHook(
        name="qwen.qsa",
        qualnames=("attention.qsa_indexer",),
        summary=(
            "Qwen QSA / group-select indexer. Not V4 Lightning. "
            "gfx1030 occupancy gate is 4 warps (128 threads); a 2-warp "
            "profile spills. Exact attention over the selected set is a "
            "different launch (fa_fdot2). This is the Qwen contract. "
            "The DeepSeek-class indexer transplant is not a new brief."
        ),
        bind_keys=("arch", "wave", "groups"),
        tensors=(
            _t(
                "groups",
                "unpinned",
                "Group selection only. q/k/v of the selected set stay on "
                "attention.fa_fdot2.",
            ),
        ),
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
        tensors=(
            _t("mixed_qkv", "fp16", "Activation. V1 refuses bf16."),
            _t("a", "fp16", "Activation."),
            _t("b", "fp16", "Activation."),
            _t("out", "fp16", "Activation."),
            _t(
                "state",
                "fp16|fp32",
                "Load/store. Recurrence stays fp32. Invalid slot does not "
                "touch state.",
            ),
        ),
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
        tensors=(
            _t(
                "input",
                "fp16|bf16",
                "Scalar FMA in. fp32 mul. Never fdot2. Not a KDA scan.",
            ),
            _t("out", "fp16|bf16", "Scalar FMA out. fp32 mul."),
            _t(
                "state",
                "fp16|bf16",
                "state_len 3 or 4. LDS 0. Lives in hybrid.heap when the "
                "model is hybrid.",
            ),
        ),
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
        tensors=(
            _t(
                "gate",
                "fp16",
                "Routed projection. fp16 act on the W4 family. "
                "expert_path=routed.",
            ),
            _t("up", "fp16", "Routed projection. Same W4 family as gate."),
            _t("down", "fp16", "Routed projection. Same W4 family as gate."),
        ),
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
        tensors=(
            _t(
                "dense",
                "bf16",
                "Unquantized leftover. cvt to fp16 then fdot2, or scalar "
                "FMA. Never fdot2.bf16. expert_path=leftover_bf16. Not "
                "moe.routed gate/up/down.",
            ),
        ),
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
        tensors=(
            _t(
                "heap",
                "fp16|fp32",
                "Caller-owned. Holds qwen.gdn state and qwen.ple state. "
                "plan sizes it; serve zeros it once; bind only views.",
            ),
        ),
    ),
)

# Catalog ops that stay as they are. No new family hook in this conversion.
LEFT_WITHOUT_NEW_BRIEF: tuple[str, ...] = (
    "attention.kda_scan",
    "attention.dsa_nope",
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
