# First-cut MAX registration for attention.fa_fdot2.
#
# This is an authoring stub, not a kernel body, not a fatbin, and not
# dest-ready. execute() does not enqueue. ABI gap: this registration is
# not hippihx_v1_plan / hippihx_v1_run. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists.
#
# Arch switch: gfx1030 (packed DOT, wave32, no WMMA gate) versus gfx900
# (mad_mix — do not ship the DOT object). Matches @extensibility.register
# device fields from the MAX custom-op surface.

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList

alias QUALNAME = "attention.fa_fdot2"
alias WAVE = 32
alias PRODUCE = False


fn _not_produce(arch: StaticString) raises -> None:
    raise Error(
        "attention.fa_fdot2 Mojo object is not dest-ready (arch=",
        arch,
        "); ABI gap versus hippihx_v1_*; dest produce stays hipcc 7.14 / "
        "hipModuleLoad / libamdhip64 until a documented HIP re-emit or "
        "MAX-serve soak exists",
    )


@extensibility.register(
    "hippihx.attention.fa_fdot2",
    type="gpu",
    api="hip",
    arch="gfx1030",
)
struct FaFdot2Gfx1030:
    """gfx1030 packed DOT authoring slot. wave32. No WMMA gate. No enqueue."""

    alias ARCH = "gfx1030"
    alias WAVE = WAVE
    alias PRODUCE = PRODUCE

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        _ = target
        _ = output
        _ = x
        _ = ctx
        _not_produce("gfx1030")


@extensibility.register(
    "hippihx.attention.fa_fdot2",
    type="gpu",
    api="hip",
    arch="gfx900",
)
struct FaFdot2MadMix:
    """gfx900 mad_mix. Do not ship the gfx1030 DOT object here."""

    alias ARCH = "gfx900"
    alias PRODUCE = PRODUCE

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        _ = target
        _ = output
        _ = x
        _ = ctx
        raise Error(
            "do not ship gfx1030 DOT objects onto gfx900 mad_mix; "
            "re-emit is still not this path"
        )


@extensibility.register_shape_function("hippihx.attention.fa_fdot2")
def fa_fdot2_shape(x: InputTensor) raises -> IndexList[x.rank]:
    _ = x
    raise Error(
        "attention.fa_fdot2 shape is not produce; "
        "re-emit HIP (hipcc 7.14 → hipModuleLoad / libamdhip64)"
    )
