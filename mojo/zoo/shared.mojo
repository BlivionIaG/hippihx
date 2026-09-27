# MAX authoring slot for moe.shared.
#
# Not a kernel body, not a fatbin, and not dest-ready. execute() does not
# enqueue. ABI gap versus hippihx_v1_*. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists.
#
# Shared DOT source with gfx1030. wave32. No WMMA gate. gfx900 mad_mix
# does not receive the DOT object.

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList
from zoo.refuse import not_produce, refuse_dot_on_gfx900

alias QUALNAME = "moe.shared"
alias WAVE = 32
alias PRODUCE = False


@extensibility.register(
    "hippihx.moe.shared",
    type="gpu",
    api="hip",
    arch="gfx1030",
)
struct SharedGfx1030:
    """gfx1030 shared-expert DOT slot. No enqueue."""

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
        not_produce(QUALNAME, "gfx1030")


@extensibility.register(
    "hippihx.moe.shared",
    type="gpu",
    api="hip",
    arch="gfx900",
)
struct SharedMadMix:
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
        refuse_dot_on_gfx900(QUALNAME)


@extensibility.register_shape_function("hippihx.moe.shared")
def shared_shape(x: InputTensor) raises -> IndexList[x.rank]:
    _ = x
    not_produce(QUALNAME, "shape")
