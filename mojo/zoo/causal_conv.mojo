# MAX authoring slot for sequence.causal_conv (qwen.ple).
#
# Not a kernel body, not a fatbin, and not dest-ready. execute() does not
# enqueue. ABI gap versus hippihx_v1_*. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists.
#
# Scalar FMA. state_len 3 or 4. fp32 mul. Not a GDN or KDA scan. State
# lives in hybrid.heap when the model is hybrid; that heap is caller-owned.
# Not a packed-DOT object, so there is no gfx900 mad_mix registration.

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList
from zoo.refuse import not_produce

alias QUALNAME = "sequence.causal_conv"
alias WAVE = 32
alias PRODUCE = False


@extensibility.register(
    "hippihx.sequence.causal_conv",
    type="gpu",
    api="hip",
    arch="gfx1030",
)
struct CausalConvGfx1030:
    """qwen.ple authoring slot. Scalar FMA. No enqueue."""

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


@extensibility.register_shape_function("hippihx.sequence.causal_conv")
def causal_conv_shape(x: InputTensor) raises -> IndexList[x.rank]:
    _ = x
    not_produce(QUALNAME, "shape")
