# MAX authoring slot for attention.qsa_indexer (qwen.qsa).
#
# Not a kernel body, not a fatbin, and not dest-ready. execute() does not
# enqueue. ABI gap versus hippihx_v1_*. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists.
#
# Group selection only. q/k/v of the selected set stay on fa_fdot2.
# gfx1030 occupancy gate is 4 warps / 128 threads. Not a DOT object.
# The DeepSeek transplant (attention.dsa_nope) is not registered here.

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList
from zoo.refuse import not_produce

alias QUALNAME = "attention.qsa_indexer"
alias WAVE = 32
alias PRODUCE = False


@extensibility.register(
    "hippihx.attention.qsa_indexer",
    type="gpu",
    api="hip",
    arch="gfx1030",
)
struct QsaIndexerGfx1030:
    """qwen.qsa authoring slot. groups view. No enqueue."""

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


@extensibility.register_shape_function("hippihx.attention.qsa_indexer")
def qsa_indexer_shape(x: InputTensor) raises -> IndexList[x.rank]:
    _ = x
    not_produce(QUALNAME, "shape")
