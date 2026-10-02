# MAX authoring slot for comm.pcie.
#
# Not a kernel body, not a fatbin, and not dest-ready. execute() does not
# enqueue. ABI gap versus hippihx_v1_*. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists.
#
# Bind keys stay arch + wave + hop + switch on the host plan. Not a
# packed-DOT object, so there is no gfx900 mad_mix registration.

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList
from zoo.refuse import not_produce

alias QUALNAME = "comm.pcie"
alias WAVE = 32
alias PRODUCE = False


@extensibility.register(
    "hippihx.comm.pcie",
    type="gpu",
    api="hip",
    arch="gfx1030",
)
struct PcieGfx1030:
    """PCIe authoring slot. No enqueue."""

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


@extensibility.register_shape_function("hippihx.comm.pcie")
def pcie_shape(x: InputTensor) raises -> IndexList[x.rank]:
    _ = x
    not_produce(QUALNAME, "shape")
