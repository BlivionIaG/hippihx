# Mojo/MAX authoring registry.
#
# Importing this package is the authoring center: every catalog op that has
# a brief registers here. execute() refuses to enqueue. Nothing in this
# package is a fatbin input or a vllm-rdna produce object.
#
# Left without a registration (catalog stubs only):
#   attention.kda_scan
#   attention.dsa_nope

from zoo.causal_conv import CausalConvGfx1030
from zoo.exl3_3inst import Exl3InstGfx1030, Exl3InstMadMix
from zoo.fa_fdot2 import FaFdot2Gfx1030, FaFdot2MadMix
from zoo.families import (
    HOOK_HYBRID_HEAP,
    HOOK_HYBRID_HEAP_CALLER_OWNED,
    HOOK_MOE_LEFTOVER_BF16,
    HOOK_MOE_ROUTED,
    HOOK_QWEN_GDN,
    HOOK_QWEN_PLE,
    HOOK_QWEN_QSA,
    LEFT_WITHOUT_NEW_BRIEF,
)
from zoo.gdn_scan import GdnScanGfx1030
from zoo.leftover_bf16 import LeftoverBf16Gfx1030
from zoo.pcie import PcieGfx1030
from zoo.qsa_indexer import QsaIndexerGfx1030
from zoo.routed import RoutedGfx1030
from zoo.shared import SharedGfx1030, SharedMadMix
from zoo.w4a16_fdot2 import W4A16Fdot2Gfx1030, W4A16Fdot2MadMix
