#include "hippihx/tile_stub.hpp"

// Contract: attn/qsa_indexer — QSA / group-select indexer class.
// gfx1030 BF16 6h×256 prefill occupancy: 4 warps (2-warp spills).
// Dest Flash-Next abort is Triton forward_qsa (not this tile profile).
// qsa_rdna2.cu is extras opt-in scaffolding (default off). Not V4 Lightning.
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_attn_qsa_indexer_stub() {}
