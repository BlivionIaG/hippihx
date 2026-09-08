#include "hippihx/dot.hpp"

// Shared DOT source (gfx1030 + gfx1100, two fatbins). Never #ifdef WMMA.
// AWQ/GPTQ are pack modes of this tile. wave32, no fdot2.bf16.
// Produce stays outside hippihx.
__global__ void hippihx_gemm_w4a16_fdot2_stub() {}
