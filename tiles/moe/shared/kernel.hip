#include "hippihx/dot.hpp"

// Shared DOT source (gfx1030 + gfx1100, two fatbins). Never #ifdef WMMA.
// Shared-expert consume; same DOT unit as W4A16 / EXL3. wave32, no fdot2.bf16.
__global__ void hippihx_moe_shared_stub() {}
