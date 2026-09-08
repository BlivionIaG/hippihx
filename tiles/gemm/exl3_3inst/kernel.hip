#include "hippihx/dot.hpp"

// Shared DOT source (gfx1030 + gfx1100, two fatbins). Never #ifdef WMMA.
// EXL3 consume hook: 3inst → half2 → fdot2. No fdot2.bf16. wave32.
// Produce (`-cb 3inst`) is not a hippihx directory.
__global__ void hippihx_gemm_exl3_3inst_stub() {}
