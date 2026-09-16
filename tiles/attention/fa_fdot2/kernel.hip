#include "hippihx/dot.hpp"

// Shared DOT source (gfx1030 + gfx1100, two fatbins). Never #ifdef WMMA.
// wave32, fdot2 / v_dot2c, no fdot2.bf16.
// LDS / __launch_bounds__ locked in README.md at migrate.
__global__ void hippihx_attention_fa_fdot2_stub() {}
