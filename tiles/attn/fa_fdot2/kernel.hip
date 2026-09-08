#include "hippihx/tile_stub.hpp"

// Contract: attn/fa_fdot2 — flash-attn class using fdot2 (v_dot2c) on gfx1030.
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_attn_fa_fdot2_stub() {}
