#include "hippihx/tile_stub.hpp"

// Contract: moe/leftover_bf16 — leftover dense BF16 (unquantized / cvt→fdot2).
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_moe_leftover_bf16_stub() {}
