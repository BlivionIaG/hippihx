#include "hippihx/tile_stub.hpp"

// Contract: gemm/w4a16_fdot2 — W4A16 nibble+ZP via fdot2. GPTQ/AWQ are pack modes.
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_gemm_w4a16_fdot2_stub() {}
