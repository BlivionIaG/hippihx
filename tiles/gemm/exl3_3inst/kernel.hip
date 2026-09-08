#include "hippihx/tile_stub.hpp"

// Contract: gemm/exl3_3inst — HIP consume hook for 3inst-packed EXL3 (fdot2).
// Produce (`-cb 3inst`) is not a hippihx directory.
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_gemm_exl3_3inst_stub() {}
