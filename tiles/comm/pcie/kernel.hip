#include "hippihx/tile_stub.hpp"

// Contract: comm/pcie — Uncached + push all-reduce (INT8/Q8 wire class). Stub only.
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_comm_pcie_stub() {}
