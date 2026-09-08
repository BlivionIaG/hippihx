#include "hippihx/tile_stub.hpp"

// Contract: comm/pcie — Uncached+push AR Later. INT8/Q8 wire preferred.
// Leave E4M3 / f8_dma without FP8 HW. Scratch zeroed; no D2H under capture.
// Not a DOT tile — do not load FA/EXL3/AWQ objects into a Vega fatbin for this.
__global__ void hippihx_comm_pcie_stub() {}
