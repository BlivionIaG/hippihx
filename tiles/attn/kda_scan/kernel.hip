#include "hippihx/tile_stub.hpp"

// Contract: attn/kda_scan — KDA linear-state scan class (not GDN, not 128×128 fused).
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_attn_kda_scan_stub() {}
