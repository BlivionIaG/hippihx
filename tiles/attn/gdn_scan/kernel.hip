#include "hippihx/tile_stub.hpp"

// Contract: attn/gdn_scan — GDN / hybrid-state scan class (not product-named).
// fp16 activations only — refuse bf16 (dest dispatch missed this). Never fdot2.bf16.
// Prefill o varlen: local chunk i_t_local (dest 6c5ff94).
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_attn_gdn_scan_stub() {}
