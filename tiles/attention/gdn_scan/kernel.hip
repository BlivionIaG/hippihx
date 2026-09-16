#include "hippihx/tile_stub.hpp"

// Contract: attention/gdn_scan — GDN / hybrid-state scan class.
// fp16 activations only — refuse bf16 (dest dispatch missed this). Never fdot2.bf16.
// SSM state dest @ 02adbfd4: fp16 or fp32; recurrence stays fp32 VGPR. Do not dump ST.
// Prefill o varlen: local chunk i_t_local (dest 6c5ff94).
// LDS / __launch_bounds__ locked in README.md when the real tile lands.
__global__ void hippihx_attention_gdn_scan_stub() {}
