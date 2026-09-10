#include "hippihx/tile_stub.hpp"

// Contract: sequence/causal_conv — short-window causal conv (scalar FMA).
// state_len ≈ 4. Not a GDN scan and not under gdn_scan/.
// fp16 or bf16; bf16 promotes mul to fp32 — never fdot2.bf16.
// Scratch zeroed by the caller; no D2H under capture.
__global__ void hippihx_sequence_causal_conv_stub() {}
