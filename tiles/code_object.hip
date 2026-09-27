// One device-only translation unit per fatbin slot. CMake builds it into
// hippihx_<arch>.hsaco: one --offload-arch, --cuda-device-only,
// --no-gpu-bundle-output, so the file is one raw AMDGPU ELF.
// hippihx_v1_load checks its EF_AMDGPU_MACH, then hipModuleLoadData loads it.
// DOT tiles are in the object only on a DOT slot (never gfx900).

#include "hippihx/arch.hpp"

// hippihx:gen begin code_object_tiles -- python -m hippihx._lib.codegen; do not edit
#include "attention/gdn_scan/kernel.hip"
#include "attention/kda_scan/kernel.hip"
#include "attention/qsa_indexer/kernel.hip"
#include "attention/dsa_nope/kernel.hip"
#include "moe/routed/kernel.hip"
#include "moe/leftover_bf16/kernel.hip"
#include "sequence/causal_conv/kernel.hip"
#include "comm/pcie/kernel.hip"
#if defined(HIPPIHX_DOT_SLOT)
#include "attention/fa_fdot2/kernel.hip"
#include "gemm/w4a16_fdot2/kernel.hip"
#include "gemm/exl3_3inst/kernel.hip"
#include "moe/shared/kernel.hip"
#endif
// hippihx:gen end code_object_tiles
