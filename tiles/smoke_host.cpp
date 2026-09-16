#include <cstdio>
#include <cstring>

#include "hippihx/smoke.hpp"
#include "hippihx/v1.h"

int main() {
  const int magic = hippihx_smoke_magic();
  if (magic != HIPPIHX_SMOKE_MAGIC) {
    std::fprintf(stderr, "hippihx smoke: unexpected magic %d\n", magic);
    return 1;
  }

  if (hippihx_v1_abi_revision() != HIPPIHX_V1_ABI_REVISION) {
    std::fprintf(stderr, "hippihx v1: bad abi revision\n");
    return 1;
  }
  if (hippihx_v1_op_count() != HIPPIHX_V1_OP_COUNT) {
    std::fprintf(stderr, "hippihx v1: op count mismatch\n");
    return 1;
  }

  const char* fa = hippihx_v1_op_name(HIPPIHX_V1_OP_ATTN_FA_FDOT2);
  if (fa == nullptr || std::strcmp(fa, "attention.fa_fdot2") != 0) {
    std::fprintf(stderr, "hippihx v1: fa name mismatch\n");
    return 1;
  }
  if (!hippihx_v1_op_is_dot(HIPPIHX_V1_OP_GEMM_EXL3_3INST)) {
    std::fprintf(stderr, "hippihx v1: exl3 should be DOT\n");
    return 1;
  }
  if (hippihx_v1_op_is_dot(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV)) {
    std::fprintf(stderr, "hippihx v1: causal_conv must not be DOT\n");
    return 1;
  }
  if (!hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_ATTN_GDN_SCAN) ||
      !hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_GEMM_W4A16_FDOT2)) {
    std::fprintf(stderr, "hippihx v1: GDN/W4 must be fp16_act\n");
    return 1;
  }
  if (hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV) ||
      hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_MOE_LEFTOVER_BF16)) {
    std::fprintf(stderr, "hippihx v1: leftover/conv must not be fp16_act\n");
    return 1;
  }

  hippihx_v1_caps caps{};
  caps.arch = "gfx1030";
  caps.wave = 32;
  hippihx_v1_scratch_spec specs[1];
  size_t nspecs = 1;
  const int plan_rc =
      hippihx_v1_plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, &caps, specs, &nspecs);
  if (plan_rc != HIPPIHX_V1_OK || nspecs != 1 || specs[0].nbytes != 0 ||
      specs[0].zeroed != 1) {
    std::fprintf(stderr, "hippihx v1: causal_conv plan failed rc=%d\n", plan_rc);
    return 1;
  }

  const int run_rc =
      hippihx_v1_run(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, &caps, nullptr, 0);
  if (run_rc != HIPPIHX_V1_ERR_NOT_READY) {
    std::fprintf(stderr, "hippihx v1: expected NOT_READY, got %d\n", run_rc);
    return 1;
  }

  // bf16 on DOT / GDN HIP is refused (no fdot2.bf16). causal_conv is ok.
  caps.dtype = HIPPIHX_V1_DTYPE_BF16;
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_ATTN_FA_FDOT2, &caps, specs, &nspecs) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    std::fprintf(stderr, "hippihx v1: FA must refuse bf16\n");
    return 1;
  }
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_ATTN_GDN_SCAN, &caps, specs, &nspecs) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    std::fprintf(stderr, "hippihx v1: GDN must refuse bf16\n");
    return 1;
  }
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, &caps, specs,
                      &nspecs) != HIPPIHX_V1_OK) {
    std::fprintf(stderr, "hippihx v1: causal_conv must accept bf16\n");
    return 1;
  }
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_MOE_LEFTOVER_BF16, &caps, specs, &nspecs) !=
      HIPPIHX_V1_OK) {
    std::fprintf(stderr, "hippihx v1: leftover_bf16 must accept bf16\n");
    return 1;
  }
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_GEMM_W4A16_FDOT2, &caps, specs, &nspecs) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    std::fprintf(stderr, "hippihx v1: W4 must refuse bf16\n");
    return 1;
  }
  caps.dtype = HIPPIHX_V1_DTYPE_FP32;
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_ATTN_GDN_SCAN, &caps, specs, &nspecs) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    std::fprintf(stderr, "hippihx v1: GDN must refuse fp32\n");
    return 1;
  }
  caps.dtype = HIPPIHX_V1_DTYPE_UNSET;

  // DOT refused on gfx900; Later gfx906 refused entirely.
  caps.arch = "gfx900";
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_GEMM_EXL3_3INST, &caps, specs, &nspecs) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_ARCH) {
    std::fprintf(stderr, "hippihx v1: gfx900 must refuse EXL3 DOT\n");
    return 1;
  }
  caps.arch = "gfx906";
  nspecs = 1;
  if (hippihx_v1_plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, &caps, specs,
                      &nspecs) != HIPPIHX_V1_ERR_UNSUPPORTED_ARCH) {
    std::fprintf(stderr, "hippihx v1: gfx906 must be refused\n");
    return 1;
  }

  std::printf("hippihx smoke ok magic=0x%x v1_abi=%d ops=%d\n", magic,
              hippihx_v1_abi_revision(), hippihx_v1_op_count());
  return 0;
}
