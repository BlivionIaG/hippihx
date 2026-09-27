#include <cstdio>
#include <cstring>

#include "hippihx/smoke.hpp"
#include "hippihx/v1.h"

namespace {

int fail(const char* what, int rc) {
  std::fprintf(stderr, "hippihx v1: %s (rc=%d)\n", what, rc);
  return 1;
}

int plan(hippihx_v1_op_id op, const hippihx_v1_caps& caps,
         const hippihx_v1_param* params, size_t nparams, hippihx_v1_plan_t* out) {
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(*out);
  return hippihx_v1_plan(op, &caps, params, nparams, out);
}

}  // namespace

int main() {
  const int magic = hippihx_smoke_magic();
  if (magic != HIPPIHX_SMOKE_MAGIC) {
    std::fprintf(stderr, "hippihx smoke: unexpected magic %d\n", magic);
    return 1;
  }

  if (hippihx_v1_abi_revision() != HIPPIHX_V1_ABI_REVISION) {
    return fail("bad abi revision", hippihx_v1_abi_revision());
  }
  if (hippihx_v1_op_count() != HIPPIHX_V1_OP_COUNT) {
    return fail("op count mismatch", hippihx_v1_op_count());
  }
  const char* fa = hippihx_v1_op_name(HIPPIHX_V1_OP_ATTN_FA_FDOT2);
  if (fa == nullptr || std::strcmp(fa, "attention.fa_fdot2") != 0) {
    return fail("fa name mismatch", 0);
  }
  if (!hippihx_v1_op_is_dot(HIPPIHX_V1_OP_GEMM_EXL3_3INST) ||
      hippihx_v1_op_is_dot(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV)) {
    return fail("DOT flags", 0);
  }
  if (!hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_ATTN_GDN_SCAN) ||
      !hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_GEMM_W4A16_FDOT2) ||
      hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV) ||
      hippihx_v1_op_fp16_act(HIPPIHX_V1_OP_MOE_LEFTOVER_BF16)) {
    return fail("fp16_act flags", 0);
  }
  if (hippihx_v1_op_nparams(HIPPIHX_V1_OP_ATTN_FA_FDOT2) !=
          HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS ||
      hippihx_v1_op_ntensors(HIPPIHX_V1_OP_ATTN_FA_FDOT2) !=
          HIPPIHX_V1_ATTN_FA_FDOT2_NTENSORS) {
    return fail("fa schema sizes", 0);
  }

  hippihx_v1_caps caps{};
  caps.arch = "gfx1030";
  caps.wave = 32;
  hippihx_v1_plan_t p{};

  // Unpinned op: no params, no scratch, not ready, run says NOT_READY.
  int rc = plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, caps, nullptr, 0, &p);
  if (rc != HIPPIHX_V1_OK || p.ready != 0 || p.nscratch != 0 ||
      p.scratch_nbytes != 0) {
    return fail("causal_conv plan", rc);
  }
  rc = hippihx_v1_run(&p, nullptr, 0, nullptr, 0, nullptr);
  if (rc != HIPPIHX_V1_ERR_NOT_READY) {
    return fail("causal_conv run should be NOT_READY", rc);
  }

  // FA decode, the first pinned schema. Split-K partials are plan scratch.
  hippihx_v1_param fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS] = {};
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_MODE].i = HIPPIHX_V1_ATTN_FA_FDOT2_MODE_DECODE;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_HEAD_DIM].i = 128;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_NUM_Q_HEADS].i = 32;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_NUM_KV_HEADS].i = 8;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_BLOCK_SIZE].i = 16;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_KV_SPLITS].i = 4;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_SLIDING_WINDOW].i = 0;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_CAUSAL].i = 1;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_MAX_TOKENS].i = 8;
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_SCALE].f = 0.08838834764831845;
  rc = plan(HIPPIHX_V1_OP_ATTN_FA_FDOT2, caps, fa_params,
            HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS, &p);
  // o_partial 8*32*4*128*4 B, then m/l 8*32*4*4 B each, 256-aligned.
  if (rc != HIPPIHX_V1_OK || p.ready != 0 || p.variant != 0 ||
      p.nscratch != 3 || p.scratch[0].nbytes != 524288 ||
      p.scratch[1].offset != 524288 || p.scratch[2].offset != 528384 ||
      p.scratch_nbytes != 532480) {
    return fail("fa decode plan", rc);
  }

  // A failed plan is zeroed and refused by run.
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_HEAD_DIM].i = 64;
  rc = plan(HIPPIHX_V1_OP_ATTN_FA_FDOT2, caps, fa_params,
            HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS, &p);
  if (rc != HIPPIHX_V1_ERR_PARAM ||
      hippihx_v1_run(&p, nullptr, 0, nullptr, 0, nullptr) != HIPPIHX_V1_ERR_ABI) {
    return fail("head_dim 64 must be refused", rc);
  }
  fa_params[HIPPIHX_V1_ATTN_FA_FDOT2_P_HEAD_DIM].i = 128;

  // bf16 on DOT / GDN HIP is refused (no fdot2.bf16). causal_conv is ok.
  caps.dtype = HIPPIHX_V1_DTYPE_BF16;
  if (plan(HIPPIHX_V1_OP_ATTN_FA_FDOT2, caps, fa_params,
           HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS, &p) != HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE ||
      plan(HIPPIHX_V1_OP_ATTN_GDN_SCAN, caps, nullptr, 0, &p) !=
          HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE ||
      plan(HIPPIHX_V1_OP_GEMM_W4A16_FDOT2, caps, nullptr, 0, &p) !=
          HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    return fail("bf16 must be refused on FA / GDN / W4", 0);
  }
  if (plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, caps, nullptr, 0, &p) !=
          HIPPIHX_V1_OK ||
      plan(HIPPIHX_V1_OP_MOE_LEFTOVER_BF16, caps, nullptr, 0, &p) !=
          HIPPIHX_V1_OK) {
    return fail("causal_conv / leftover_bf16 must accept bf16", 0);
  }
  caps.dtype = HIPPIHX_V1_DTYPE_FP32;
  if (plan(HIPPIHX_V1_OP_ATTN_GDN_SCAN, caps, nullptr, 0, &p) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE) {
    return fail("GDN must refuse fp32", 0);
  }
  caps.dtype = HIPPIHX_V1_DTYPE_UNSET;

  // DOT is wave32 only. gfx900 refuses DOT; Later gfx906 is refused.
  caps.wave = 64;
  if (plan(HIPPIHX_V1_OP_GEMM_EXL3_3INST, caps, nullptr, 0, &p) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_ARCH) {
    return fail("DOT must refuse wave64", 0);
  }
  caps.wave = 0;
  caps.arch = "gfx900";
  if (plan(HIPPIHX_V1_OP_GEMM_EXL3_3INST, caps, nullptr, 0, &p) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_ARCH) {
    return fail("gfx900 must refuse EXL3 DOT", 0);
  }
  if (plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, caps, nullptr, 0, &p) !=
          HIPPIHX_V1_OK ||
      p.wave != 64) {
    return fail("gfx900 non-DOT plans at wave64", 0);
  }
  caps.arch = "gfx906";
  if (plan(HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV, caps, nullptr, 0, &p) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_ARCH) {
    return fail("gfx906 must be refused", 0);
  }
  caps.arch = "gfx1030";

  // comm keys on hop + switch: PIX on 88096 plans, PHB stays RCCL.
  hippihx_v1_fabric fabric{};
  fabric.hop = HIPPIHX_V1_HOP_PIX;
  fabric.pcie_switch = HIPPIHX_V1_SWITCH_PEX88096;
  fabric.width = 16;
  fabric.acs_clear = 1;
  fabric.large_bar = 1;
  caps.fabric = &fabric;
  hippihx_v1_param ar[HIPPIHX_V1_COMM_PCIE_NPARAMS] = {};
  ar[HIPPIHX_V1_COMM_PCIE_P_MAX_BYTES].i = 512 * 1024;
  if (plan(HIPPIHX_V1_OP_COMM_PCIE, caps, ar, HIPPIHX_V1_COMM_PCIE_NPARAMS, &p) !=
      HIPPIHX_V1_OK) {
    return fail("PIX 88096 must plan", 0);
  }
  fabric.hop = HIPPIHX_V1_HOP_PHB;
  if (plan(HIPPIHX_V1_OP_COMM_PCIE, caps, ar, HIPPIHX_V1_COMM_PCIE_NPARAMS, &p) !=
      HIPPIHX_V1_ERR_UNSUPPORTED_FABRIC) {
    return fail("PHB must stay RCCL", 0);
  }

  std::printf("hippihx smoke ok magic=0x%x v1_abi=%d ops=%d\n", magic,
              hippihx_v1_abi_revision(), hippihx_v1_op_count());
  return 0;
}
