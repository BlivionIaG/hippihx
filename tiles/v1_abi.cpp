#include "hippihx/v1.h"

#include <cstring>

namespace {

struct OpRow {
  const char* name;
  int is_dot;
  // Observed extras scratch for the zoo stub. Real sizes land with migrate.
  // causal_conv: register-only (0). Others: placeholder workspace slab.
  size_t scratch_nbytes;
  // 1 = fp16 activations only. Refuse bf16/fp32 (no fdot2.bf16; GDN HIP
  // is fp16). Unset dtype still plans.
  int fp16_act;
};

// Keep order identical to hippihx_v1_op_id and hippihx.list_ops().
constexpr OpRow kOps[HIPPIHX_V1_OP_COUNT] = {
    {"attn.fa_fdot2", 1, 0, 1},
    {"attn.gdn_scan", 0, 0, 1},
    {"attn.kda_scan", 0, 0, 0},
    {"attn.qsa_indexer", 0, 0, 0},
    {"attn.dsa_nope", 0, 0, 0},
    {"gemm.w4a16_fdot2", 1, 0, 1},
    {"gemm.exl3_3inst", 1, 0, 1},
    {"moe.routed", 0, 0, 0},
    {"moe.shared", 1, 0, 1},
    {"moe.leftover_bf16", 0, 0, 0},
    {"sequence.causal_conv", 0, 0, 0},
    {"comm.pcie", 0, 0, 0},
};

bool arch_ok(const char* arch, int is_dot) {
  if (arch == nullptr || arch[0] == '\0') {
    return false;
  }
  if (std::strchr(arch, ',') != nullptr || std::strchr(arch, ' ') != nullptr) {
    return false;  // one fatbin slot per artifact
  }
  // Later non-DOT only — refuse here (same as Caps).
  if (std::strcmp(arch, "gfx906") == 0) {
    return false;
  }
  static const char* kAll[] = {
      "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151",
      "gfx1031", "gfx1032", "gfx1033", "gfx1035", "gfx1036",
      "gfx1013", "gfx900",
  };
  bool known = false;
  for (const char* a : kAll) {
    if (std::strcmp(arch, a) == 0) {
      known = true;
      break;
    }
  }
  if (!known) {
    return false;
  }
  if (is_dot && std::strcmp(arch, "gfx900") == 0) {
    return false;  // Vega stub — no FA/EXL3 DOT objects
  }
  return true;
}

bool valid_op(hippihx_v1_op_id op) {
  return static_cast<int>(op) >= 0 &&
         static_cast<int>(op) < HIPPIHX_V1_OP_COUNT;
}

bool dtype_ok(hippihx_v1_op_id op, int dtype) {
  if (dtype == HIPPIHX_V1_DTYPE_UNSET || dtype == HIPPIHX_V1_DTYPE_FP16) {
    return true;
  }
  if (dtype != HIPPIHX_V1_DTYPE_BF16 && dtype != HIPPIHX_V1_DTYPE_FP32) {
    return false;
  }
  // bf16 / fp32 activations: never DOT (that would be fdot2.bf16) and
  // never dest GDN HIP (fp16-only; extras dispatch missed this guard).
  return kOps[op].fp16_act == 0;
}

}  // namespace

extern "C" int hippihx_v1_abi_revision(void) { return HIPPIHX_V1_ABI_REVISION; }

extern "C" int hippihx_v1_op_count(void) { return HIPPIHX_V1_OP_COUNT; }

extern "C" const char* hippihx_v1_op_name(hippihx_v1_op_id op) {
  if (!valid_op(op)) {
    return nullptr;
  }
  return kOps[op].name;
}

extern "C" int hippihx_v1_op_is_dot(hippihx_v1_op_id op) {
  if (!valid_op(op)) {
    return 0;
  }
  return kOps[op].is_dot;
}

extern "C" int hippihx_v1_op_fp16_act(hippihx_v1_op_id op) {
  if (!valid_op(op)) {
    return 0;
  }
  return kOps[op].fp16_act;
}

extern "C" int hippihx_v1_plan(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                               hippihx_v1_scratch_spec* out_specs,
                               size_t* inout_nspecs) {
  if (!valid_op(op)) {
    return HIPPIHX_V1_ERR_UNKNOWN_OP;
  }
  if (caps == nullptr || inout_nspecs == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (!arch_ok(caps->arch, kOps[op].is_dot)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (!dtype_ok(op, caps->dtype)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE;
  }
  if (*inout_nspecs < 1 || out_specs == nullptr) {
    *inout_nspecs = 1;
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  out_specs[0].name = "workspace";
  out_specs[0].nbytes = kOps[op].scratch_nbytes;
  out_specs[0].zeroed = 1;
  *inout_nspecs = 1;
  return HIPPIHX_V1_OK;
}

extern "C" int hippihx_v1_run(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                              void* scratch, size_t scratch_nbytes) {
  if (!valid_op(op)) {
    return HIPPIHX_V1_ERR_UNKNOWN_OP;
  }
  if (caps == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (!arch_ok(caps->arch, kOps[op].is_dot)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (!dtype_ok(op, caps->dtype)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE;
  }
  const size_t need = kOps[op].scratch_nbytes;
  if (scratch_nbytes < need) {
    return HIPPIHX_V1_ERR_SCRATCH;
  }
  if (need > 0 && scratch == nullptr) {
    return HIPPIHX_V1_ERR_SCRATCH;
  }
  // Contract is live; HIP body still in extras. Serve must not treat OK
  // as a completed migrate — NOT_READY is the intentional stub result.
  (void)scratch;
  return HIPPIHX_V1_ERR_NOT_READY;
}
