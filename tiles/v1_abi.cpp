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

// Built fatbin slot. dot = shared DOT slot (same dot.hpp source).
struct ArchRow {
  const char* name;
  int dot;
};

// Op rows keep hippihx_v1_op_id / hippihx.list_ops() order. Arch rows
// are hippihx._lib.fatbin.KNOWN_ARCHES. Later slots (gfx906, gfx1013)
// are not rows, so they are refused like any unknown arch (same as Caps).
// hippihx:gen begin v1_table -- python -m hippihx._lib.codegen; do not edit
constexpr OpRow kOps[HIPPIHX_V1_OP_COUNT] = {
    {"attention.fa_fdot2", 1, 0, 1},
    {"attention.gdn_scan", 0, 0, 1},
    {"attention.kda_scan", 0, 0, 0},
    {"attention.qsa_indexer", 0, 0, 0},
    {"attention.dsa_nope", 0, 0, 0},
    {"gemm.w4a16_fdot2", 1, 0, 1},
    {"gemm.exl3_3inst", 1, 0, 1},
    {"moe.routed", 0, 0, 0},
    {"moe.shared", 1, 0, 1},
    {"moe.leftover_bf16", 0, 0, 0},
    {"sequence.causal_conv", 0, 0, 0},
    {"comm.pcie", 0, 0, 0},
};

constexpr ArchRow kArches[] = {
    {"gfx1030", 1},
    {"gfx1100", 1},
    {"gfx1101", 1},
    {"gfx1102", 1},
    {"gfx1151", 1},
    {"gfx1031", 1},
    {"gfx1032", 1},
    {"gfx1033", 1},
    {"gfx1035", 1},
    {"gfx1036", 1},
    {"gfx900", 0},
};
// hippihx:gen end v1_table

bool arch_ok(const char* arch, int is_dot) {
  if (arch == nullptr || arch[0] == '\0') {
    return false;
  }
  if (std::strchr(arch, ',') != nullptr || std::strchr(arch, ' ') != nullptr) {
    return false;  // one fatbin slot per artifact
  }
  for (const ArchRow& row : kArches) {
    if (std::strcmp(arch, row.name) == 0) {
      // A non-DOT slot (gfx900 Vega stub) never loads FA/EXL3 DOT objects.
      return row.dot != 0 || is_dot == 0;
    }
  }
  return false;
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
