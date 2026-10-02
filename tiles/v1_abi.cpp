#include "hippihx/v1.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

// Built as libhippihx_v1.so with HIP, this links libamdhip64 and loads code
// objects. Without HIP (host stub, the link-smoke .a) the loader validates
// images and returns HIPPIHX_V1_ERR_NO_HIP.
#if defined(HIPPIHX_V1_WITH_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace {

// Row types for the generated tables. A param reference inside a row
// (CondRow.param, CheckRow, DimRow of kind kParam / kLeParam) is an index
// into the op's own params, i.e. into plan->params.
enum ParamKind { kInt = 0, kEnum = 1, kBool = 2, kFloat = 3 };
enum CondOp { kEq = 0, kNe = 1, kGt = 2, kGe = 3, kLt = 4, kLe = 5 };
enum DimKind { kAny = 0, kParam = 1, kLeParam = 2, kLiteral = 3 };
enum Layout { kContiguous = 0, kRows = 1, kStrided = 2 };

// Built fatbin slot. dot = shared DOT slot (same dot.hpp source).
struct ArchRow {
  const char* name;
  int dot;
  int wave;  // default when caps.wave == 0
  int mach;  // EF_AMDGPU_MACH of this slot's code object
};

struct ParamRow {
  const char* name;
  int kind;
  int64_t lo;  // kInt: inclusive bounds
  int64_t hi;
  int values_first;  // kEnum: kEnumValues range
  int values_count;
};

struct CondRow {
  int param;
  int op;
  int64_t value;
};

struct CheckRow {
  int num;  // params[num] % params[by] == 0
  int by;
};

struct DimRow {
  int kind;
  int64_t value;  // param index, or the literal extent
};

struct SlotRow {
  const char* name;
  int dtype;
  int rank;
  int dims_first;
  int layout;
  int when_first;  // required iff any cond holds; none = always
  int when_count;
};

struct ScratchRow {
  const char* name;
  int64_t elem_bytes;
  int dims_first;  // elem_bytes * prod(dims) bytes
  int dims_count;
  int64_t max_elems;  // > 0: bound on prod(dims) (kernel index width)
  int when_first;     // present iff any cond holds; none = always
  int when_count;
};

// Columns: name, is_dot, fp16_act (refuse bf16/fp32 act: no fdot2.bf16,
// GDN HIP is fp16), ready, needs_fabric, then first/count into kParams,
// kChecks, kSlots, kScratch, then the variant param (-1 = none) and its
// first kVariantRanks row (explore rank per enum value).
struct OpRow {
  const char* name;
  int is_dot;
  int fp16_act;
  int ready;
  int needs_fabric;
  int params_first;
  int params_count;
  int checks_first;
  int checks_count;
  int slots_first;
  int slots_count;
  int scratch_first;
  int scratch_count;
  int variant_param;
  int variant_first;
};

// Op rows keep hippihx_v1_op_id / hippihx.list_ops() order. Arch rows are
// hippihx._lib.fatbin.KNOWN_ARCHES. Later slots (gfx906, gfx1013) are not
// rows, so they are refused like any unknown arch (same as Caps).
// hippihx:gen begin v1_table -- python -m hippihx._lib.codegen; do not edit
constexpr OpRow kOps[HIPPIHX_V1_OP_COUNT] = {
    {"attention.fa_fdot2", 1, 1, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, 0},
    {"attention.gdn_scan", 0, 1, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"attention.kda_scan", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"attention.qsa_indexer", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"attention.dsa_nope", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"gemm.w4a16_fdot2", 1, 1, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"gemm.exl3_3inst", 1, 1, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"moe.routed", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"moe.shared", 1, 1, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"moe.leftover_bf16", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"sequence.causal_conv", 0, 0, 0, 0, 10, 0, 1, 0, 7, 0, 3, 0, -1, 0},
    {"comm.pcie", 0, 0, 0, 1, 10, 1, 1, 0, 7, 0, 3, 0, -1, 0},
};

constexpr ArchRow kArches[HIPPIHX_V1_ARCH_COUNT] = {
    {"gfx1030", 1, 32, 0x36},
    {"gfx1100", 1, 32, 0x41},
    {"gfx1101", 1, 32, 0x46},
    {"gfx1102", 1, 32, 0x47},
    {"gfx1200", 1, 32, 0x48},
    {"gfx1151", 1, 32, 0x4A},
    {"gfx1031", 1, 32, 0x37},
    {"gfx1032", 1, 32, 0x38},
    {"gfx1033", 1, 32, 0x39},
    {"gfx1035", 1, 32, 0x3D},
    {"gfx1036", 1, 32, 0x45},
    {"gfx900", 0, 64, 0x2C},
};

constexpr ParamRow kParams[] = {
    {"mode", kEnum, INT64_MIN, INT64_MAX, 0, 2},
    {"head_dim", kEnum, INT64_MIN, INT64_MAX, 2, 2},
    {"num_q_heads", kInt, 1, INT64_MAX, 4, 0},
    {"num_kv_heads", kInt, 1, INT64_MAX, 4, 0},
    {"block_size", kInt, 1, INT64_MAX, 4, 0},
    {"kv_splits", kInt, 1, 16, 4, 0},
    {"sliding_window", kInt, 0, INT64_MAX, 4, 0},
    {"causal", kBool, INT64_MIN, INT64_MAX, 4, 0},
    {"max_tokens", kInt, 1, INT64_MAX, 4, 0},
    {"scale", kFloat, INT64_MIN, INT64_MAX, 4, 0},
    {"max_bytes", kInt, 1, 524288, 4, 0},
};
constexpr int64_t kEnumValues[] = {
    0,
    1,
    128,
    256,
};
constexpr CondRow kConds[] = {
    {0, kEq, 1},
    {0, kEq, 0},
    {5, kGt, 1},
    {0, kEq, 0},
    {5, kGt, 1},
    {0, kEq, 0},
    {5, kGt, 1},
};
constexpr CheckRow kChecks[] = {
    {2, 3},
};
constexpr DimRow kDims[] = {
    {kLeParam, 8},
    {kParam, 2},
    {kParam, 1},
    {kAny, 0},
    {kParam, 3},
    {kAny, 0},
    {kParam, 4},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kAny, 0},
    {kLeParam, 8},
    {kParam, 2},
    {kParam, 1},
    {kParam, 8},
    {kParam, 2},
    {kParam, 5},
    {kParam, 1},
    {kParam, 8},
    {kParam, 2},
    {kParam, 5},
    {kParam, 8},
    {kParam, 2},
    {kParam, 5},
};
constexpr SlotRow kSlots[] = {
    {"q", HIPPIHX_V1_DTYPE_FP16, 3, 0, kContiguous, 0, 0},
    {"k_cache", HIPPIHX_V1_DTYPE_FP16, 5, 3, kStrided, 0, 0},
    {"v_cache", HIPPIHX_V1_DTYPE_FP16, 5, 8, kStrided, 0, 0},
    {"block_table", HIPPIHX_V1_DTYPE_I32, 2, 13, kRows, 0, 0},
    {"seq_lens", HIPPIHX_V1_DTYPE_I32, 1, 15, kContiguous, 0, 0},
    {"cu_query_lens", HIPPIHX_V1_DTYPE_I32, 1, 16, kContiguous, 0, 1},
    {"out", HIPPIHX_V1_DTYPE_FP16, 3, 17, kContiguous, 1, 0},
};
constexpr ScratchRow kScratch[] = {
    {"o_partial", 4, 20, 4, 2147483647, 1, 2},
    {"m_partial", 4, 24, 3, 0, 3, 2},
    {"l_partial", 4, 27, 3, 0, 5, 2},
};
constexpr int kVariantRanks[] = {
    0,
    1,
};

constexpr int kNumHops = 3;
constexpr int kNumSwitches = 3;
constexpr int kArHops[] = {
    HIPPIHX_V1_HOP_PIX,
};
constexpr int kArSwitches[] = {
    HIPPIHX_V1_SWITCH_PEX88096,
    HIPPIHX_V1_SWITCH_GENERIC,
};
constexpr int kLinkWidths[] = {
    1,
    2,
    4,
    8,
    16,
};
// hippihx:gen end v1_table

bool valid_op(int op) { return op >= 0 && op < HIPPIHX_V1_OP_COUNT; }

// One loaded code object per slot, for the process lifetime. Loads happen
// at serve init, single-threaded, before any capture; plan and run only
// read this table.
struct LoadedSlot {
  int loaded;
  std::unique_ptr<unsigned char[]> image;  // kept alive for the module
  void* module;                            // hipModule_t
};
LoadedSlot g_slots[HIPPIHX_V1_ARCH_COUNT];

int find_arch(const char* arch) {
  if (arch == nullptr || arch[0] == '\0') {
    return -1;
  }
  if (std::strchr(arch, ',') != nullptr || std::strchr(arch, ' ') != nullptr) {
    return -1;  // one fatbin slot per artifact
  }
  for (int i = 0; i < HIPPIHX_V1_ARCH_COUNT; ++i) {
    if (std::strcmp(arch, kArches[i].name) == 0) {
      return i;
    }
  }
  return -1;
}

bool wave_ok(int is_dot, int wave) {
  if (is_dot) {
    return wave == 32;  // DOT tiles are wave32 only
  }
  return wave == 32 || wave == 64;
}

bool dtype_ok(const OpRow& row, int dtype) {
  if (dtype == HIPPIHX_V1_DTYPE_UNSET || dtype == HIPPIHX_V1_DTYPE_FP16) {
    return true;
  }
  if (dtype != HIPPIHX_V1_DTYPE_BF16 && dtype != HIPPIHX_V1_DTYPE_FP32) {
    return false;  // not an activation dtype
  }
  // bf16 / fp32 activations: never DOT (that would be fdot2.bf16) and
  // never dest GDN HIP (fp16-only; extras dispatch missed this guard).
  return row.fp16_act == 0;
}

bool cond_holds(const CondRow& cond, const hippihx_v1_param* params) {
  const int64_t v = params[cond.param].i;
  switch (cond.op) {
    case kEq:
      return v == cond.value;
    case kNe:
      return v != cond.value;
    case kGt:
      return v > cond.value;
    case kGe:
      return v >= cond.value;
    case kLt:
      return v < cond.value;
    default:
      return v <= cond.value;
  }
}

bool any_cond(int first, int count, const hippihx_v1_param* params) {
  if (count == 0) {
    return true;
  }
  for (int i = 0; i < count; ++i) {
    if (cond_holds(kConds[first + i], params)) {
      return true;
    }
  }
  return false;
}

bool param_ok(const ParamRow& row, const hippihx_v1_param& value) {
  switch (row.kind) {
    case kFloat:
      return std::isfinite(value.f) && value.f > 0.0;
    case kBool:
      return value.i == 0 || value.i == 1;
    case kEnum:
      for (int i = 0; i < row.values_count; ++i) {
        if (value.i == kEnumValues[row.values_first + i]) {
          return true;
        }
      }
      return false;
    default:
      return value.i >= row.lo && value.i <= row.hi;
  }
}

bool in_list(int value, const int* list, int count) {
  for (int i = 0; i < count; ++i) {
    if (list[i] == value) {
      return true;
    }
  }
  return false;
}

template <typename T, int N>
constexpr int count_of(const T (&)[N]) {
  return N;
}

int fabric_status(const hippihx_v1_fabric* fabric) {
  if (fabric == nullptr) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_FABRIC;  // no fabric: RCCL
  }
  if (fabric->hop < 1 || fabric->hop > kNumHops || fabric->pcie_switch < 1 ||
      fabric->pcie_switch > kNumSwitches ||
      !in_list(fabric->width, kLinkWidths, count_of(kLinkWidths)) ||
      (fabric->acs_clear != 0 && fabric->acs_clear != 1) ||
      (fabric->large_bar != 0 && fabric->large_bar != 1)) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (fabric->hop == HIPPIHX_V1_HOP_PIX && fabric->acs_clear == 0) {
    return HIPPIHX_V1_ERR_BAD_ARG;  // SrcValid+ is a host bounce, not a PIX hop
  }
  const bool custom_ar =
      in_list(fabric->hop, kArHops, count_of(kArHops)) &&
      in_list(fabric->pcie_switch, kArSwitches, count_of(kArSwitches)) &&
      fabric->acs_clear == 1 && fabric->large_bar == 1;
  return custom_ar ? HIPPIHX_V1_OK : HIPPIHX_V1_ERR_UNSUPPORTED_FABRIC;
}

bool align_up(int64_t n, int64_t* out) {
  const int64_t a = HIPPIHX_V1_SCRATCH_ALIGN;
  if (n > INT64_MAX - (a - 1)) {
    return false;
  }
  *out = (n + a - 1) / a * a;
  return true;
}

int plan_scratch(const OpRow& row, const hippihx_v1_param* params,
                 hippihx_v1_plan_t* out) {
  int64_t cursor = 0;
  int n = 0;
  for (int i = 0; i < row.scratch_count; ++i) {
    const ScratchRow& rule = kScratch[row.scratch_first + i];
    if (!any_cond(rule.when_first, rule.when_count, params)) {
      continue;
    }
    int64_t elems = 1;
    for (int d = 0; d < rule.dims_count; ++d) {
      const DimRow& dim = kDims[rule.dims_first + d];
      const int64_t factor =
          dim.kind == kLiteral ? dim.value : params[dim.value].i;
      if (factor < 1 || __builtin_mul_overflow(elems, factor, &elems)) {
        return HIPPIHX_V1_ERR_PARAM;
      }
    }
    if (rule.max_elems > 0 && elems > rule.max_elems) {
      return HIPPIHX_V1_ERR_PARAM;  // kernel index width
    }
    int64_t nbytes = 0;
    int64_t offset = 0;
    if (__builtin_mul_overflow(elems, rule.elem_bytes, &nbytes) ||
        !align_up(cursor, &offset) ||
        __builtin_add_overflow(offset, nbytes, &cursor) ||
        cursor > INT64_MAX - HIPPIHX_V1_SCRATCH_ALIGN) {
      return HIPPIHX_V1_ERR_PARAM;
    }
    out->scratch[n].name = rule.name;
    out->scratch[n].nbytes = static_cast<size_t>(nbytes);
    out->scratch[n].offset = static_cast<size_t>(offset);
    out->scratch[n].zeroed = 1;
    ++n;
  }
  int64_t total = 0;
  align_up(cursor, &total);  // cursor was bounded above
  out->nscratch = n;
  out->scratch_nbytes = static_cast<size_t>(total);
  return HIPPIHX_V1_OK;
}

int plan_into(hippihx_v1_op_id op, const hippihx_v1_caps& caps,
              const hippihx_v1_param* params, size_t nparams,
              hippihx_v1_plan_t* out) {
  const OpRow& row = kOps[op];
  const int arch = find_arch(caps.arch);
  if (arch < 0) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (row.is_dot && !kArches[arch].dot) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;  // gfx900 never loads DOT objects
  }
  const int wave = caps.wave == 0 ? kArches[arch].wave : caps.wave;
  if (!wave_ok(row.is_dot, wave)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (!dtype_ok(row, caps.dtype)) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE;
  }
  if (nparams > 0 && params == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (nparams != static_cast<size_t>(row.params_count)) {
    return HIPPIHX_V1_ERR_PARAM;
  }
  for (int i = 0; i < row.params_count; ++i) {
    if (!param_ok(kParams[row.params_first + i], params[i])) {
      return HIPPIHX_V1_ERR_PARAM;
    }
  }
  for (int i = 0; i < row.checks_count; ++i) {
    const CheckRow& check = kChecks[row.checks_first + i];
    const int64_t by = params[check.by].i;
    if (by == 0 || params[check.num].i % by != 0) {
      return HIPPIHX_V1_ERR_PARAM;
    }
  }
  if (row.needs_fabric) {
    const int rc = fabric_status(caps.fabric);
    if (rc != HIPPIHX_V1_OK) {
      return rc;
    }
  }

  out->abi_revision = HIPPIHX_V1_ABI_REVISION;
  out->op = op;
  out->arch = arch;
  out->wave = wave;
  out->dtype = caps.dtype;
  out->ready = row.ready && g_slots[arch].loaded;
  out->nparams = row.params_count;
  for (int i = 0; i < row.params_count; ++i) {
    out->params[i] = params[i];
  }
  out->variant = 0;
  if (row.variant_param >= 0) {
    const ParamRow& param = kParams[row.params_first + row.variant_param];
    for (int i = 0; i < param.values_count; ++i) {
      if (kEnumValues[param.values_first + i] == params[row.variant_param].i) {
        out->variant = kVariantRanks[row.variant_first + i];
      }
    }
  }
  return plan_scratch(row, params, out);
}

uint32_t read_le(const unsigned char* p, int nbytes) {
  uint32_t v = 0;
  for (int i = nbytes - 1; i >= 0; --i) {
    v = (v << 8) | p[i];
  }
  return v;
}

// One raw AMDGPU HSA ELF (ELF64, little-endian, ET_DYN, EM_AMDGPU) whose
// EF_AMDGPU_MACH is the slot's. Offload bundles and .o files are refused.
int check_image(int slot, const unsigned char* image, size_t nbytes) {
  constexpr unsigned kElfClass64 = 2;
  constexpr unsigned kElfData2Lsb = 1;
  constexpr unsigned kOsAbiAmdgpuHsa = 64;
  constexpr uint32_t kEtDyn = 3;
  constexpr uint32_t kEmAmdgpu = 224;
  if (nbytes < 64 || std::memcmp(image, "\x7f" "ELF", 4) != 0 ||
      image[4] != kElfClass64 || image[5] != kElfData2Lsb ||
      image[7] != kOsAbiAmdgpuHsa || read_le(image + 16, 2) != kEtDyn ||
      read_le(image + 18, 2) != kEmAmdgpu) {
    return HIPPIHX_V1_ERR_CODE_OBJECT;
  }
  if (static_cast<int>(read_le(image + 48, 4) & 0xFF) != kArches[slot].mach) {
    return HIPPIHX_V1_ERR_FOREIGN_ISA;  // never load arch A's object on B
  }
  return HIPPIHX_V1_OK;
}

bool hsa_override_set() {
  const char* v = std::getenv("HSA_OVERRIDE_GFX_VERSION");
  return v != nullptr && v[0] != '\0';
}

int load_slot(int slot, const unsigned char* image, size_t nbytes) {
  const int rc = check_image(slot, image, nbytes);
  if (rc != HIPPIHX_V1_OK || g_slots[slot].loaded) {
    return rc;  // a second load of a slot is a no-op
  }
#if defined(HIPPIHX_V1_WITH_HIP)
  // Keep a private copy alive for the module's lifetime.
  std::unique_ptr<unsigned char[]> copy(new unsigned char[nbytes]);
  std::memcpy(copy.get(), image, nbytes);
  hipModule_t module = nullptr;
  if (hipModuleLoadData(&module, copy.get()) != hipSuccess) {
    return HIPPIHX_V1_ERR_CODE_OBJECT;
  }
  g_slots[slot].image = std::move(copy);
  g_slots[slot].module = module;
  g_slots[slot].loaded = 1;
  return HIPPIHX_V1_OK;
#else
  return HIPPIHX_V1_ERR_NO_HIP;
#endif
}

bool layout_ok(int layout, const hippihx_v1_tensor& t) {
  if (layout == kStrided) {
    return true;
  }
  // Size-1 axes carry no stride contract (same as torch is_contiguous).
  int64_t span = 1;  // elements covered by the axes inside d
  for (int d = t.rank - 1; d >= 0; --d) {
    const bool unit = t.shape[d] == 1;
    if (!unit) {
      if (layout == kContiguous || d == t.rank - 1) {
        if (t.stride[d] != span) {
          return false;
        }
      } else if (t.stride[d] < span) {
        return false;  // kRows: padded outer strides, no overlap
      }
    }
    const int64_t step = unit ? span : t.stride[d];
    if (__builtin_mul_overflow(step, t.shape[d], &span)) {
      return false;
    }
  }
  return true;
}

bool tensor_ok(const SlotRow& slot, const hippihx_v1_tensor& t,
               const hippihx_v1_param* params) {
  if (t.data == nullptr || t.dtype != slot.dtype || t.rank != slot.rank) {
    return false;
  }
  for (int d = 0; d < t.rank; ++d) {
    const int64_t extent = t.shape[d];
    if (extent < 1 || t.stride[d] < 0) {
      return false;
    }
    const DimRow& dim = kDims[slot.dims_first + d];
    if ((dim.kind == kParam && extent != params[dim.value].i) ||
        (dim.kind == kLeParam && extent > params[dim.value].i) ||
        (dim.kind == kLiteral && extent != dim.value)) {
      return false;
    }
  }
  return layout_ok(slot.layout, t);
}

}  // namespace

extern "C" int hippihx_v1_abi_revision(void) { return HIPPIHX_V1_ABI_REVISION; }

extern "C" int hippihx_v1_op_count(void) { return HIPPIHX_V1_OP_COUNT; }

extern "C" const char* hippihx_v1_op_name(hippihx_v1_op_id op) {
  return valid_op(op) ? kOps[op].name : nullptr;
}

extern "C" int hippihx_v1_op_is_dot(hippihx_v1_op_id op) {
  return valid_op(op) ? kOps[op].is_dot : 0;
}

extern "C" int hippihx_v1_op_fp16_act(hippihx_v1_op_id op) {
  return valid_op(op) ? kOps[op].fp16_act : 0;
}

extern "C" int hippihx_v1_op_nparams(hippihx_v1_op_id op) {
  return valid_op(op) ? kOps[op].params_count : -1;
}

extern "C" const char* hippihx_v1_op_param_name(hippihx_v1_op_id op, int index) {
  if (!valid_op(op) || index < 0 || index >= kOps[op].params_count) {
    return nullptr;
  }
  return kParams[kOps[op].params_first + index].name;
}

extern "C" int hippihx_v1_op_ntensors(hippihx_v1_op_id op) {
  return valid_op(op) ? kOps[op].slots_count : -1;
}

extern "C" const char* hippihx_v1_op_tensor_name(hippihx_v1_op_id op, int index) {
  if (!valid_op(op) || index < 0 || index >= kOps[op].slots_count) {
    return nullptr;
  }
  return kSlots[kOps[op].slots_first + index].name;
}

extern "C" int hippihx_v1_load_image(const char* arch, const void* image,
                                     size_t nbytes) {
  const int slot = find_arch(arch);
  if (slot < 0) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (hsa_override_set()) {
    return HIPPIHX_V1_ERR_FOREIGN_ISA;
  }
  if (image == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  return load_slot(slot, static_cast<const unsigned char*>(image), nbytes);
}

extern "C" int hippihx_v1_load(const char* arch, const char* path) {
  const int slot = find_arch(arch);
  if (slot < 0) {
    return HIPPIHX_V1_ERR_UNSUPPORTED_ARCH;
  }
  if (hsa_override_set()) {
    return HIPPIHX_V1_ERR_FOREIGN_ISA;
  }
  if (path == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    return HIPPIHX_V1_ERR_CODE_OBJECT;
  }
  long size = -1;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    size = std::ftell(f);
  }
  if (size < 0 || std::fseek(f, 0, SEEK_SET) != 0) {
    std::fclose(f);
    return HIPPIHX_V1_ERR_CODE_OBJECT;
  }
  const size_t nbytes = static_cast<size_t>(size);
  std::unique_ptr<unsigned char[]> image(new unsigned char[nbytes > 0 ? nbytes : 1]);
  const size_t got = std::fread(image.get(), 1, nbytes, f);
  std::fclose(f);
  if (got != nbytes) {
    return HIPPIHX_V1_ERR_CODE_OBJECT;
  }
  return load_slot(slot, image.get(), nbytes);
}

extern "C" int hippihx_v1_loaded(const char* arch) {
  const int slot = find_arch(arch);
  return slot >= 0 && g_slots[slot].loaded;
}

extern "C" int hippihx_v1_plan(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                               const hippihx_v1_param* params, size_t nparams,
                               hippihx_v1_plan_t* out) {
  if (!valid_op(op)) {
    return HIPPIHX_V1_ERR_UNKNOWN_OP;
  }
  if (caps == nullptr || out == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (out->struct_size != sizeof(hippihx_v1_plan_t)) {
    return HIPPIHX_V1_ERR_ABI;  // header / library mismatch; out untouched
  }
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(hippihx_v1_plan_t);
  const int rc = plan_into(op, *caps, params, nparams, out);
  if (rc != HIPPIHX_V1_OK) {
    // A failed plan keeps abi_revision == 0, so run refuses it.
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(hippihx_v1_plan_t);
  }
  return rc;
}

extern "C" int hippihx_v1_run(const hippihx_v1_plan_t* plan,
                              const hippihx_v1_tensor* tensors, size_t ntensors,
                              void* scratch, size_t scratch_nbytes,
                              void* stream) {
  if (plan == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  if (plan->struct_size != sizeof(hippihx_v1_plan_t) ||
      plan->abi_revision != HIPPIHX_V1_ABI_REVISION) {
    return HIPPIHX_V1_ERR_ABI;
  }
  if (!valid_op(plan->op)) {
    return HIPPIHX_V1_ERR_UNKNOWN_OP;
  }
  const OpRow& row = kOps[plan->op];
  if (ntensors != static_cast<size_t>(row.slots_count)) {
    return HIPPIHX_V1_ERR_TENSOR;
  }
  if (ntensors > 0 && tensors == nullptr) {
    return HIPPIHX_V1_ERR_BAD_ARG;
  }
  for (int i = 0; i < row.slots_count; ++i) {
    const SlotRow& slot = kSlots[row.slots_first + i];
    if (!any_cond(slot.when_first, slot.when_count, plan->params)) {
      continue;  // optional slot this plan does not use
    }
    if (!tensor_ok(slot, tensors[i], plan->params)) {
      return HIPPIHX_V1_ERR_TENSOR;
    }
  }
  if (scratch_nbytes < plan->scratch_nbytes ||
      (plan->scratch_nbytes > 0 && scratch == nullptr) ||
      reinterpret_cast<uintptr_t>(scratch) % HIPPIHX_V1_SCRATCH_ALIGN != 0) {
    return HIPPIHX_V1_ERR_SCRATCH;
  }
  (void)stream;
  // No migrated body is linked into this library yet, so every plan has
  // ready == 0. A ready plan dispatches on plan->variant here.
  return HIPPIHX_V1_ERR_NOT_READY;
}
