#pragma once

// Stable C consume ABI for rdna_extras (and any serve layer).
//
// One V1 entry family per op. Serve wraps these as torch.ops.hippihx.* —
// hippihx itself stays torch-free. Scratch is sized by plan; the caller
// allocates and zeros for cudagraph page-commit. bind/run must not
// allocate. run must not D2H under capture.
//
// Bodies still live in extras until extras rewires onto these symbols.
// Until then, hippihx_v1_run is a no-op that validates plan/scratch
// sizes only.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  HIPPIHX_V1_OK = 0,
  HIPPIHX_V1_ERR_UNKNOWN_OP = 1,
  HIPPIHX_V1_ERR_UNSUPPORTED_ARCH = 2,
  HIPPIHX_V1_ERR_BAD_ARG = 3,
  HIPPIHX_V1_ERR_SCRATCH = 4,
  HIPPIHX_V1_ERR_NOT_READY = 5,  // contract exists; HIP body not migrated
  HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE = 6,  // e.g. bf16 on fdot2 / GDN HIP
};

// Activation dtype. 0 = unset (do not refuse). DOT and GDN HIP are fp16;
// never fdot2.bf16 (gfx1030 LLVM ISel abort).
typedef enum hippihx_v1_dtype {
  HIPPIHX_V1_DTYPE_UNSET = 0,
  HIPPIHX_V1_DTYPE_FP16 = 1,
  HIPPIHX_V1_DTYPE_BF16 = 2,
  HIPPIHX_V1_DTYPE_FP32 = 3,
} hippihx_v1_dtype;

// Stable ids — match hippihx.list_ops() order. Do not renumber.
typedef enum hippihx_v1_op_id {
  HIPPIHX_V1_OP_ATTN_FA_FDOT2 = 0,
  HIPPIHX_V1_OP_ATTN_GDN_SCAN = 1,
  HIPPIHX_V1_OP_ATTN_KDA_SCAN = 2,
  HIPPIHX_V1_OP_ATTN_QSA_INDEXER = 3,
  HIPPIHX_V1_OP_ATTN_DSA_NOPE = 4,
  HIPPIHX_V1_OP_GEMM_W4A16_FDOT2 = 5,
  HIPPIHX_V1_OP_GEMM_EXL3_3INST = 6,
  HIPPIHX_V1_OP_MOE_ROUTED = 7,
  HIPPIHX_V1_OP_MOE_SHARED = 8,
  HIPPIHX_V1_OP_MOE_LEFTOVER_BF16 = 9,
  HIPPIHX_V1_OP_SEQUENCE_CAUSAL_CONV = 10,
  HIPPIHX_V1_OP_COMM_PCIE = 11,
  HIPPIHX_V1_OP_COUNT = 12,
} hippihx_v1_op_id;

typedef struct hippihx_v1_caps {
  const char* arch;  // e.g. "gfx1030"; required
  int wave;          // 32 / 64; 0 = unset
  int dtype;         // hippihx_v1_dtype; 0 = unset
} hippihx_v1_caps;

typedef struct hippihx_v1_scratch_spec {
  const char* name;
  size_t nbytes;
  int zeroed;  // always 1 — caller zeros for page-commit
} hippihx_v1_scratch_spec;

// Number of registered V1 ops (HIPPIHX_V1_OP_COUNT).
int hippihx_v1_op_count(void);

// Qualname like "gemm.exl3_3inst". NULL if id is out of range.
const char* hippihx_v1_op_name(hippihx_v1_op_id op);

// 1 if the op is a shared-DOT tile (FA / W4 / EXL3 / moe.shared).
int hippihx_v1_op_is_dot(hippihx_v1_op_id op);

// 1 if activations must be fp16 (DOT + GDN HIP). 0 for leftover / conv.
int hippihx_v1_op_fp16_act(hippihx_v1_op_id op);

// Host plan: fill out_specs[0..*inout_nspecs). On entry *inout_nspecs is
// capacity; on success it is the count written (currently always 1).
// Returns HIPPIHX_V1_OK or an ERR_* code. No device work. May be called
// under capture (host-only).
int hippihx_v1_plan(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                    hippihx_v1_scratch_spec* out_specs, size_t* inout_nspecs);

// Capture-safe enqueue stub. Validates scratch nbytes from plan.
// Returns HIPPIHX_V1_ERR_NOT_READY until the HIP body is migrated into
// the matching tiles/<class>/kernel.hip. Never allocates. Never D2H.
int hippihx_v1_run(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                   void* scratch, size_t scratch_nbytes);

// ABI revision for extras loaders (bump on breaking layout changes).
// Rev 2: caps.dtype + HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE (no fdot2.bf16).
// Rev 3: qualnames attn.* → attention.* (ids unchanged; extras has not bound).
enum { HIPPIHX_V1_ABI_REVISION = 3 };
int hippihx_v1_abi_revision(void);

#ifdef __cplusplus
}
#endif
