#pragma once

// Stable C consume ABI for rdna_extras (and any serve layer). Rev 4.
//
// One V1 entry family per op. Serve wraps these as torch.ops.hippihx.* —
// hippihx itself stays torch-free, and this header stays HIP-free.
//
// Artifacts: libhippihx_v1.so (these symbols and tables, no device code,
// one build for every slot) plus one hippihx_<arch>.hsaco per slot (a raw
// AMDGPU ELF, one --offload-arch).
//
// Flow (docs/CONSUME.md):
//   0. hippihx_v1_load at serve init, once per process: checks the code
//      object's arch and loads it with hipModuleLoadData.
//   1. hippihx_v1_plan: host-only, before capture, once per capture bucket
//      and layer config. It checks caps and params, sizes scratch, picks the
//      explore variant and reports ready. ready == 0 means keep the serve
//      fallback. Decide that before capture, never inside it.
//   2. Serve allocates plan.scratch_nbytes (or a slice of a shared arena at
//      a HIPPIHX_V1_SCRATCH_ALIGN offset) and zeros it once for page-commit.
//   3. hippihx_v1_run: capture-safe enqueue on the given stream. It never
//      allocates, never reads device memory, never syncs.
//
// Params are host scalars and a captured graph bakes them in. Anything that
// changes between replays (sequence lengths, block tables, query offsets) is
// a tensor the kernel reads on device, never a param.
//
// The enums below and the tables in tiles/v1_abi.cpp are generated from
// hippihx/_lib/catalog.py (python -m hippihx._lib.codegen). Bodies still
// live in extras until extras rewires onto these symbols. Until then every
// plan has ready == 0 and hippihx_v1_run checks its arguments, then returns
// HIPPIHX_V1_ERR_NOT_READY.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes. Append only.
// hippihx:gen begin v1_status -- python -m hippihx._lib.codegen; do not edit
enum {
  HIPPIHX_V1_OK = 0,
  HIPPIHX_V1_ERR_UNKNOWN_OP = 1,
  HIPPIHX_V1_ERR_UNSUPPORTED_ARCH = 2,  // slot not built for this op, or wave not allowed
  HIPPIHX_V1_ERR_BAD_ARG = 3,  // NULL pointer or malformed fabric
  HIPPIHX_V1_ERR_SCRATCH = 4,  // scratch too small, NULL, or misaligned
  HIPPIHX_V1_ERR_NOT_READY = 5,  // contract exists; HIP body not migrated
  HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE = 6,  // e.g. bf16 on fdot2 / GDN HIP
  HIPPIHX_V1_ERR_ABI = 7,  // struct_size / revision mismatch, or a failed plan
  HIPPIHX_V1_ERR_PARAM = 8,  // param count, domain, cross-check, or scratch overflow
  HIPPIHX_V1_ERR_TENSOR = 9,  // tensor count, dtype, rank, extent, layout, or NULL data
  HIPPIHX_V1_ERR_UNSUPPORTED_FABRIC = 10,  // comm: not a custom-AR hop; serve keeps RCCL
  HIPPIHX_V1_ERR_CODE_OBJECT = 11,  // not one raw AMDGPU ELF code object, or unreadable
  HIPPIHX_V1_ERR_FOREIGN_ISA = 12,  // code object mach is not the slot's, or HSA_OVERRIDE set
  HIPPIHX_V1_ERR_NO_HIP = 13,  // library built without HIP (host stub); nothing loaded
};
// hippihx:gen end v1_status

// hippihx:gen begin v1_limits -- python -m hippihx._lib.codegen; do not edit
#define HIPPIHX_V1_MAX_RANK 6
#define HIPPIHX_V1_MAX_PARAMS 16
#define HIPPIHX_V1_MAX_SCRATCH 4
#define HIPPIHX_V1_SCRATCH_ALIGN 256
// hippihx:gen end v1_limits

// Element dtype. caps.dtype is the activation dtype: 0 = unset (do not
// refuse), and only FP16 / BF16 / FP32 are activation dtypes. DOT and GDN
// HIP are fp16; never fdot2.bf16 (gfx1030 LLVM ISel abort). Tensor
// descriptors use the whole list.
// hippihx:gen begin v1_dtype -- python -m hippihx._lib.codegen; do not edit
typedef enum hippihx_v1_dtype {
  HIPPIHX_V1_DTYPE_UNSET = 0,
  HIPPIHX_V1_DTYPE_FP16 = 1,
  HIPPIHX_V1_DTYPE_BF16 = 2,
  HIPPIHX_V1_DTYPE_FP32 = 3,
  HIPPIHX_V1_DTYPE_I32 = 4,
  HIPPIHX_V1_DTYPE_I64 = 5,
  HIPPIHX_V1_DTYPE_U8 = 6,
  HIPPIHX_V1_DTYPE_I8 = 7,
} hippihx_v1_dtype;
// hippihx:gen end v1_dtype

// Built fatbin slots (hippihx._lib.fatbin.KNOWN_ARCHES order). plan resolves
// caps.arch to one of these; run never compares strings. Later slots
// (gfx906, gfx1013) are not slots.
// hippihx:gen begin v1_arch -- python -m hippihx._lib.codegen; do not edit
typedef enum hippihx_v1_arch {
  HIPPIHX_V1_ARCH_GFX1030 = 0,
  HIPPIHX_V1_ARCH_GFX1100 = 1,
  HIPPIHX_V1_ARCH_GFX1101 = 2,
  HIPPIHX_V1_ARCH_GFX1102 = 3,
  HIPPIHX_V1_ARCH_GFX1200 = 4,
  HIPPIHX_V1_ARCH_GFX1151 = 5,
  HIPPIHX_V1_ARCH_GFX1031 = 6,
  HIPPIHX_V1_ARCH_GFX1032 = 7,
  HIPPIHX_V1_ARCH_GFX1033 = 8,
  HIPPIHX_V1_ARCH_GFX1035 = 9,
  HIPPIHX_V1_ARCH_GFX1036 = 10,
  HIPPIHX_V1_ARCH_GFX900 = 11,
  HIPPIHX_V1_ARCH_COUNT = 12,
} hippihx_v1_arch;
// hippihx:gen end v1_arch

// PCIe fabric classes for comm plans (hippihx._lib.fabric). 0 = unset.
// hippihx:gen begin v1_fabric -- python -m hippihx._lib.codegen; do not edit
typedef enum hippihx_v1_hop {
  HIPPIHX_V1_HOP_UNSET = 0,
  HIPPIHX_V1_HOP_PIX = 1,
  HIPPIHX_V1_HOP_PXB = 2,
  HIPPIHX_V1_HOP_PHB = 3,
} hippihx_v1_hop;

typedef enum hippihx_v1_switch {
  HIPPIHX_V1_SWITCH_UNSET = 0,
  HIPPIHX_V1_SWITCH_PEX88096 = 1,
  HIPPIHX_V1_SWITCH_PEX8749 = 2,
  HIPPIHX_V1_SWITCH_GENERIC = 3,
} hippihx_v1_switch;
// hippihx:gen end v1_fabric

// Stable ids — match hippihx.list_ops() order. Do not renumber.
// hippihx:gen begin v1_ops -- python -m hippihx._lib.codegen; do not edit
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
// hippihx:gen end v1_ops

// Per-op schema, pinned one op at a time. _P_* indexes params, _T_* indexes
// tensor slots. An op with no params block takes nparams == 0; an op with
// no slots block takes ntensors == 0.
// hippihx:gen begin v1_schema -- python -m hippihx._lib.codegen; do not edit
// attention.fa_fdot2 params, in order.
enum {
  HIPPIHX_V1_ATTN_FA_FDOT2_P_MODE = 0,  // enum decode=0 prefill=1: decode: split-K + combine; prefill: paged varlen
  HIPPIHX_V1_ATTN_FA_FDOT2_P_HEAD_DIM = 1,  // enum 128|256: D; extras ships D=128 and D=256
  HIPPIHX_V1_ATTN_FA_FDOT2_P_NUM_Q_HEADS = 2,  // int >= 1: H_q
  HIPPIHX_V1_ATTN_FA_FDOT2_P_NUM_KV_HEADS = 3,  // int >= 1: H_kv; GQA group is H_q / H_kv
  HIPPIHX_V1_ATTN_FA_FDOT2_P_BLOCK_SIZE = 4,  // int >= 1: KV page size in tokens
  HIPPIHX_V1_ATTN_FA_FDOT2_P_KV_SPLITS = 5,  // int [1, 16]: split-K partitions (extras MAX_SPLITS=16)
  HIPPIHX_V1_ATTN_FA_FDOT2_P_SLIDING_WINDOW = 6,  // int >= 0: 0 = no window
  HIPPIHX_V1_ATTN_FA_FDOT2_P_CAUSAL = 7,  // bool 0/1: prefill mask; decode ignores it
  HIPPIHX_V1_ATTN_FA_FDOT2_P_MAX_TOKENS = 8,  // int >= 1: bound on q rows for this plan (the capture bucket); sizes the partials
  HIPPIHX_V1_ATTN_FA_FDOT2_P_SCALE = 9,  // float > 0: softmax scale; extras passes 1/sqrt(D)
  HIPPIHX_V1_ATTN_FA_FDOT2_NPARAMS = 10,
};
enum {
  HIPPIHX_V1_ATTN_FA_FDOT2_MODE_DECODE = 0,
  HIPPIHX_V1_ATTN_FA_FDOT2_MODE_PREFILL = 1,
};
// attention.fa_fdot2 tensor slots, in order.
enum {
  HIPPIHX_V1_ATTN_FA_FDOT2_T_Q = 0,  // in fp16 [<=max_tokens, num_q_heads, head_dim] contiguous
  HIPPIHX_V1_ATTN_FA_FDOT2_T_K_CACHE = 1,  // in fp16 [*, num_kv_heads, *, block_size, *] strided
  HIPPIHX_V1_ATTN_FA_FDOT2_T_V_CACHE = 2,  // in fp16 [*, *, *, *, *] strided
  HIPPIHX_V1_ATTN_FA_FDOT2_T_BLOCK_TABLE = 3,  // in i32 [*, *] rows
  HIPPIHX_V1_ATTN_FA_FDOT2_T_SEQ_LENS = 4,  // in i32 [*] contiguous
  HIPPIHX_V1_ATTN_FA_FDOT2_T_CU_QUERY_LENS = 5,  // in i32 [*] contiguous when mode == prefill
  HIPPIHX_V1_ATTN_FA_FDOT2_T_OUT = 6,  // out fp16 [<=max_tokens, num_q_heads, head_dim] contiguous
  HIPPIHX_V1_ATTN_FA_FDOT2_NTENSORS = 7,
};

// comm.pcie params, in order.
enum {
  HIPPIHX_V1_COMM_PCIE_P_MAX_BYTES = 0,  // int [1, 524288]: largest all-reduce payload on this plan; RCCL above AR_MAX_KB
  HIPPIHX_V1_COMM_PCIE_NPARAMS = 1,
};
// hippihx:gen end v1_schema

typedef union hippihx_v1_param {
  int64_t i;  // int / enum / bool params
  double f;   // float params (finite, > 0)
} hippihx_v1_param;

typedef struct hippihx_v1_fabric {
  int32_t hop;          // hippihx_v1_hop
  int32_t pcie_switch;  // hippihx_v1_switch
  int32_t width;        // PCIe link width: 1, 2, 4, 8 or 16
  int32_t acs_clear;    // 1 = ACS cleared (required on a PIX hop)
  int32_t large_bar;    // 1 = large BAR
} hippihx_v1_fabric;

typedef struct hippihx_v1_caps {
  const char* arch;                 // e.g. "gfx1030"; required, one slot
  int wave;                         // 32 / 64; 0 = the slot default
  int dtype;                        // activation hippihx_v1_dtype; 0 = unset
  const hippihx_v1_fabric* fabric;  // comm plans; NULL = no fabric
} hippihx_v1_caps;

typedef struct hippihx_v1_tensor {
  void* data;                           // device pointer; NULL = slot not used
  int32_t dtype;                        // hippihx_v1_dtype
  int32_t rank;                         // 1 .. HIPPIHX_V1_MAX_RANK
  int64_t shape[HIPPIHX_V1_MAX_RANK];
  int64_t stride[HIPPIHX_V1_MAX_RANK];  // in elements (torch convention)
} hippihx_v1_tensor;

typedef struct hippihx_v1_scratch_spec {
  const char* name;  // library-owned static string
  size_t nbytes;
  size_t offset;     // from the plan's scratch base; HIPPIHX_V1_SCRATCH_ALIGN multiple
  int zeroed;        // always 1 — serve zeros once for page-commit
} hippihx_v1_scratch_spec;

// Caller-owned plan (_t: the function owns the plain name). Set
// struct_size = sizeof(hippihx_v1_plan_t) before hippihx_v1_plan. Plain
// data: copy it and cache it per capture bucket, but never edit it (plan
// again instead). A failed plan is zeroed and cannot run.
typedef struct hippihx_v1_plan_t {
  size_t struct_size;
  int32_t abi_revision;  // HIPPIHX_V1_ABI_REVISION
  int32_t op;            // hippihx_v1_op_id
  int32_t arch;          // hippihx_v1_arch
  int32_t wave;          // resolved: 32 or 64
  int32_t dtype;         // activation dtype from caps
  int32_t ready;         // 1 = body migrated and slot loaded; 0 = serve fallback
  int32_t variant;       // explore rank picked (hippihx.mojo.ranked_explore)
  int32_t nparams;
  hippihx_v1_param params[HIPPIHX_V1_MAX_PARAMS];
  int32_t nscratch;
  hippihx_v1_scratch_spec scratch[HIPPIHX_V1_MAX_SCRATCH];
  size_t scratch_nbytes;  // bytes to allocate; HIPPIHX_V1_SCRATCH_ALIGN multiple
} hippihx_v1_plan_t;

// Number of registered V1 ops (HIPPIHX_V1_OP_COUNT).
int hippihx_v1_op_count(void);

// Qualname like "gemm.exl3_3inst". NULL if id is out of range.
const char* hippihx_v1_op_name(hippihx_v1_op_id op);

// 1 if the op is a shared-DOT tile (FA / W4 / EXL3 / moe.shared).
int hippihx_v1_op_is_dot(hippihx_v1_op_id op);

// 1 if activations must be fp16 (DOT + GDN HIP). 0 for leftover / conv.
int hippihx_v1_op_fp16_act(hippihx_v1_op_id op);

// Schema sizes and names, in catalog order. -1 / NULL when out of range.
int hippihx_v1_op_nparams(hippihx_v1_op_id op);
const char* hippihx_v1_op_param_name(hippihx_v1_op_id op, int index);
int hippihx_v1_op_ntensors(hippihx_v1_op_id op);
const char* hippihx_v1_op_tensor_name(hippihx_v1_op_id op, int index);

// Load this slot's code object. Host-only, at serve init, single-threaded,
// before any plan or capture; it stays loaded for the process (one worker
// per GPU, current HIP device). A second load of a loaded slot is a no-op.
// Returns HIPPIHX_V1_OK, or UNSUPPORTED_ARCH (not a built slot),
// FOREIGN_ISA (HSA_OVERRIDE_GFX_VERSION is set, or the ELF mach is not the
// slot's), CODE_OBJECT (unreadable, not one raw AMDGPU ELF, or rejected by
// HIP), BAD_ARG, or NO_HIP (library built without HIP; nothing loaded).
int hippihx_v1_load(const char* arch, const char* path);
int hippihx_v1_load_image(const char* arch, const void* image, size_t nbytes);

// 1 when this slot has a loaded code object.
int hippihx_v1_loaded(const char* arch);

// Host plan. No device work, no allocation, safe under capture. params is
// the op's _P_* order and nparams its _NPARAMS (0 when unpinned). Returns
// HIPPIHX_V1_OK, or UNKNOWN_OP, BAD_ARG (NULL caps / out / params, bad
// fabric), ABI (out->struct_size), UNSUPPORTED_ARCH (slot, or wave for the
// op), UNSUPPORTED_DTYPE, PARAM, or UNSUPPORTED_FABRIC (comm: serve keeps
// RCCL).
int hippihx_v1_plan(hippihx_v1_op_id op, const hippihx_v1_caps* caps,
                    const hippihx_v1_param* params, size_t nparams,
                    hippihx_v1_plan_t* out);

// Capture-safe enqueue on stream (a hipStream_t; NULL is the null stream).
// tensors is the op's _T_* order and ntensors its _NTENSORS (0 when
// unpinned). Checks the plan, each required tensor (data, dtype, rank,
// extents against params, layout) and scratch (size, NULL, alignment to
// HIPPIHX_V1_SCRATCH_ALIGN). Never allocates. Never D2H. Returns NOT_READY
// while plan->ready == 0, which is every op today.
int hippihx_v1_run(const hippihx_v1_plan_t* plan,
                   const hippihx_v1_tensor* tensors, size_t ntensors,
                   void* scratch, size_t scratch_nbytes, void* stream);

// ABI revision for extras loaders (bump on breaking layout changes).
// Rev 2: caps.dtype + HIPPIHX_V1_ERR_UNSUPPORTED_DTYPE (no fdot2.bf16).
// Rev 3: qualnames attn.* → attention.* (ids unchanged; extras has not bound).
// Rev 4: caller-owned plan (ready, variant, scratch offsets), params, tensor
//        descriptors, stream, caps.fabric (extras has not bound).
// hippihx:gen begin v1_revision -- python -m hippihx._lib.codegen; do not edit
enum { HIPPIHX_V1_ABI_REVISION = 4 };
// hippihx:gen end v1_revision
int hippihx_v1_abi_revision(void);

#ifdef __cplusplus
}
#endif
