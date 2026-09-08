#include "hippihx/smoke.hpp"

#if defined(HIPPIHX_HOST_STUB)
#include "hippihx/host_hip_stub.hpp"
#else
#include <hip/hip_runtime.h>
#endif

// Tiny compile/link stub. Not a GEMM, not attention.
__global__ void hippihx_smoke_kernel(int* out) {
  if (out != nullptr) {
    *out = HIPPIHX_SMOKE_MAGIC;
  }
}

extern "C" int hippihx_smoke_magic(void) { return HIPPIHX_SMOKE_MAGIC; }
