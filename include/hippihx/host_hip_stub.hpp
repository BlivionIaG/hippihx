#pragma once

// Host-only stand-ins so `.hip` stubs compile as C++ when hipcc is absent.
// Not an ABI, not a runtime, not a substitute for ROCm.

#ifndef HIPPIHX_HOST_HIP_STUB_HPP
#define HIPPIHX_HOST_HIP_STUB_HPP

#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

#endif
