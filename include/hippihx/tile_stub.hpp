#pragma once

#include "hippihx/arch.hpp"

#if defined(HIPPIHX_HOST_STUB)
#include "hippihx/host_hip_stub.hpp"
#else
#include <hip/hip_runtime.h>
#endif

// Shared prelude for tile-contract stubs. Each tile's kernel.hip owns a
// uniquely named empty entry so archives do not collide. Production tiles
// replace that entry and lock LDS / __launch_bounds__ in the tile README.
