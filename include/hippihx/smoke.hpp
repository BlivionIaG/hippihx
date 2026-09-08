#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum { HIPPIHX_SMOKE_MAGIC = 0x1030 };

// Host-callable symbol from tiles/smoke.hip. Used to prove the fatbin
// archive links; not a production kernel.
int hippihx_smoke_magic(void);

#ifdef __cplusplus
}
#endif
