# attn/gdn_scan

GDN / hybrid-state scan class (Qwen-style linear attention state). Distinct
from `kda_scan`.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — lock here before the production kernel |
| Wave | 32 on gfx1030 |
| Graph | allocate zeroed state once; never per-step D2H under capture |

Not a product fuse. Not `gdn_decode_rdna2` by another name — this is the
hippihx contract the serve layer will call.
