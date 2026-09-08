# comm/pcie

Stub only. **Later:** size-gated **Uncached + push** custom all-reduce on
PCIe (`hipDeviceMallocUncached`).

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — few blocks to fill PCIe, not an FA occupancy knob |
| Staging | Uncached + push; do not trust Finegrained on V620 |
| Wire class | **INT8 / Q8 preferred** |
| Leave | E4M3 / `f8_dma` without FP8 hardware — do not land FP8 wire on gfx1030 |
| Graph | IPC scratch + device-resident seq; in/out stay local; scratch **zeroed**; **no D2H under capture** |

Wire codec (INT8 / Q8) is a class, not a product name. RCCL remains the
fallback above the byte gate. Not a DOT tile — Vega may carry this stub;
do not load FA/EXL3/AWQ objects into a gfx900 fatbin just to ship AR.
