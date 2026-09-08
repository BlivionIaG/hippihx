# comm/pcie

Stub only. Later: size-gated **push** custom all-reduce on PCIe, Uncached
staging (`hipDeviceMallocUncached`) + INT8/Q8 **wire class**.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — few blocks to fill PCIe, not an FA occupancy knob |
| Staging | Uncached + push; do not trust Finegrained on V620 |
| Graph | IPC scratch + device-resident seq; in/out stay local; no D2H under capture |

Wire codec (INT8 / Q8) is a class, not a product name. RCCL remains the
fallback above the byte gate.
