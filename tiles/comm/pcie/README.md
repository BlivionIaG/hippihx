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

extras PR #1 (`cursor/leapdragon-cherry-d2a4`) is a **review-only**
cherry-pick of leapdragon `rdna_ar` (Uncached+push, boot self-test).
**Authorship is inviolable: Aron Hsiao** `<leapdragon@gmail.com>` —
Author **and** Committer on every unique commit. Keep
`Co-Authored-By: Claude Fable 5`. Do **not** `GIT_AUTHOR_NAME=BlivionIaG`.
Do **not** `--reset-author`. Being on dest extras does **not** make this
this lab. Do not add Blivion/Cursor trailers. **AR default stays off.**
Occupancy pin stays closed. Do not merge that ISA into this stub until
extras dest-locks it.
