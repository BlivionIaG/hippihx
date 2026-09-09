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

extras [PR #1](https://github.com/opengfx1030/vllm-rdna/pull/1)
(`cursor/leapdragon-cherry-d2a4`) squash-merged onto dest `rdna_extras`
@ `a4060647cfbb` (2026-09-08). Sources are now dest *presence*
(`csrc/rocm/rdna_allreduce.{cu,cuh}`): Uncached+push, host-coherent
flags, boot self-test, `VLLM_RDNA_AR_BLOCKS` / `_PACE` / `_MAX_KB`.
**Authorship of unique HIP is Aron Hsiao** `<leapdragon@gmail.com>` —
Author **and** Committer on every unique commit. Keep
`Co-Authored-By: Claude Fable 5`. Do not add Blivion/Cursor trailers.
Do **not** pick dest squash `a4060647` (Author BlivionIaG). **AR default
stays off** (`getenv("VLLM_RDNA_AR", "0") == "1"`). Occupancy pin stays
closed. `VLLM_RDNA_AR_MAX_KB` default **512**. C consume id:
`HIPPIHX_V1_OP_COMM_PCIE` (`include/hippihx/v1.h`) — stub only; do not
dump the ATen wrapper until extras rewires onto this V1 entry.
