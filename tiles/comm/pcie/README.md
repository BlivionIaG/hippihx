# comm/pcie

Size-gated **Uncached + push** custom all-reduce on PCIe
(`hipDeviceMallocUncached`). Host fabric contracts:
`hippihx.comm.fabric.Fabric` + `hippihx.comm.pcie.policy`.

| Lock | Status |
|---|---|
| LDS bytes | TBD — lock here before the production kernel |
| `__launch_bounds__` | TBD — few blocks to fill PCIe, not an FA occupancy knob |
| Staging | Uncached + push; do not trust Finegrained on V620 |
| Wire class | **INT8 / Q8 preferred** |
| Leave | E4M3 / `f8_dma` without FP8 hardware; NTB; switch DMA |
| Graph | IPC scratch + device-resident seq; in/out stay local; scratch **zeroed**; **no D2H under capture** |
| Bind key | **arch + wave + hop + switch**. PIX on PEX/PLX **88096** may custom-AR. PHB/PXB and 8749 stay RCCL. |

Wire codec (INT8 / Q8) is a class, not a product name. RCCL remains the
fallback above `AR_MAX_KB=512` and off PIX. Not a DOT tile — Vega may
carry this stub; do not load FA/EXL3/AWQ objects into a gfx900 fatbin
just to ship AR. Do not mix gfx900 with gfx1030 on one 88096 host.

Mapped-peer MoE A2A (later) is BAR0 peer-store on the same PIX hop — not
IBGDA, not PEX switch DMA.

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
closed. `VLLM_RDNA_AR_MAX_KB` zoo lock **512**. C consume id:
`HIPPIHX_V1_OP_COMM_PCIE` (`include/hippihx/v1.h`) — stub only; do not
dump the ATen wrapper until extras rewires onto this V1 entry.

Dest extras @ `849292ec` / `4d25a048` made **vLLM/ROCm** custom AR
(not this Uncached+push tile) cudagraph-correct and default-on in the
gfx1030 launcher. That stay extras. Never `empty_like` on a real
forward. Do not dump `custom_all_reduce.py`. Do not copy tok/s.

**Later (not in dest squash):** leapdragon `3cfe000` moves host-coherent
flags **beside each receiving GPU’s uncached staging** (PCIe ordering /
poll traffic). extras draft [PR **#13**](https://github.com/opengfx1030/vllm-rdna/pull/13)
is a Cursor-authored T44b port of that layout plus a wedge check.
**Not dest.** Do not pick that PR. `VLLM_RDNA_AR` stays opt-in.
`VLLM_RDNA_AR_MAX_KB` zoo lock stays **512** (the draft would default
**64**). If dest locks T44b, pick unique Aron Hsiao commits — not the
Cursor rewrite. Do not dump the ATen `.cu`.
