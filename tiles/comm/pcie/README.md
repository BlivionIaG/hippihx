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
(`csrc/rocm/rdna_allreduce.{cu,cuh}`): Uncached+push, boot self-test,
`VLLM_RDNA_AR_BLOCKS` / `_PACE` / `_MAX_KB`. Dest extras @ `3b59ee16`
(merged PR **#13**) moved flags into uncached VRAM and added the T44b
wedge check. **AR default stays off.** Occupancy pin stays closed.
`VLLM_RDNA_AR_MAX_KB` zoo lock **512** (dest extras default is now
**64**). **Authorship of unique HIP is Aron Hsiao**
`<leapdragon@gmail.com>` — Author **and** Committer on unique commits.
Do **not** pick dest squash `a4060647` or `3b59ee16` (Cursor rewrite).
C consume id: `HIPPIHX_V1_OP_COMM_PCIE` — stub only; do not dump the
ATen wrapper until extras rewires onto this V1 entry.

Dest extras @ `849292ec` / `4d25a048` made **vLLM/ROCm** custom AR
(not this Uncached+push tile) cudagraph-correct and default-on in the
gfx1030 launcher. That stay extras. Never `empty_like` on a real
forward. Do not dump `custom_all_reduce.py`. Do not copy tok/s.
