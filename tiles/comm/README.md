# comm

Collectives. `pcie/` is Uncached+push AR. Hop class (PIX / PXB / PHB) and
switch SKU (PEX/PLX **88096**, 8749, similar) live in
`hippihx.comm.fabric` — not a second V1 op.

RoCE / gfx12 HIP are out of this tree. PIX helpers (`lspci` / ACS) stay
extras / `v620_toolbox`. The zoo owns the bind key so kernels can
specialize without serve inventing hop policy.
