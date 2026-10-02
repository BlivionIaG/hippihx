# comm

Collectives. `pcie/` is Uncached+push AR. Hop class (PIX / PXB / PHB) and
switch SKU (PEX/PLX **88096**, 8749, similar) live in
`hippihx.comm.fabric` — not a second V1 op.

RoCE stays out of this tree. A gfx1200 DOT fatbin is not a comm path.
PIX helpers (`lspci` / ACS) stay
extras / `v620_toolbox`. The zoo owns the bind key so kernels can
specialize without serve inventing hop policy.
