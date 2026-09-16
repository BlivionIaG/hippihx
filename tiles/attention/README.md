# attention

Attention **classes**, not product names. Each child directory is one tile
contract. Serve wiring (`rdna_extras`) binds these; it does not own ISA.

| Directory | Class |
|---|---|
| `fa_fdot2` | Flash-attn / paged FA using `fdot2` (`v_dot2c`) — **shared DOT source** (gfx1030 + gfx1100) |
| `gdn_scan` | GDN / hybrid linear-state scan |
| `kda_scan` | KDA linear-state scan (not GDN, not a fused 128×128 product kernel) |
| `qsa_indexer` | QSA / group-select indexer |
| `dsa_nope` | DSA MLA-NoPE (not V4 compressed Lightning) |

**Lock per tile, before ISA lands:** LDS bytes, `__launch_bounds__`, wave
size (**32** on DOT tiles), and whether the tile may use `fdot2`. No
`fdot2.bf16`. Never `#ifdef WMMA` on shared DOT paths. GDN vs KDA layouts
differ — do not retarget GDN 16/48 onto KDA 64×128. Short-window conv is
`sequence/causal_conv`, not under `gdn_scan`.
