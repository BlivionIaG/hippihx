# attn

Attention **classes**, not product names. Each child directory is one tile
contract. Serve wiring (`rdna_extras`) binds these; it does not own ISA.

| Directory | Class |
|---|---|
| `fa_fdot2` | Flash-attn / paged FA using `fdot2` (`v_dot2c`) |
| `gdn_scan` | GDN / hybrid linear-state scan |
| `kda_scan` | KDA linear-state scan (not GDN, not a fused 128×128 product kernel) |
| `qsa_indexer` | QSA / group-select indexer |
| `dsa_nope` | DSA MLA-NoPE (not V4 compressed Lightning) |

**Lock per tile, before ISA lands:** LDS bytes, `__launch_bounds__`, wave
size (gfx1030 = wave32), and whether the tile may use `fdot2`. No WMMA /
MFMA / FP8 hardware assumptions on gfx1030.
