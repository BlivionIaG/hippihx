# sequence

Short-window / recurrent **helpers**, not scan tiles.

| Directory | Class |
|---|---|
| `causal_conv` | Scalar-FMA causal conv, `state_len≈4` |

GDN and KDA **scans** live under `attention/gdn_scan` and `attention/kda_scan`. Do not
put this conv under `gdn_scan`. GDN vs KDA state layouts differ — do not
retarget a GDN 16/48 window onto KDA 64×128.
