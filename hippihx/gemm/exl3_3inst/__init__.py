from hippihx._ops import make_op

_op = make_op(
    "gemm.exl3_3inst",
    "EXL3 3inst consume hook (shared DOT source; produce stays outside)",
    dot=True,
)
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
