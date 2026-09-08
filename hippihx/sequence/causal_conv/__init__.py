from hippihx._ops import make_op

_op = make_op(
    "sequence.causal_conv",
    "Short-window causal conv (scalar FMA, state_len≈4; not gdn_scan)",
    dot=False,
)
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
