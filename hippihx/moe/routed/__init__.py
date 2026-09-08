from hippihx._ops import make_op

_op = make_op("moe.routed", "Routed expert gate / up / down")
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
