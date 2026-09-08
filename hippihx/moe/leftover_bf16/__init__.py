from hippihx._ops import make_op

_op = make_op("moe.leftover_bf16", "Leftover dense BF16 / cvt-fdot2")
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
