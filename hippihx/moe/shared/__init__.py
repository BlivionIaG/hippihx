from hippihx._ops import make_op

_op = make_op(
    "moe.shared",
    "Shared expert path (shared gfx1030+gfx1100 DOT source)",
    dot=True,
)
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
