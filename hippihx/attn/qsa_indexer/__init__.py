from hippihx._ops import make_op

_op = make_op("attn.qsa_indexer", "QSA / group-select indexer")
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
