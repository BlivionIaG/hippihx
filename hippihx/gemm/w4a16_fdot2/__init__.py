from hippihx._ops import make_op

_op = make_op("gemm.w4a16_fdot2", "W4A16 nibble+ZP via fdot2 (GPTQ/AWQ pack modes)")
META = _op.META
Caps = _op.Caps
plan = _op.plan
bind = _op.bind
run = _op.run
is_supported = _op.is_supported
