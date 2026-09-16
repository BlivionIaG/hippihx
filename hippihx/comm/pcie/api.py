from hippihx._lib.ops import export
from hippihx._lib.protocol import Plan
from hippihx.comm.pcie.policy import plan_meta

export(globals())

_plan = plan


def plan(caps=None) -> Plan:
    """Uncached+push AR. PIX+ACS may custom-AR; PHB/PXB stay RCCL."""
    p = _plan(caps)
    c = caps if caps is not None else Caps()
    nbytes = p.specs[0].nbytes if p.specs else 0
    return Plan(
        qualname=p.qualname,
        arch=p.arch,
        specs=p.specs,
        meta={**dict(p.meta), **plan_meta(c, nbytes=nbytes)},
    )
