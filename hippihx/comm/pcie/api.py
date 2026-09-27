from dataclasses import replace

from hippihx._lib.ops import export
from hippihx._lib.protocol import Caps, Plan
from hippihx.comm.pcie.policy import plan_meta

export(globals())

_plan = globals()["plan"]


def plan(caps=None, **params) -> Plan:
    """Uncached+push AR. PIX+ACS may custom-AR; PHB/PXB stay RCCL.

    With ``max_bytes`` this is the V1 rev 4 sized plan: a hop that is not
    custom-AR raises ``V1Error(ERR_UNSUPPORTED_FABRIC)``, as the C plan does.
    """
    p = _plan(caps, **params)
    c = caps if caps is not None else Caps()
    nbytes = int(p.params.get("max_bytes", 0))
    return replace(p, meta={**dict(p.meta), **plan_meta(c, nbytes=nbytes)})
