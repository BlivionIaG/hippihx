"""Uncached+push AR policy. extras imports this; do not dump ``rdna_allreduce.cu``.

Custom AR is PIX-only, size-gated. PHB/PXB stay RCCL. Occupancy pin closed.
"""

from __future__ import annotations

from typing import Any

from hippihx._lib.fabric import Fabric, mapped_peer_ok, refuse_leave

AR_MAX_KB = 512
STAGING = "uncached_push"
WIRE = "int8_q8"
RCCL_FALLBACK = "rccl"


def custom_ar_ok(fabric: Fabric | None, *, nbytes: int = 0) -> bool:
    if fabric is None:
        return False
    if fabric.hop != "pix" or not fabric.acs_clear or not fabric.large_bar:
        return False
    if fabric.switch == "pex8749":
        return False  # Gen3 SKU; same class, not dest AR mesh
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    if nbytes > AR_MAX_KB * 1024:
        return False
    return True


def plan_meta(caps: Any, *, nbytes: int = 0) -> dict[str, Any]:
    fabric: Fabric | None = getattr(caps, "fabric", None)
    ar = custom_ar_ok(fabric, nbytes=nbytes)
    meta: dict[str, Any] = {
        "staging": STAGING,
        "wire": WIRE,
        "ar_max_kb": AR_MAX_KB,
        "custom_ar": ar,
        "fallback": RCCL_FALLBACK,
        "mapped_peer": mapped_peer_ok(fabric),
        "hop": None if fabric is None else fabric.hop,
        "switch": None if fabric is None else fabric.switch,
        "bind_key": None
        if fabric is None
        else fabric.bind_key(caps.arch, caps.wave),
    }
    if not ar:
        meta["reason"] = (
            "no PIX fabric"
            if fabric is None
            else f"hop={fabric.hop} switch={fabric.switch} (RCCL until PIX+ACS+large-BAR)"
        )
    return meta


def refuse_finegrained() -> None:
    refuse_leave("finegrained")


def refuse_e4m3_wire() -> None:
    refuse_leave("e4m3")
