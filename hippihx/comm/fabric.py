"""Re-export PCIe/PLX fabric contracts."""

from hippihx._lib.fabric import (
    HOPS,
    LEAVE_FABRIC,
    SWITCHES,
    SWITCH_ALIASES,
    Fabric,
    SwitchSku,
    mapped_peer_ok,
    normalize_switch,
    payload_gbs,
    refuse_leave,
    refuse_vega_mix,
    ring_n4_algbw_gbs,
)

__all__ = [
    "HOPS",
    "LEAVE_FABRIC",
    "SWITCHES",
    "SWITCH_ALIASES",
    "Fabric",
    "SwitchSku",
    "mapped_peer_ok",
    "normalize_switch",
    "payload_gbs",
    "refuse_leave",
    "refuse_vega_mix",
    "ring_n4_algbw_gbs",
]
