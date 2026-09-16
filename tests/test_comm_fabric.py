from __future__ import annotations

import pytest

from hippihx.comm.fabric import (
    Fabric,
    mapped_peer_ok,
    payload_gbs,
    refuse_leave,
    refuse_vega_mix,
    ring_n4_algbw_gbs,
)
from hippihx.comm.pcie import policy
from hippihx.comm.pcie.policy import AR_MAX_KB
from hippihx.protocol import Caps


def test_88096_pix_payload_math() -> None:
    assert round(payload_gbs(gen=4, width=16), 3) == 31.508
    assert round(ring_n4_algbw_gbs(gen=4, width=16), 3) == 21.005
    assert round(payload_gbs(gen=3, width=16), 3) == 15.754
    pix = Fabric(hop="pix", switch="plx88096")
    assert pix.switch == "pex88096"
    assert pix.sku.data_lanes == 96
    assert pix.sku.x16_slots == 5
    assert mapped_peer_ok(pix)
    assert policy.custom_ar_ok(pix)
    assert not policy.custom_ar_ok(pix, nbytes=AR_MAX_KB * 1024 + 1)


def test_phb_and_8749_are_rccl() -> None:
    phb = Fabric(hop="phb", switch="88096")
    assert not policy.custom_ar_ok(phb)
    assert not mapped_peer_ok(phb)
    pxb = Fabric(hop="pxb", switch="pex88096")
    assert not policy.custom_ar_ok(pxb)
    gen3 = Fabric(hop="pix", switch="8749")
    assert gen3.gen == 3
    assert not policy.custom_ar_ok(gen3)
    assert not mapped_peer_ok(gen3)


def test_pix_requires_acs_clear() -> None:
    with pytest.raises(ValueError, match="ACS"):
        Fabric(hop="pix", switch="pex88096", acs_clear=False)
    with pytest.raises(ValueError, match="Leave"):
        refuse_leave("ntb")
    refuse_vega_mix("gfx1030", "gfx1100")
    with pytest.raises(ValueError, match="mix"):
        refuse_vega_mix("gfx1030", "gfx900")
    with pytest.raises(ValueError, match="Leave"):
        policy.refuse_finegrained()
    with pytest.raises(ValueError, match="e4m3"):
        policy.refuse_e4m3_wire()


def test_comm_pcie_plan_keys_on_hop() -> None:
    from hippihx.comm import pcie

    none = pcie.plan(Caps(arch="gfx1030"))
    assert none.meta["custom_ar"] is False
    assert none.meta["staging"] == "uncached_push"
    assert none.meta["wire"] == "int8_q8"
    pix = pcie.plan(Caps(arch="gfx1030", fabric=Fabric(hop="pix")))
    assert pix.meta["custom_ar"] is True
    assert pix.meta["mapped_peer"] is True
    assert pix.meta["bind_key"] == ("gfx1030", 32, "pix", "pex88096")
    phb = pcie.plan(Caps(arch="gfx1030", fabric=Fabric(hop="phb")))
    assert phb.meta["custom_ar"] is False
    assert phb.meta["fallback"] == "rccl"
