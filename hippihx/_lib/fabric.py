"""PCIe P2P / PLX fabric contracts. Host-only. No measured GB/s.

V620 has no XGMI. TP all-reduce and mapped-peer A2A are BAR0 TLPs through
a packet switch. Dest mesh is PEX/PLX **88096** (Gen4, 96 data lanes).
**8749** and similar fan-outs are the same *class* (PIX vs PHB/PXB),
not a second ISA.

Wiki: ``engine/plx.md``, ``silicon/plx-p2p-mmio.md``. Toolbox benches
stay in ``v620_toolbox/pcie_p2p``. extras PIX helpers stay serve.
"""

from __future__ import annotations

from dataclasses import dataclass

HOPS: tuple[str, ...] = ("pix", "pxb", "phb")
SWITCH_ALIASES: dict[str, str] = {
    "pex88096": "pex88096",
    "plx88096": "pex88096",
    "88096": "pex88096",
    "pex8749": "pex8749",
    "plx8749": "pex8749",
    "8749": "pex8749",
    "generic": "generic",
}

# Payload one way, 128b/130b. Math ceiling, not a hipMemcpyPeer measurement.
_GT = {3: 8, 4: 16, 5: 32}


@dataclass(frozen=True, slots=True)
class SwitchSku:
    name: str
    gen: int
    data_lanes: int
    x16_slots: int  # GPU x16 ports that fit with one CPU USP, 0 = unknown


SWITCHES: dict[str, SwitchSku] = {
    "pex88096": SwitchSku("pex88096", gen=4, data_lanes=96, x16_slots=5),
    "pex8749": SwitchSku("pex8749", gen=3, data_lanes=48, x16_slots=2),
    "generic": SwitchSku("generic", gen=4, data_lanes=0, x16_slots=0),
}

# Leave: not the TP / mapped-peer path.
LEAVE_FABRIC: tuple[str, ...] = (
    "ntb",
    "switch_dma",
    "finegrained",
    "e4m3",
    "hsa_override",
    "pcie_acs_override",  # kernel patch; dest is setpci ECAP_ACS+0x6
)


def normalize_switch(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    if key not in SWITCH_ALIASES:
        known = tuple(sorted(set(SWITCH_ALIASES.values())))
        raise ValueError(f"unknown switch {name!r}; known {known}")
    return SWITCH_ALIASES[key]


def payload_gbs(*, gen: int, width: int) -> float:
    """PCIe payload GB/s one way (128b/130b). Not measured busbw."""
    if gen not in _GT:
        raise ValueError(f"unknown PCIe gen {gen}; known {tuple(_GT)}")
    if width not in (1, 2, 4, 8, 16):
        raise ValueError(f"width must be a xN link, got {width}")
    return _GT[gen] * width * 128 / 130 / 8


def ring_n4_algbw_gbs(*, gen: int, width: int) -> float:
    """Ring n=4 algbw ceiling at 1.5 S/B. Math only."""
    return payload_gbs(gen=gen, width=width) / 1.5


@dataclass(frozen=True, slots=True)
class Fabric:
    """Hop class for comm plan/bind. Serve keys on arch + wave + hop + switch."""

    hop: str
    switch: str = "pex88096"
    width: int = 16
    acs_clear: bool = True
    large_bar: bool = True

    def __post_init__(self) -> None:
        hop = self.hop.strip().lower()
        if hop not in HOPS:
            raise ValueError(f"unknown hop {self.hop!r}; known {HOPS}")
        object.__setattr__(self, "hop", hop)
        object.__setattr__(self, "switch", normalize_switch(self.switch))
        sku = SWITCHES[self.switch]
        if self.width not in (1, 2, 4, 8, 16):
            raise ValueError(f"width must be a xN link, got {self.width}")
        if self.switch == "pex8749" and self.width == 16 and sku.x16_slots < 5:
            # 8749 cannot host a 5-slot x16 backplane. x16 endpoint is still valid.
            pass
        if not self.acs_clear and hop == "pix":
            raise ValueError(
                "PIX requires ACS clear (setpci ECAP_ACS+0x6.w=0000). "
                "SrcValid+ is a host bounce, not a dest hop."
            )

    @property
    def sku(self) -> SwitchSku:
        return SWITCHES[self.switch]

    @property
    def gen(self) -> int:
        return self.sku.gen

    def payload_gbs(self) -> float:
        return payload_gbs(gen=self.gen, width=self.width)

    def bind_key(self, arch: str, wave: int | None) -> tuple[str, int | None, str, str]:
        return (arch, wave, self.hop, self.switch)


def refuse_leave(feature: str) -> None:
    key = feature.strip().lower()
    if key in LEAVE_FABRIC:
        raise ValueError(
            f"{feature} is Leave on this mesh: NTB/switch-DMA/Finegrained/"
            "E4M3/HSA_OVERRIDE/pcie_acs_override are not dest. "
            "PIX is ACS-cleared BAR0 cut-through."
        )


def refuse_vega_mix(*arches: str) -> None:
    names = {a.lower() for a in arches}
    has_vega = "gfx900" in names or "gfx906" in names
    has_rdna = any(a.startswith("gfx10") or a.startswith("gfx11") for a in names)
    if has_vega and has_rdna:
        raise ValueError(
            "do not mix gfx900/gfx906 with gfx1030 on one 88096/ROCm host "
            "(HSA remapped MMIO trap)"
        )


def mapped_peer_ok(fabric: Fabric | None) -> bool:
    """BAR0 peer-store scatter. Not switch DMA, not NTB, not IBGDA."""
    if fabric is None:
        return False
    return (
        fabric.hop == "pix"
        and fabric.acs_clear
        and fabric.large_bar
        and fabric.switch in {"pex88096", "generic"}
    )
