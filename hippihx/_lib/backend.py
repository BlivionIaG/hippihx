"""Zoo backends: HIP fatbins, the FlyDSL compiler, and Mojo/MAX authoring.

HIP and FlyDSL are dest *zoo* backends (plan / bind / run). extras V1
consume is still the HIP fatbin (``hippihx_v1_*``). FlyDSL extras consume
waits on graph-safe JIT.

Mojo/MAX (``backend="mojo"``) is the maintainability authoring surface.
It is not dest-ready. ``MOJO_PRODUCE``, ``MOJO_V1_CONSUME``, and
``MOJO_DEST_READY`` stay false. Dest produce for vllm-rdna stays hipcc
7.14 / ``hipModuleLoad`` / ``libamdhip64`` until a documented HIP re-emit
or MAX-serve soak exists. The ABI gap is MAX ``execute`` versus
``hippihx_v1_*``. See ``docs/MOJO.md``.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    HIP = "hip"
    FLYDSL = "flydsl"
    MOJO = "mojo"


class MojoNotProduce(RuntimeError):
    """``run`` refused to enqueue a Mojo object.

    Mojo objects are not dest-ready. Authoring may plan and bind. Dest
    produce stays hipcc 7.14 / hipModuleLoad / libamdhip64 until a
    documented HIP re-emit or MAX-serve soak exists.
    """


MOJO_NOT_PRODUCE_MSG = (
    "Mojo objects are not dest-ready. Dest produce for vllm-rdna stays "
    "hipcc 7.14 / hipModuleLoad / libamdhip64 until a documented HIP "
    "re-emit or MAX-serve soak exists. ABI gap: MAX execute(OutputTensor, "
    "DeviceContext) is not hippihx_v1_plan / hippihx_v1_run"
)

FATBIN_BACKEND = Backend.HIP
DEST_BACKEND = FATBIN_BACKEND
FLYDSL_DEST = True
FLYDSL_V1_CONSUME = False

# Mojo/MAX is an authoring backend. It is not dest-ready and not a produce pin.
MOJO_AUTHORING = True
MOJO_PRODUCE = False
MOJO_V1_CONSUME = False
MOJO_DEST_READY = False

# Compiler + vec-add is dest (lab-tested). DOT wrappers / skinny / extras JIT next.
FLYDSL_GATE0_OBJECT = True
FLYDSL_GATE1_DOT_WRAPPERS = False
FLYDSL_GATE2_SKINNY = False
FLYDSL_GATE3_TABLE = False

FORBIDDEN_FLYDSL_KERNELS: tuple[str, ...] = (
    "preshuffle_gemm",
    "moe_gemm_2stage",
    "rdna3_f16_gemm",
    "rdna_f16_gemm",
    "rdna_fp8_preshuffle_gemm",
    "flash_attn_generic",
)


def is_zoo_backend(backend: Backend | str) -> bool:
    """Plan/bind backends. Includes Mojo authoring."""

    try:
        return Backend(backend) in (Backend.HIP, Backend.FLYDSL, Backend.MOJO)
    except ValueError:
        return False


def is_dest_backend(backend: Backend | str) -> bool:
    """Dest zoo backends. Mojo is authoring, not dest produce."""

    try:
        return Backend(backend) in (Backend.HIP, Backend.FLYDSL)
    except ValueError:
        return False


def is_v1_consume_backend(backend: Backend | str) -> bool:
    """extras ``hipModuleLoad`` path. HIP only, until a consume flag flips."""

    try:
        return Backend(backend) is Backend.HIP
    except ValueError:
        return False


def flydsl_ready() -> bool:
    """Graph-safe extras consume. Zoo Caps do not wait on this."""
    return FLYDSL_V1_CONSUME
