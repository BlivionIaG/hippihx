"""Zoo backends: HIP fatbins, the FlyDSL compiler, and Mojo/MAX authoring.

HIP and FlyDSL are dest *zoo* backends (plan / bind / run). extras V1
consume is still the HIP fatbin (``hippihx_v1_*``). FlyDSL extras consume
waits on graph-safe JIT.

Mojo/MAX (``backend="mojo"``) is the maintainability authoring surface.
``MOJO_PRODUCE`` and ``MOJO_V1_CONSUME`` stay false: a Mojo object is not
loaded by ``hipModuleLoad``. Sandbox winners re-emit HIP (hipcc 7.14) or
an explicit MAX serve path is chosen. See ``docs/MOJO.md``.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    HIP = "hip"
    FLYDSL = "flydsl"
    MOJO = "mojo"


class MojoNotProduce(RuntimeError):
    """``run`` refused to enqueue a Mojo object.

    Authoring may plan and bind. Soak under vllm-rdna needs a HIP object
    (hipcc 7.14 → hipModuleLoad / libamdhip64) or an explicit MAX serve path.
    """


MOJO_NOT_PRODUCE_MSG = (
    "Mojo objects are not produce for vllm-rdna until re-emitted to HIP "
    "(hipcc 7.14 → hipModuleLoad / libamdhip64) or an explicit MAX serve "
    "path is chosen"
)

FATBIN_BACKEND = Backend.HIP
DEST_BACKEND = FATBIN_BACKEND
FLYDSL_DEST = True
FLYDSL_V1_CONSUME = False

# Mojo/MAX is an authoring backend. It is not a dest produce pin.
MOJO_AUTHORING = True
MOJO_PRODUCE = False
MOJO_V1_CONSUME = False

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
