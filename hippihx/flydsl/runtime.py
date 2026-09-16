"""Optional FlyDSL compiler. Import is cheap; the wheel is not required."""

from __future__ import annotations

from hippihx._lib.backend import FORBIDDEN_FLYDSL_KERNELS
from hippihx._lib.fatbin import DEFAULT_ARCH, DOT_WAVE


def available() -> bool:
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def require() -> None:
    if not available():
        raise ModuleNotFoundError(
            "FlyDSL is an optional extra. Install with: pip install hippihx[flydsl]"
        )


def env_arch(arch: str = DEFAULT_ARCH) -> dict[str, str]:
    return {"FLYDSL_GPU_ARCH": arch, "HIPPIHX_FLY_WAVE": str(DOT_WAVE)}


def refuse_shipped_kernel(name: str) -> None:
    """ROCm/FlyDSL MFMA/WMMA kernels are not dest on gfx1030."""
    key = name.strip().rsplit(".", 1)[-1]
    if key in FORBIDDEN_FLYDSL_KERNELS or name in FORBIDDEN_FLYDSL_KERNELS:
        raise ValueError(
            f"{name} is a ROCm/FlyDSL MFMA/WMMA kernel; not dest on gfx1030. "
            "Use hippihx.flydsl.atoms (fdot2/sdot4) instead."
        )
