"""W4A16 consume contract (host). Extracted so extras does not own pack math.

GPTQ and AWQ are pack/zeros modes of one GEMM. Produce stays outside hippihx.
Do not copy dest extras ``prep_zero_scale_fp16`` (scale-baked ZP).
"""

from __future__ import annotations

# GPTQ uint4b8 stores q+1. AWQ zeros are literal.
GPTQ_ZERO_OFFSET = 1
AWQ_ZERO_OFFSET = 0
K_STEP = 32
CONFIG_A_K_STEP = 32
CONFIG_H_K_STEP = 64  # dest-reverted; refuse


def zero_offset(*, use_v2_format: bool) -> int:
    """``use_v2_format`` is extras AWQ. GPTQ is +1, AWQ is literal 0."""
    return AWQ_ZERO_OFFSET if use_v2_format else GPTQ_ZERO_OFFSET


def dequant_nibble(q: int, zero: int, scale: float) -> float:
    """Integer ``q - zero``, then scale. Never bake ``scale * (-1024 - zero)``."""
    return float(q - zero) * scale


def k_per_split(k: int, split_k: int) -> int:
    """Equal splits, multiple of ``K_STEP``. Refuse dest K=640 → 16×40."""
    if split_k < 1:
        raise ValueError("split_k must be >= 1")
    if k % split_k != 0:
        raise ValueError(f"K={k} not divisible by split_k={split_k}")
    width = k // split_k
    if width % K_STEP != 0:
        raise ValueError(
            f"k_per_split={width} is not a multiple of K_STEP={K_STEP} "
            f"(dest extras can pick 40-wide; zoo refuses it)"
        )
    return width


def refuse_config_h(k_step: int) -> None:
    if k_step == CONFIG_H_K_STEP:
        raise ValueError("ConfigH K_STEP=64 is dest-reverted; keep ConfigA K_STEP=32")
    if k_step != CONFIG_A_K_STEP:
        raise ValueError(f"W4 consume K_STEP must be {CONFIG_A_K_STEP}, got {k_step}")
