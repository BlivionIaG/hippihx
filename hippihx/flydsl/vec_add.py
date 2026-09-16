"""Dest FlyDSL kernel: gfx1030 vector-add.

Adapted from ``ROCm/FlyDSL`` ``examples/01-vectorAdd.py`` (Apache-2.0,
FlyDSL Project Contributors). Target-neutral ``flydsl.expr``; wave32 on
RDNA. Import is cheap — the FlyDSL wheel is only required at launch.
"""

from __future__ import annotations

import functools
from typing import Any

from hippihx._lib.fatbin import DEFAULT_ARCH
from hippihx.flydsl.runtime import env_arch, require


@functools.lru_cache(maxsize=1)
def _vector_add() -> Any:
    require()
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @flyc.kernel
    def vector_add_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        tiled_copy: fx.TiledCopy,
    ) -> None:
        tid = fx.thread_idx.x
        bid_x, bid_y = fx.block_idx.x, fx.block_idx.y
        M, N = A.shape.unpack()
        idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
        TileMN = tiled_copy.tile_mn
        gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
        gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
        gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
        cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]
        thr_copy = tiled_copy.get_slice(tid)
        thr_gA = thr_copy.partition_S(gA)
        thr_gB = thr_copy.partition_S(gB)
        thr_gC = thr_copy.partition_D(gC)
        thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]
        thr_rA = fx.make_fragment_like(thr_gA)
        thr_rB = fx.make_fragment_like(thr_gB)
        thr_rC = fx.make_fragment_like(thr_gC)
        thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)
        for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
            thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))
        copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
        fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)
        thr_rC.store(thr_rA.load() + thr_rB.load())
        fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)

    @flyc.jit
    def vector_add(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        tiled_copy = fx.make_tiled_copy_tv(
            copy_atom,
            fx.make_ordered_layout((8, 16), order=(1, 0)),
            fx.make_ordered_layout((1, 4), order=(0, 1)),
        )
        tile_m, tile_n = tiled_copy.tile_mn.unpack()
        M, N = A.shape.unpack()
        grid_m = (M + tile_m - 1) // tile_m
        grid_n = (N + tile_n - 1) // tile_n
        vector_add_kernel(A, B, C, tiled_copy).launch(
            grid=(grid_m, grid_n, 1),
            block=(8 * 16, 1, 1),
            stream=stream,
        )

    return vector_add


def launch(a: Any, b: Any, c: Any, *, arch: str = DEFAULT_ARCH, stream: Any = None) -> None:
    """Compile/run dest vec-add. Caller owns device tensors. Never allocates."""
    require()
    import os

    os.environ.update(env_arch(arch))
    fn = _vector_add()
    if stream is None:
        fn(a, b, c)
    else:
        fn(a, b, c, stream=stream)
