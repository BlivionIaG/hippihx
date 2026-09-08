#pragma once

#include "hippihx/tile_stub.hpp"

// Shared DOT source for gfx1030 and gfx1100.
//
// The same .hip (FA / EXL3 / AWQ-W4A16 / moe.shared) is compiled twice:
//   --offload-arch=gfx1030    and    --offload-arch=gfx1100
// as two fatbins. Never one multi-arch object.
//
// Craft locks (room-locked):
//   * wave32 only
//   * fdot2 / v_dot2c — never fdot2.bf16
//   * never #ifdef WMMA (or any WMMA-only gate) on this path
//   * WMMA is a gfx1100-only Later overlay, optional, never required
//   * gfx900 / gfx906 do not load these tiles (mad_mix / pk_fma slot)
//
// Do not import CUDA, CuTe, or b12x objects here.

#if defined(HIPPIHX_GFX900) || defined(HIPPIHX_GFX906)
#error "hippihx DOT tiles are gfx1030+gfx1100 only; Vega fatbins do not load DOT objects"
#endif

#if defined(HIPPIHX_WAVE_SIZE) && (HIPPIHX_WAVE_SIZE != 32)
#error "hippihx DOT tiles are wave32 only"
#endif

#ifndef HIPPIHX_NO_FDOT2_BF16
#define HIPPIHX_NO_FDOT2_BF16 1
#endif

#ifndef HIPPIHX_DOT_SHARED_SOURCE
#define HIPPIHX_DOT_SHARED_SOURCE 1
#endif
