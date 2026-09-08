#pragma once

#include "hippihx/tile_stub.hpp"

// Shared DOT source for built RDNA DOT slots:
//   gfx1030 + gfx1100/1101/1102
// Same .hip (FA / EXL3 / AWQ-W4A16 / moe.shared), one --offload-arch each.
// Never one multi-arch object. Never HSA_OVERRIDE / foreign ISA load.
//
// Craft locks (room-locked):
//   * wave32 only
//   * fdot2 / v_dot2c — never fdot2.bf16
//   * never #ifdef WMMA (or any WMMA-only gate) on this path
//   * WMMA is a gfx110x-only Later overlay, optional, never required
//
// Do not load these tiles on:
//   gfx900 / gfx906 (Vega mad_mix / pk_fma; gfx906 = Vega20/MI50, not BC-250)
//   gfx1013 (BC-250 / Cyan Skillfish) until VERIFY — ≠ Navi21
//
// Later Deck gfx103x / gfx1151 may share this header after VERIFY.
// Do not import CUDA, CuTe, or b12x objects here.

#if defined(HIPPIHX_GFX900) || defined(HIPPIHX_GFX906)
#error "hippihx DOT tiles are not for Vega (gfx900/gfx906); never load FA/EXL3 DOT"
#endif
#if defined(HIPPIHX_GFX1013)
#error "gfx1013 (BC-250 / Cyan Skillfish) must VERIFY before sharing dot.hpp"
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
