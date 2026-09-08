#pragma once

#include "hippihx/tile_stub.hpp"

// Shared DOT source. Same .hip (FA / EXL3 / AWQ-W4A16 / moe.shared),
// one --offload-arch per fatbin. Never one multi-arch object.
// Never HSA_OVERRIDE / foreign ISA load.
//
// Built slots: gfx1030, gfx1100/1101/1102, gfx1151, gfx1031..1036.
// gfx1151 / Deck gfx103x are portable — can run, not dest-tuned.
// gfx1013 (BC-250 / Cyan Skillfish) is Later VERIFY: RADV reports warp 64.
// Do not share this header until hipDeviceProp.warpSize is measured.
// If HIP is wave64, Skillfish-only — never force -mwavefrontsize32.
//
// Craft locks:
//   * wave32 only
//   * fdot2 / v_dot2c — never fdot2.bf16
//   * never #ifdef WMMA on this path
//
// Vega (gfx900 / gfx906) does not load these tiles.
// Do not import CUDA, CuTe, or b12x objects here.

#if defined(HIPPIHX_GFX900) || defined(HIPPIHX_GFX906)
#error "hippihx DOT tiles are not for Vega (gfx900/gfx906); never load FA/EXL3 DOT"
#endif

#if defined(HIPPIHX_GFX1013)
#error "gfx1013 Later VERIFY: do not share wave32 DOT/FA until warpSize measured; never HSA_OVERRIDE"
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
