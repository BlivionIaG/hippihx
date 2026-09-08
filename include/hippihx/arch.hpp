#pragma once

// Compile-time arch contract for hippihx fatbins.
//
// gfx1030 (V620, ROCm 7.14, wave32) is first. gfx1100 and gfx900 are
// separate fatbin slots — never compile more than one of these into the
// same object. Do not assume WMMA, MFMA, or FP8 hardware on gfx1030.

#ifndef HIPPIHX_WAVE_SIZE
#if defined(__gfx900__)
#define HIPPIHX_WAVE_SIZE 64
#else
#define HIPPIHX_WAVE_SIZE 32
#endif
#endif

#if defined(__gfx1030__) || defined(HIPPIHX_ARCH_gfx1030)
#define HIPPIHX_GFX1030 1
#endif
#if defined(__gfx1100__) || defined(HIPPIHX_ARCH_gfx1100)
#define HIPPIHX_GFX1100 1
#endif
#if defined(__gfx900__) || defined(HIPPIHX_ARCH_gfx900)
#define HIPPIHX_GFX900 1
#endif

// Hardware matrix / FP8 units are out of scope for the gfx1030 contract.
#if defined(HIPPIHX_GFX1030)
#define HIPPIHX_HAS_WMMA 0
#define HIPPIHX_HAS_MFMA 0
#define HIPPIHX_HAS_FP8_HW 0
#endif
