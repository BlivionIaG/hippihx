#pragma once

// Compile-time arch contract for hippihx fatbins.
//
// DOT consumers (same tile source, two builds):
//   gfx1030  V620, ROCm 7.14, wave32, fdot2 / v_dot2c
//   gfx1100  first-class DOT consumer, wave32, same source
//            WMMA is a Later overlay only — never required, never #ifdef'd
//            into the shared DOT tiles.
//
// Vega (not DOT):
//   gfx900   third fatbin: mad_mix / pk_fma — do not load DOT objects
//   gfx906   Later fourth slot / Vega variant — documented, not built yet
//
// Never compile more than one of these into the same object.

#ifndef HIPPIHX_WAVE_SIZE
#if defined(__gfx900__) || defined(__gfx906__)
#define HIPPIHX_WAVE_SIZE 64
#else
#define HIPPIHX_WAVE_SIZE 32
#endif
#endif

#if defined(__gfx1030__) || defined(HIPPIHX_ARCH_gfx1030)
#define HIPPIHX_GFX1030 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1100__) || defined(HIPPIHX_ARCH_gfx1100)
#define HIPPIHX_GFX1100 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx900__) || defined(HIPPIHX_ARCH_gfx900)
#define HIPPIHX_GFX900 1
#endif
#if defined(__gfx906__) || defined(HIPPIHX_ARCH_gfx906)
#define HIPPIHX_GFX906 1
#endif

// Shared DOT paths assume no matrix / FP8 hardware. gfx1100 WMMA must not
// leak into those files (Later overlay only).
#if defined(HIPPIHX_DOT_SLOT)
#define HIPPIHX_HAS_WMMA 0
#define HIPPIHX_HAS_MFMA 0
#define HIPPIHX_HAS_FP8_HW 0
#define HIPPIHX_HAS_FDOT2_BF16 0
#endif
