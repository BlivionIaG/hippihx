#pragma once

// Compile-time arch contract for hippihx fatbins.
//
// Built DOT (same tile source, separate --offload-arch objects):
//   gfx1030           V620 dest, ROCm 7.14, wave32, fdot2 / v_dot2c
//   gfx1100/1101/1102 first-class DOT consumers, wave32, same dot.hpp
//                     WMMA is a Later overlay only — never #ifdef'd in
//
// Built non-DOT:
//   gfx900            mad_mix / pk_fma stub — do not load FA/EXL3 DOT
//
// Later (not built):
//   gfx1151           Strix Halo — Later DOT; VERIFY then likely share if wave32
//   gfx1031..1036     Deck/mobile RDNA2 — Later DOT class; separate objects
//   gfx1013           BC-250 / Cyan Skillfish — VERIFY before sharing dot.hpp
//                     ≠ Navi21. --offload-arch=gfx1013 only.
//   gfx906            real Vega20/MI50 — Later non-DOT. NOT BC-250.
//
// Never compile more than one of these into the same object.
// Never HSA_OVERRIDE_GFX_VERSION / never load a foreign ISA object.

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
#if defined(__gfx1101__) || defined(HIPPIHX_ARCH_gfx1101)
#define HIPPIHX_GFX1101 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1102__) || defined(HIPPIHX_ARCH_gfx1102)
#define HIPPIHX_GFX1102 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx900__) || defined(HIPPIHX_ARCH_gfx900)
#define HIPPIHX_GFX900 1
#endif
#if defined(__gfx906__) || defined(HIPPIHX_ARCH_gfx906)
#define HIPPIHX_GFX906 1
#endif
#if defined(__gfx1013__) || defined(HIPPIHX_ARCH_gfx1013)
#define HIPPIHX_GFX1013 1
#endif
#if defined(__gfx1151__) || defined(HIPPIHX_ARCH_gfx1151)
#define HIPPIHX_GFX1151 1
#endif

// Shared DOT paths assume no matrix / FP8 hardware. gfx110x WMMA must not
// leak into those files (Later overlay only).
#if defined(HIPPIHX_DOT_SLOT)
#define HIPPIHX_HAS_WMMA 0
#define HIPPIHX_HAS_MFMA 0
#define HIPPIHX_HAS_FP8_HW 0
#define HIPPIHX_HAS_FDOT2_BF16 0
#endif
