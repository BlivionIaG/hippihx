#pragma once

// Compile-time arch contract for hippihx fatbins.
//
// Built DOT (same tile source, separate --offload-arch objects):
//   gfx1030           V620 dest, ROCm 7.14, wave32, fdot2 / v_dot2c
//   gfx1100/1101/1102 first-class DOT, wave32, same dot.hpp
//   gfx1151           Strix Halo — portable/unoptimized, can run
//   gfx1031..1036     Deck/mobile RDNA2 — portable/unoptimized, can run
//
// Built non-DOT:
//   gfx900            mad_mix / pk_fma stub — do not load FA/EXL3 DOT
//
// Later (not built):
//   gfx1013           BC-250 / Cyan Skillfish — Later VERIFY.
//                     RADV/llama.cpp report warp size 64, no matrix cores.
//                     Do not share wave32 dot.hpp / FA until warpSize is
//                     measured. If HIP is wave64: Skillfish-only
//                     (-mwavefrontsize64 or leave wave32 off). Never force
//                     -mwavefrontsize32 / HSA_OVERRIDE. Serve bind keys on
//                     arch + wave size.
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
#if defined(__gfx1151__) || defined(HIPPIHX_ARCH_gfx1151)
#define HIPPIHX_GFX1151 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1031__) || defined(HIPPIHX_ARCH_gfx1031)
#define HIPPIHX_GFX1031 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1032__) || defined(HIPPIHX_ARCH_gfx1032)
#define HIPPIHX_GFX1032 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1033__) || defined(HIPPIHX_ARCH_gfx1033)
#define HIPPIHX_GFX1033 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1035__) || defined(HIPPIHX_ARCH_gfx1035)
#define HIPPIHX_GFX1035 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1036__) || defined(HIPPIHX_ARCH_gfx1036)
#define HIPPIHX_GFX1036 1
#define HIPPIHX_DOT_SLOT 1
#endif
#if defined(__gfx1013__) || defined(HIPPIHX_ARCH_gfx1013)
#define HIPPIHX_GFX1013 1
// Later VERIFY: do not set HIPPIHX_DOT_SLOT. Wave size unconfirmed.
#endif
#if defined(__gfx900__) || defined(HIPPIHX_ARCH_gfx900)
#define HIPPIHX_GFX900 1
#endif
#if defined(__gfx906__) || defined(HIPPIHX_ARCH_gfx906)
#define HIPPIHX_GFX906 1
#endif

// Shared DOT paths assume no matrix / FP8 hardware. gfx110x WMMA must not
// leak into those files (Later overlay only).
#if defined(HIPPIHX_DOT_SLOT)
#define HIPPIHX_HAS_WMMA 0
#define HIPPIHX_HAS_MFMA 0
#define HIPPIHX_HAS_FP8_HW 0
#define HIPPIHX_HAS_FDOT2_BF16 0
#endif
