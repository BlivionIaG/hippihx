#pragma once

// Durable ISA contracts. Independent of emit language.
//
// HIP fatbins, FlyDSL atoms, and Mojo/MAX authoring (mojo/) must match
// hippihx/_lib/isa.py. Mojo objects are not dest-ready. Dest produce stays
// hipcc 7.14 / hipModuleLoad / libamdhip64 until a documented HIP re-emit
// or MAX-serve soak exists. ABI gap: MAX execute is not hippihx_v1_*.
// See docs/ISA.md and docs/MOJO.md.
//
// gfx1030 packed DOT: fdot2 / v_dot2c and sdot4 / v_dot4. No WMMA gate.
// Never emit fdot2.bf16 (gfx1030 LLVM ISel abort).
// LDS: 32 banks × 4 bytes. W4 A-tile pad is 8.
// DOT occupancy gate: wave32 only.
// Do not ship those DOT objects onto gfx900/gfx906/gfx1013 mad_mix paths.

// hippihx:gen begin isa_macros -- python -m hippihx._lib.codegen; do not edit
#ifndef HIPPIHX_ISA_WAVE32
#define HIPPIHX_ISA_WAVE32 32
#endif

#ifndef HIPPIHX_LDS_BANKS
#define HIPPIHX_LDS_BANKS 32
#endif

#ifndef HIPPIHX_LDS_BANK_BYTES
#define HIPPIHX_LDS_BANK_BYTES 4
#endif

#ifndef HIPPIHX_LDS_PAD
#define HIPPIHX_LDS_PAD 8
#endif

#ifndef HIPPIHX_ISA_FDOT2
#define HIPPIHX_ISA_FDOT2 1
#endif

#ifndef HIPPIHX_ISA_SDOT4
#define HIPPIHX_ISA_SDOT4 1
#endif

#ifndef HIPPIHX_ISA_WMMA_GATE
#define HIPPIHX_ISA_WMMA_GATE 0
#endif
// hippihx:gen end isa_macros

static_assert(HIPPIHX_ISA_WAVE32 == 32, "DOT occupancy gate is wave32");
static_assert(HIPPIHX_LDS_BANKS == 32, "RDNA LDS bank count");
static_assert(HIPPIHX_LDS_BANK_BYTES == 4, "RDNA LDS bank width");
static_assert(HIPPIHX_LDS_PAD == 8, "W4 A-tile LDS pad");
static_assert(HIPPIHX_ISA_FDOT2 == 1, "packed DOT includes fdot2");
static_assert(HIPPIHX_ISA_SDOT4 == 1, "packed DOT includes sdot4");
static_assert(HIPPIHX_ISA_WMMA_GATE == 0, "no WMMA gate on gfx1030 packed DOT");
