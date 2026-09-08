# Fatbin policy: one architecture per artifact. Never a shared object
# that covers more than one GFX target, and never reuse an object
# compiled for arch A on arch B. Never HSA_OVERRIDE / foreign ISA load.
#
# Built DOT (same dot.hpp source, separate --offload-arch objects):
#   gfx1030              V620 dest, primary
#   gfx1100/1101/1102    first-class DOT, no WMMA gate
#   gfx1151              Strix Halo — portable/unoptimized, can run
#   gfx1031..1036        Deck/mobile RDNA2 — portable/unoptimized, wave32
#                        (Steam Deck gfx1033 is wave32, same class as gfx1030)
#   gfx1013              BC-250 / Cyan Skillfish — portable/unoptimized,
#                        can run. Wave VERIFY: RADV/llama.cpp report warp
#                        size 64, no matrix cores. Do not force
#                        -mwavefrontsize32 or HSA_OVERRIDE. If HIP is
#                        wave64, Skillfish-only (-mwavefrontsize64 or
#                        leave wave32 off). Serve bind keys on arch + wave.
# Built non-DOT:
#   gfx900               stub; mad_mix / pk_fma — does not load DOT tiles
# Later (configure refused):
#   gfx906               real Vega20/MI50 — not BC-250, not DOT

set(HIPPIHX_KNOWN_ARCHES
  gfx1030
  gfx1100 gfx1101 gfx1102
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
  gfx1013
  gfx900
)
set(HIPPIHX_DOT_ARCHES
  gfx1030
  gfx1100 gfx1101 gfx1102
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
  gfx1013
)
set(HIPPIHX_DOT_UNOPTIMIZED_ARCHES
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
  gfx1013
)
set(HIPPIHX_VERIFY_WAVE_ARCHES gfx1013)
set(HIPPIHX_LATER_NONDOT_ARCHES gfx906)
set(HIPPIHX_LATER_ARCHES ${HIPPIHX_LATER_NONDOT_ARCHES})
set(HIPPIHX_DEFAULT_ARCH gfx1030)

if(NOT DEFINED HIPPIHX_ARCH)
  set(HIPPIHX_ARCH "${HIPPIHX_DEFAULT_ARCH}" CACHE STRING
      "Single offload arch (see HIPPIHX_KNOWN_ARCHES)")
endif()

# Reject multi-arch lists and CMAKE_HIP_ARCHITECTURES bags.
string(REPLACE "," ";" _hippihx_arch_list "${HIPPIHX_ARCH}")
string(REPLACE " " ";" _hippihx_arch_list "${_hippihx_arch_list}")
list(LENGTH _hippihx_arch_list _hippihx_arch_count)
if(_hippihx_arch_count GREATER 1)
  message(FATAL_ERROR
    "HIPPIHX_ARCH must be a single architecture (got '${HIPPIHX_ARCH}'). "
    "DOT source may be shared; fatbins are not — one tree per --offload-arch. "
    "Never HSA_OVERRIDE a foreign ISA into another slot.")
endif()

if(DEFINED CMAKE_HIP_ARCHITECTURES)
  list(LENGTH CMAKE_HIP_ARCHITECTURES _hip_arch_count)
  if(_hip_arch_count GREATER 1)
    message(FATAL_ERROR
      "CMAKE_HIP_ARCHITECTURES has ${_hip_arch_count} entries. "
      "hippihx forbids multi-arch shared objects. Built slots: "
      "${HIPPIHX_KNOWN_ARCHES}")
  endif()
endif()

list(FIND HIPPIHX_LATER_ARCHES "${HIPPIHX_ARCH}" _hippihx_later_idx)
if(NOT _hippihx_later_idx EQUAL -1)
  message(FATAL_ERROR
    "HIPPIHX_ARCH=gfx906 is Later non-DOT (real Vega20/MI50). "
    "Not BC-250 — BC-250 is gfx1013 / Cyan Skillfish (built, portable; "
    "wave VERIFY). Never load FA/EXL3 DOT objects. "
    "Built slots: ${HIPPIHX_KNOWN_ARCHES}")
endif()

list(FIND HIPPIHX_KNOWN_ARCHES "${HIPPIHX_ARCH}" _hippihx_arch_idx)
if(_hippihx_arch_idx EQUAL -1)
  message(FATAL_ERROR
    "Unknown HIPPIHX_ARCH='${HIPPIHX_ARCH}'. Built: ${HIPPIHX_KNOWN_ARCHES}. "
    "Later: ${HIPPIHX_LATER_ARCHES}")
endif()

set(HIPPIHX_ROCM_PIN "7.14" CACHE STRING
    "Documented ROCm pin for V620 / gfx1030. Not auto-enforced when HIP is absent.")

set(HIPPIHX_IS_DOT_SLOT OFF)
set(HIPPIHX_DOT_UNOPTIMIZED OFF)
set(HIPPIHX_WAVE_UNVERIFIED OFF)
list(FIND HIPPIHX_DOT_ARCHES "${HIPPIHX_ARCH}" _hippihx_dot_idx)
if(NOT _hippihx_dot_idx EQUAL -1)
  set(HIPPIHX_IS_DOT_SLOT ON)
  list(FIND HIPPIHX_DOT_UNOPTIMIZED_ARCHES "${HIPPIHX_ARCH}" _hippihx_unopt_idx)
  if(NOT _hippihx_unopt_idx EQUAL -1)
    set(HIPPIHX_DOT_UNOPTIMIZED ON)
  endif()
  list(FIND HIPPIHX_VERIFY_WAVE_ARCHES "${HIPPIHX_ARCH}" _hippihx_wave_idx)
  if(NOT _hippihx_wave_idx EQUAL -1)
    # Wave unverified (RADV reports 64 on Skillfish). Build the slot;
    # do not assume wave32 and never pass -mwavefrontsize32.
    set(HIPPIHX_WAVE_UNVERIFIED ON)
  else()
    set(HIPPIHX_WAVE_SIZE 32)
  endif()
elseif(HIPPIHX_ARCH STREQUAL "gfx900")
  set(HIPPIHX_WAVE_SIZE 64)
  set(HIPPIHX_IS_DOT_SLOT OFF)
endif()

# Shared DOT tiles never assume WMMA, including on gfx110x / gfx1151.
set(HIPPIHX_ASSUME_WMMA OFF)
set(HIPPIHX_ASSUME_MFMA OFF)
set(HIPPIHX_ASSUME_FP8_HW OFF)

function(hippihx_apply_arch_flags target)
  set(_hippihx_defs HIPPIHX_ARCH_${HIPPIHX_ARCH}=1)
  if(DEFINED HIPPIHX_WAVE_SIZE)
    list(APPEND _hippihx_defs HIPPIHX_WAVE_SIZE=${HIPPIHX_WAVE_SIZE})
  endif()
  target_compile_definitions(${target} PRIVATE ${_hippihx_defs})
  if(HIPPIHX_IS_DOT_SLOT)
    target_compile_definitions(${target} PRIVATE HIPPIHX_DOT_SLOT=1)
  endif()
  if(HIPPIHX_HAS_HIP)
    set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES "${HIPPIHX_ARCH}")
    target_compile_options(${target} PRIVATE
      "--offload-arch=${HIPPIHX_ARCH}"
    )
    # Never force -mwavefrontsize32 (especially not on gfx1013 / Skillfish).
  endif()
endfunction()
