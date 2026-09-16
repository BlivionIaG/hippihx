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
# Built non-DOT:
#   gfx900               stub; mad_mix / pk_fma — does not load DOT tiles
# Later (configure refused):
#   gfx1013              BC-250 / Cyan Skillfish — not true RDNA2, not dest,
#                        not a portable DOT fatbin with gfx1030. Never
#                        HSA_OVERRIDE a dest object onto it.
#   gfx906               real Vega20/MI50 — not BC-250, not DOT

set(HIPPIHX_KNOWN_ARCHES
  gfx1030
  gfx1100 gfx1101 gfx1102
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
  gfx900
)
set(HIPPIHX_DOT_ARCHES
  gfx1030
  gfx1100 gfx1101 gfx1102
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
)
set(HIPPIHX_DOT_UNOPTIMIZED_ARCHES
  gfx1151
  gfx1031 gfx1032 gfx1033 gfx1035 gfx1036
)
set(HIPPIHX_LATER_NONDOT_ARCHES gfx906)
set(HIPPIHX_LATER_ARCHES gfx906 gfx1013)
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
    "HIPPIHX_ARCH='${HIPPIHX_ARCH}' is Later (not a dest fatbin). "
    "gfx1013 BC-250/Cyan Skillfish is not true RDNA2 — not dest, not portable "
    "DOT with gfx1030. gfx906 is Vega20/MI50, not BC-250. Never HSA_OVERRIDE "
    "a dest object onto Later SKUs. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
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
list(FIND HIPPIHX_DOT_ARCHES "${HIPPIHX_ARCH}" _hippihx_dot_idx)
if(NOT _hippihx_dot_idx EQUAL -1)
  set(HIPPIHX_IS_DOT_SLOT ON)
  list(FIND HIPPIHX_DOT_UNOPTIMIZED_ARCHES "${HIPPIHX_ARCH}" _hippihx_unopt_idx)
  if(NOT _hippihx_unopt_idx EQUAL -1)
    set(HIPPIHX_DOT_UNOPTIMIZED ON)
  endif()
  set(HIPPIHX_WAVE_SIZE 32)
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
    # Never force -mwavefrontsize32 / HSA_OVERRIDE onto a Later SKU.
  endif()
endfunction()
