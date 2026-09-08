# Fatbin policy: one architecture per artifact. Never a shared object
# that covers more than one GFX target, and never reuse an object
# compiled for arch A on arch B. Never HSA_OVERRIDE / foreign ISA load.
#
# Built:
#   gfx1030              V620 dest, DOT primary
#   gfx1100/1101/1102    shared DOT source (dot.hpp), no WMMA gate, separate fatbins
#   gfx900               stub; mad_mix / pk_fma — does not load DOT tiles
# Later (configure refused):
#   gfx1151              Strix Halo — Later DOT; VERIFY then likely share if wave32
#   gfx1031..1036        Deck/mobile RDNA2 — Later DOT class; separate objects
#   gfx1013              BC-250 / Cyan Skillfish — Later; VERIFY before sharing
#                        dot.hpp (≠ Navi21). --offload-arch=gfx1013 only.
#   gfx906               real Vega20/MI50 — Later non-DOT. NOT BC-250.

set(HIPPIHX_KNOWN_ARCHES gfx1030 gfx1100 gfx1101 gfx1102 gfx900)
set(HIPPIHX_DOT_ARCHES gfx1030 gfx1100 gfx1101 gfx1102)
set(HIPPIHX_LATER_DOT_ARCHES gfx1151 gfx1031 gfx1032 gfx1033 gfx1035 gfx1036)
set(HIPPIHX_LATER_VERIFY_DOT_ARCHES gfx1013)
set(HIPPIHX_LATER_NONDOT_ARCHES gfx906)
set(HIPPIHX_LATER_ARCHES
  ${HIPPIHX_LATER_DOT_ARCHES}
  ${HIPPIHX_LATER_VERIFY_DOT_ARCHES}
  ${HIPPIHX_LATER_NONDOT_ARCHES}
)
set(HIPPIHX_DEFAULT_ARCH gfx1030)

if(NOT DEFINED HIPPIHX_ARCH)
  set(HIPPIHX_ARCH "${HIPPIHX_DEFAULT_ARCH}" CACHE STRING
      "Single offload arch (built: gfx1030 | gfx1100 | gfx1101 | gfx1102 | gfx900)")
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
  if(HIPPIHX_ARCH STREQUAL "gfx1013")
    message(FATAL_ERROR
      "HIPPIHX_ARCH=gfx1013 is Later (BC-250 / Cyan Skillfish, not Vega20). "
      "VERIFY before sharing dot.hpp — ≠ Navi21/gfx1030; akandr uses RDNA1-macro "
      "paths for 1010/1012/1013. When opened: --offload-arch=gfx1013 only. "
      "Never HSA_OVERRIDE_GFX_VERSION / never load gfx1030 objects. "
      "gfx906 is real Vega20/MI50, not BC-250. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
  elseif(HIPPIHX_ARCH STREQUAL "gfx906")
    message(FATAL_ERROR
      "HIPPIHX_ARCH=gfx906 is Later non-DOT (real Vega20/MI50). "
      "Not BC-250 — BC-250 is gfx1013 / Cyan Skillfish. "
      "Never load FA/EXL3 DOT objects. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
  elseif(HIPPIHX_ARCH STREQUAL "gfx1151")
    message(FATAL_ERROR
      "HIPPIHX_ARCH=gfx1151 is a Later DOT fatbin (Strix Halo). "
      "VERIFY then likely share dot.hpp if wave32; not WMMA-gated. "
      "Separate object; never HSA_OVERRIDE. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
  else()
    message(FATAL_ERROR
      "HIPPIHX_ARCH='${HIPPIHX_ARCH}' is a Later Deck/mobile RDNA2 DOT fatbin. "
      "Same DOT class as gfx1030, separate objects when opened. "
      "Never HSA_OVERRIDE / foreign ISA load. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
  endif()
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
if(HIPPIHX_ARCH STREQUAL "gfx1030")
  set(HIPPIHX_WAVE_SIZE 32)
  set(HIPPIHX_IS_DOT_SLOT ON)
elseif(HIPPIHX_ARCH MATCHES "^gfx110[012]$")
  set(HIPPIHX_WAVE_SIZE 32)
  set(HIPPIHX_IS_DOT_SLOT ON)
elseif(HIPPIHX_ARCH STREQUAL "gfx900")
  set(HIPPIHX_WAVE_SIZE 64)
  set(HIPPIHX_IS_DOT_SLOT OFF)
endif()

# Shared DOT tiles never assume WMMA, including on gfx110x.
set(HIPPIHX_ASSUME_WMMA OFF)
set(HIPPIHX_ASSUME_MFMA OFF)
set(HIPPIHX_ASSUME_FP8_HW OFF)

function(hippihx_apply_arch_flags target)
  target_compile_definitions(${target} PRIVATE
    HIPPIHX_ARCH_${HIPPIHX_ARCH}=1
    HIPPIHX_WAVE_SIZE=${HIPPIHX_WAVE_SIZE}
  )
  if(HIPPIHX_IS_DOT_SLOT)
    target_compile_definitions(${target} PRIVATE HIPPIHX_DOT_SLOT=1)
  endif()
  if(HIPPIHX_HAS_HIP)
    set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES "${HIPPIHX_ARCH}")
    target_compile_options(${target} PRIVATE
      "--offload-arch=${HIPPIHX_ARCH}"
    )
  endif()
endfunction()
