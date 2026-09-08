# Fatbin policy: one architecture per artifact. Never a shared object
# that covers more than one GFX target, and never reuse an object
# compiled for arch A on arch B.
#
# DOT slots (same tile source, two builds):
#   gfx1030  V620, ROCm 7.14, wave32
#   gfx1100  first-class DOT consumer, wave32 — not an afterthought
# Vega:
#   gfx900   built slot: mad_mix / pk_fma — does not load DOT tiles
#   gfx906   Later fourth slot — documented, not a CMake target yet

set(HIPPIHX_KNOWN_ARCHES gfx1030 gfx1100 gfx900)
set(HIPPIHX_DOT_ARCHES gfx1030 gfx1100)
set(HIPPIHX_LATER_ARCHES gfx906)
set(HIPPIHX_DEFAULT_ARCH gfx1030)

if(NOT DEFINED HIPPIHX_ARCH)
  set(HIPPIHX_ARCH "${HIPPIHX_DEFAULT_ARCH}" CACHE STRING
      "Single offload arch for this fatbin (gfx1030 | gfx1100 | gfx900)")
endif()

# Reject multi-arch lists and CMAKE_HIP_ARCHITECTURES bags.
string(REPLACE "," ";" _hippihx_arch_list "${HIPPIHX_ARCH}")
string(REPLACE " " ";" _hippihx_arch_list "${_hippihx_arch_list}")
list(LENGTH _hippihx_arch_list _hippihx_arch_count)
if(_hippihx_arch_count GREATER 1)
  message(FATAL_ERROR
    "HIPPIHX_ARCH must be a single architecture (got '${HIPPIHX_ARCH}'). "
    "DOT source is shared, fatbins are not — configure one tree per arch "
    "(gfx1030 and gfx1100 are two builds of the same tiles).")
endif()

if(DEFINED CMAKE_HIP_ARCHITECTURES)
  list(LENGTH CMAKE_HIP_ARCHITECTURES _hip_arch_count)
  if(_hip_arch_count GREATER 1)
    message(FATAL_ERROR
      "CMAKE_HIP_ARCHITECTURES has ${_hip_arch_count} entries. "
      "hippihx forbids multi-arch shared objects. Set HIPPIHX_ARCH to one of: "
      "${HIPPIHX_KNOWN_ARCHES}")
  endif()
endif()

list(FIND HIPPIHX_LATER_ARCHES "${HIPPIHX_ARCH}" _hippihx_later_idx)
if(NOT _hippihx_later_idx EQUAL -1)
  message(FATAL_ERROR
    "HIPPIHX_ARCH='${HIPPIHX_ARCH}' is a Later fatbin slot "
    "(Vega variant; mad_mix / pk_fma — not DOT). Documented only; "
    "do not build it yet. Built slots: ${HIPPIHX_KNOWN_ARCHES}")
endif()

list(FIND HIPPIHX_KNOWN_ARCHES "${HIPPIHX_ARCH}" _hippihx_arch_idx)
if(_hippihx_arch_idx EQUAL -1)
  message(FATAL_ERROR
    "Unknown HIPPIHX_ARCH='${HIPPIHX_ARCH}'. Built slots: ${HIPPIHX_KNOWN_ARCHES}. "
    "Later slots (not built): ${HIPPIHX_LATER_ARCHES}")
endif()

set(HIPPIHX_ROCM_PIN "7.14" CACHE STRING
    "Documented ROCm pin for V620 / gfx1030. Not auto-enforced when HIP is absent.")

set(HIPPIHX_IS_DOT_SLOT OFF)
if(HIPPIHX_ARCH STREQUAL "gfx1030")
  set(HIPPIHX_WAVE_SIZE 32)
  set(HIPPIHX_IS_DOT_SLOT ON)
elseif(HIPPIHX_ARCH STREQUAL "gfx1100")
  set(HIPPIHX_WAVE_SIZE 32)
  set(HIPPIHX_IS_DOT_SLOT ON)
elseif(HIPPIHX_ARCH STREQUAL "gfx900")
  set(HIPPIHX_WAVE_SIZE 64)
  set(HIPPIHX_IS_DOT_SLOT OFF)
endif()

# Shared DOT tiles never assume WMMA, including on gfx1100.
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
