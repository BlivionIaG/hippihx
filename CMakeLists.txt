cmake_minimum_required(VERSION 3.21)

# Validate the fatbin slot before enabling languages so a broken host
# compiler cannot mask a multi-arch misconfigure.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HippihxArch.cmake")

# Languages are decided after we know whether hipcc is present. CXX is
# always enabled so this tree stays configure-able in CI without ROCm.
project(hippihx VERSION 0.0.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(HIP)

option(HIPPIHX_FORCE_HOST_STUB
       "Compile .hip sources as host C++ (no hipcc). For layout CI only." OFF)

set(HIPPIHX_HAS_HIP OFF)
if(NOT HIPPIHX_FORCE_HOST_STUB AND CMAKE_HIP_COMPILER)
  enable_language(HIP)
  set(HIPPIHX_HAS_HIP ON)
  message(STATUS "hippihx: HIP compiler ${CMAKE_HIP_COMPILER}")
else()
  if(HIPPIHX_FORCE_HOST_STUB)
    message(STATUS "hippihx: host stub forced (HIPPIHX_FORCE_HOST_STUB=ON)")
  else()
    message(WARNING
      "hippihx: no HIP compiler — building the host compile stub. "
      "On V620 use ROCm ${HIPPIHX_ROCM_PIN} hipcc with -DHIPPIHX_ARCH=gfx1030.")
  endif()
endif()

message(STATUS "hippihx: fatbin slot ${HIPPIHX_ARCH} (wave${HIPPIHX_WAVE_SIZE})")
message(STATUS "hippihx: ROCm pin for V620 is ${HIPPIHX_ROCM_PIN}")

add_library(hippihx STATIC)
add_library(hippihx::hippihx ALIAS hippihx)

set_target_properties(hippihx PROPERTIES
  OUTPUT_NAME "hippihx_${HIPPIHX_ARCH}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fatbin/${HIPPIHX_ARCH}"
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

if(HIPPIHX_HAS_HIP)
  set_target_properties(hippihx PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
endif()

target_include_directories(hippihx PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Shared DOT source: FA / EXL3 / AWQ-W4A16 / moe.shared.
# Same .hip for gfx1030 and gfx1100; two --offload-arch builds, never one
# multi-arch object. gfx900 does not load these (mad_mix / pk_fma slot).
# Never #ifdef WMMA in these files.
set(HIPPIHX_DOT_SOURCES
  tiles/attn/fa_fdot2/kernel.hip
  tiles/gemm/w4a16_fdot2/kernel.hip
  tiles/gemm/exl3_3inst/kernel.hip
  tiles/moe/shared/kernel.hip
)

# Non-DOT contracts compile for every built fatbin (stubs).
set(HIPPIHX_COMMON_SOURCES
  tiles/smoke.hip
  tiles/attn/gdn_scan/kernel.hip
  tiles/attn/kda_scan/kernel.hip
  tiles/attn/qsa_indexer/kernel.hip
  tiles/attn/dsa_nope/kernel.hip
  tiles/moe/routed/kernel.hip
  tiles/moe/leftover_bf16/kernel.hip
  tiles/sequence/causal_conv/kernel.hip
  tiles/comm/pcie/kernel.hip
)

set(HIPPIHX_TILE_SOURCES ${HIPPIHX_COMMON_SOURCES})
if(HIPPIHX_IS_DOT_SLOT)
  list(APPEND HIPPIHX_TILE_SOURCES ${HIPPIHX_DOT_SOURCES})
  message(STATUS "hippihx: DOT shared source ON for ${HIPPIHX_ARCH} (wave32, no WMMA gate)")
else()
  message(STATUS "hippihx: Vega slot ${HIPPIHX_ARCH} — no DOT tiles (mad_mix / pk_fma)")
endif()

if(NOT HIPPIHX_HAS_HIP)
  foreach(_src IN LISTS HIPPIHX_TILE_SOURCES)
    set_source_files_properties("${_src}" PROPERTIES LANGUAGE CXX)
  endforeach()
  target_compile_definitions(hippihx PRIVATE HIPPIHX_HOST_STUB=1)
endif()

target_sources(hippihx PRIVATE ${HIPPIHX_TILE_SOURCES})
hippihx_apply_arch_flags(hippihx)

# Host-callable smoke symbol so a later Python extension can link a real .a.
add_executable(hippihx_smoke_host tiles/smoke_host.cpp)
target_link_libraries(hippihx_smoke_host PRIVATE hippihx::hippihx)
target_include_directories(hippihx_smoke_host PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
set_target_properties(hippihx_smoke_host PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fatbin/${HIPPIHX_ARCH}"
)

# Convenience: document slots without mixing them into this .a.
add_custom_target(hippihx_list_fatbin_slots
  COMMAND ${CMAKE_COMMAND} -E echo
          "DOT fatbins: gfx1030 | gfx1100 (same source, two builds) ; Vega: gfx900 ; Later: gfx906"
)
