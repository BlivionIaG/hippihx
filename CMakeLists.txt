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

if(DEFINED HIPPIHX_WAVE_SIZE)
  message(STATUS "hippihx: fatbin slot ${HIPPIHX_ARCH} (wave${HIPPIHX_WAVE_SIZE})")
else()
  message(STATUS "hippihx: fatbin slot ${HIPPIHX_ARCH}")
endif()
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

# mojo/ is the MAX authoring tree. It is not a fatbin input.
# Mojo objects are not produce until re-emitted with hipcc (see docs/MOJO.md).

# Shared DOT source: FA / EXL3 / AWQ-W4A16 / moe.shared (catalog dot=True).
# Same .hip for every DOT slot (gfx1030, gfx110x, gfx1151, gfx103x).
# gfx1013 is Later (not true RDNA2). gfx900 does not load these.
# Never #ifdef WMMA. Never HSA_OVERRIDE.
# Non-DOT contracts compile for every built fatbin (stubs).
# tiles/v1_abi.cpp is the torch-free C consume surface extras will bind.
# hippihx:gen begin tile_sources -- python -m hippihx._lib.codegen; do not edit
set(HIPPIHX_DOT_SOURCES
  tiles/attention/fa_fdot2/kernel.hip
  tiles/gemm/w4a16_fdot2/kernel.hip
  tiles/gemm/exl3_3inst/kernel.hip
  tiles/moe/shared/kernel.hip
)
set(HIPPIHX_COMMON_SOURCES
  tiles/smoke.hip
  tiles/v1_abi.cpp
  tiles/attention/gdn_scan/kernel.hip
  tiles/attention/kda_scan/kernel.hip
  tiles/attention/qsa_indexer/kernel.hip
  tiles/attention/dsa_nope/kernel.hip
  tiles/moe/routed/kernel.hip
  tiles/moe/leftover_bf16/kernel.hip
  tiles/sequence/causal_conv/kernel.hip
  tiles/comm/pcie/kernel.hip
)
# hippihx:gen end tile_sources

set(HIPPIHX_TILE_SOURCES ${HIPPIHX_COMMON_SOURCES})
if(HIPPIHX_IS_DOT_SLOT)
  list(APPEND HIPPIHX_TILE_SOURCES ${HIPPIHX_DOT_SOURCES})
  if(HIPPIHX_DOT_UNOPTIMIZED)
    message(STATUS
      "hippihx: DOT shared source ON for ${HIPPIHX_ARCH} "
      "(portable/unoptimized; own --offload-arch; never HSA_OVERRIDE)")
  else()
    message(STATUS "hippihx: DOT shared source ON for ${HIPPIHX_ARCH} (wave32, no WMMA gate)")
  endif()
else()
  message(STATUS "hippihx: non-DOT slot ${HIPPIHX_ARCH} — no FA/EXL3 DOT (mad_mix / pk_fma)")
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

# Consume artifacts (docs/CONSUME.md). The static .a above stays the link
# smoke.
#   libhippihx_v1.so      V1 symbols + generated tables, no device code.
#                         With HIP it links libamdhip64 and hippihx_v1_load
#                         loads code objects; the host stub only validates.
#   hippihx_<arch>.hsaco  this slot's device code: one raw AMDGPU ELF (one
#                         --offload-arch, no offload bundle).
add_library(hippihx_v1 SHARED tiles/v1_abi.cpp)
target_include_directories(hippihx_v1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(hippihx_v1 PROPERTIES
  OUTPUT_NAME "hippihx_v1"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fatbin/${HIPPIHX_ARCH}"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

if(HIPPIHX_HAS_HIP)
  find_package(hip REQUIRED CONFIG)
  target_link_libraries(hippihx_v1 PRIVATE hip::host)
  target_compile_definitions(hippihx_v1 PRIVATE HIPPIHX_V1_WITH_HIP=1)

  set(HIPPIHX_CODE_OBJECT
      "${CMAKE_BINARY_DIR}/fatbin/${HIPPIHX_ARCH}/hippihx_${HIPPIHX_ARCH}.hsaco")
  file(GLOB HIPPIHX_TILE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/hippihx/*.hpp")
  add_custom_command(
    OUTPUT "${HIPPIHX_CODE_OBJECT}"
    COMMAND "${CMAKE_HIP_COMPILER}" -x hip "--offload-arch=${HIPPIHX_ARCH}"
            --cuda-device-only --no-gpu-bundle-output -O3 -std=c++17
            "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
            -o "${HIPPIHX_CODE_OBJECT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tiles/code_object.hip"
    DEPENDS tiles/code_object.hip ${HIPPIHX_TILE_SOURCES} ${HIPPIHX_TILE_HEADERS}
    COMMENT "hippihx: ${HIPPIHX_ARCH} code object (one --offload-arch, raw ELF)"
    VERBATIM
  )
  add_custom_target(hippihx_code_object ALL DEPENDS "${HIPPIHX_CODE_OBJECT}")
endif()

# Convenience: document slots without mixing them into this .a.
add_custom_target(hippihx_list_fatbin_slots
  COMMAND ${CMAKE_COMMAND} -E echo
          "DOT: gfx1030 gfx110x gfx1151 gfx103x; stub gfx900; Later gfx1013(Skillfish) gfx906(Vega20)"
)
