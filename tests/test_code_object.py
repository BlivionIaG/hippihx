"""Consume artifacts: libhippihx_v1.so + one raw hippihx_<arch>.hsaco per slot.

The loader (hippihx_v1_load*) is checked on the host-stub library against
the Python mirror, using synthetic ELF headers and real code objects from
local clang. No device, no HIP runtime: a valid image returns NO_HIP.
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from hippihx._lib import v1_ctypes as cv
from hippihx._lib.catalog import OPS
from hippihx._lib.codeobject import (
    OFFLOAD_BUNDLE_MAGIC,
    check_code_object,
    check_code_object_file,
    code_object_arch,
    code_object_mach,
)
from hippihx._lib.fatbin import DOT_ARCHES, ELF_MACH, KNOWN_ARCHES, LATER_ARCHES
from hippihx._lib.v1 import V1Status

ROOT = Path(__file__).resolve().parents[1]


def header(
    mach: int,
    *,
    flags_extra: int = 0,
    cls: int = 2,
    data: int = 1,
    osabi: int = 64,
    e_type: int = 3,
    machine: int = 224,
    magic: bytes = b"\x7fELF",
) -> bytes:
    ident = magic + bytes([cls, data, 1, osabi, 3]) + bytes(7)
    rest = struct.pack("<HHIQQQIHHHHHH", e_type, machine, 1, 0, 64, 0, mach | flags_extra, 64, 56, 0, 64, 0, 0)
    return ident + rest


def c_load_image(lib: ctypes.CDLL, arch: str, image: bytes) -> int:
    buf = ctypes.create_string_buffer(image, len(image))
    return lib.hippihx_v1_load_image(arch.encode(), buf, len(image))


def expect_loaded(status: V1Status) -> V1Status:
    """A host-stub library validates, then reports that nothing loaded."""

    return V1Status.ERR_NO_HIP if status is V1Status.OK else status


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)


def test_mach_table_covers_every_slot() -> None:
    assert set(ELF_MACH) == set(KNOWN_ARCHES) | set(LATER_ARCHES)
    assert len(set(ELF_MACH.values())) == len(ELF_MACH)


CASES = {
    "gfx1030 object": header(ELF_MACH["gfx1030"]),
    "feature bits (xnack/sramecc) ignored": header(ELF_MACH["gfx1030"], flags_extra=0x300),
    "gfx1031 object": header(ELF_MACH["gfx1031"]),
    "gfx1013 object (BC-250)": header(ELF_MACH["gfx1013"]),
    "truncated": header(ELF_MACH["gfx1030"])[:40],
    "not ELF": b"\x00" * 64,
    "offload bundle": OFFLOAD_BUNDLE_MAGIC + bytes(40),
    "ELF32": header(ELF_MACH["gfx1030"], cls=1),
    "big-endian": header(ELF_MACH["gfx1030"], data=2),
    "OS/ABI not HSA": header(ELF_MACH["gfx1030"], osabi=0),
    "relocatable .o": header(ELF_MACH["gfx1030"], e_type=1),
    "x86-64": header(ELF_MACH["gfx1030"], machine=62),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("arch", ["gfx1030", "gfx1031", "gfx900", "gfx1013", "gfx906", "gfx1200"])
def test_loader_matches_python_on_synthetic_headers(v1_lib, name: str, arch: str) -> None:
    image = CASES[name]
    want = check_code_object(arch, image)
    assert c_load_image(v1_lib, arch, image) == expect_loaded(want)
    if arch in ("gfx1013", "gfx906"):
        assert want is V1Status.ERR_UNSUPPORTED_ARCH  # Later slots never load
    assert v1_lib.hippihx_v1_loaded(arch.encode()) == 0


def test_foreign_mach_is_refused() -> None:
    gfx1030 = header(ELF_MACH["gfx1030"])
    assert check_code_object("gfx1030", gfx1030) is V1Status.OK
    for arch in KNOWN_ARCHES:
        if arch != "gfx1030":
            assert check_code_object(arch, gfx1030) is V1Status.ERR_FOREIGN_ISA, arch
    assert code_object_arch(header(ELF_MACH["gfx1013"])) == "gfx1013"
    assert code_object_mach(CASES["offload bundle"]) is None


def test_hsa_override_refuses_every_load(v1_lib, monkeypatch, tmp_path: Path) -> None:
    image = header(ELF_MACH["gfx1030"])
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    assert check_code_object("gfx1030", image) is V1Status.ERR_FOREIGN_ISA
    assert c_load_image(v1_lib, "gfx1030", image) == V1Status.ERR_FOREIGN_ISA
    path = tmp_path / "hippihx_gfx1030.hsaco"
    path.write_bytes(image)
    assert v1_lib.hippihx_v1_load(b"gfx1030", str(path).encode()) == V1Status.ERR_FOREIGN_ISA
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION")
    assert c_load_image(v1_lib, "gfx1030", image) == V1Status.ERR_NO_HIP


def test_load_from_path(v1_lib, tmp_path: Path) -> None:
    good = tmp_path / "hippihx_gfx1100.hsaco"
    good.write_bytes(header(ELF_MACH["gfx1100"]))
    missing = tmp_path / "missing.hsaco"
    for arch, path in (("gfx1100", good), ("gfx1030", good), ("gfx1100", missing), ("gfx1013", good)):
        want = check_code_object_file(arch, path)
        assert v1_lib.hippihx_v1_load(arch.encode(), str(path).encode()) == expect_loaded(want)
    assert v1_lib.hippihx_v1_load(b"gfx1100", None) == V1Status.ERR_BAD_ARG
    assert v1_lib.hippihx_v1_load_image(b"gfx1100", None, 64) == V1Status.ERR_BAD_ARG


def test_ready_needs_a_loaded_slot(v1_lib) -> None:
    caps = cv.Caps(arch=b"gfx1030")
    plan = cv.new_plan()
    conv = next(spec.v1_id for spec in OPS if spec.qualname == "sequence.causal_conv")
    assert v1_lib.hippihx_v1_plan(conv, ctypes.byref(caps), None, 0, ctypes.byref(plan)) == 0
    assert plan.ready == 0
    assert v1_lib.hippihx_v1_loaded(b"gfx1030") == 0


def test_mach_table_matches_clang(amdgpu_build, tmp_path: Path) -> None:
    source = tmp_path / "k.hip"
    source.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" __global__ void k(int* p) { if (p) *p = 1; }\n'
    )
    for arch, mach in ELF_MACH.items():
        assert code_object_mach(amdgpu_build(arch, source)) == mach, arch


@pytest.mark.parametrize("arch", KNOWN_ARCHES)
def test_hippihx_code_object_per_slot(v1_lib, amdgpu_build, arch: str) -> None:
    image = amdgpu_build(arch, ROOT / "tiles" / "code_object.hip")
    assert image[:4] == b"\x7fELF"  # raw ELF, not an offload bundle
    assert code_object_arch(image) == arch
    assert check_code_object(arch, image) is V1Status.OK
    assert c_load_image(v1_lib, arch, image) == V1Status.ERR_NO_HIP
    for other in KNOWN_ARCHES:
        if other != arch:
            assert c_load_image(v1_lib, other, image) == V1Status.ERR_FOREIGN_ISA
    for spec in OPS:
        # extern "C" entries resolve by plain name; DOT tiles only on DOT slots.
        present = spec.stub_symbol.encode() in image
        assert present is (not spec.dot or arch in DOT_ARCHES), (arch, spec.qualname)


def test_cmake_rule_uses_the_tested_flags(device_flags: tuple[str, ...]) -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "add_library(hippihx_v1 SHARED tiles/v1_abi.cpp)" in cmake
    assert "hippihx_${HIPPIHX_ARCH}.hsaco" in cmake
    assert '"--offload-arch=${HIPPIHX_ARCH}"' in cmake
    assert "HIPPIHX_V1_WITH_HIP=1" in cmake
    for flag in device_flags[2:]:
        assert flag in cmake, flag


def test_hip_branch_compiles_against_the_hip_signature(tmp_path: Path) -> None:
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no host C++ compiler")
    hip = tmp_path / "hip"
    hip.mkdir()
    # Compile-only stand-in with the ROCm declaration hippihx_v1_load uses.
    (hip / "hip_runtime_api.h").write_text(
        "#pragma once\n"
        "typedef enum hipError_t { hipSuccess = 0 } hipError_t;\n"
        "typedef struct ihipModule_t* hipModule_t;\n"
        "hipError_t hipModuleLoadData(hipModule_t* module, const void* image);\n"
    )
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-DHIPPIHX_V1_WITH_HIP=1",
            f"-I{tmp_path}",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tiles" / "v1_abi.cpp"),
        ],
        check=True,
    )
