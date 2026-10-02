"""Shared fixtures: the host-built V1 library and local AMDGPU code objects."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from hippihx._lib import v1_ctypes as cv

ROOT = Path(__file__).resolve().parents[1]

# Same device flags as the CMake hippihx_<arch>.hsaco rule. The local build
# swaps ROCm headers for a two-line stand-in (-nogpuinc -nogpulib).
DEVICE_FLAGS: tuple[str, ...] = (
    "-x",
    "hip",
    "--cuda-device-only",
    "--no-gpu-bundle-output",
    "-O3",
    "-std=c++17",
)


@pytest.fixture(scope="session")
def device_flags() -> tuple[str, ...]:
    return DEVICE_FLAGS


@pytest.fixture(scope="session")
def v1_lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """tiles/v1_abi.cpp built as a host-stub shared object (no HIP)."""

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no host C++ compiler")
    out = tmp_path_factory.mktemp("v1") / "libhippihx_v1.so"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            f"-I{ROOT / 'include'}",
            str(ROOT / "tiles" / "v1_abi.cpp"),
            "-o",
            str(out),
        ],
        check=True,
    )
    return cv.load(out)


@pytest.fixture(scope="session")
def amdgpu_build(tmp_path_factory: pytest.TempPathFactory):
    """``build(arch, source) -> bytes``: a raw AMDGPU code object from local clang."""

    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("no clang")
    targets = subprocess.run(
        [clang, "-print-targets"], capture_output=True, text=True, check=False
    ).stdout
    if "amdgcn" not in targets:
        pytest.skip("clang has no AMDGPU backend")
    work = tmp_path_factory.mktemp("amdgpu")
    fake = work / "include" / "hip"
    fake.mkdir(parents=True)
    (fake / "hip_runtime.h").write_text(
        "#pragma once\n#define __global__ __attribute__((global))\n", encoding="utf-8"
    )

    def build(arch: str, source: Path) -> bytes:
        out = work / f"{source.stem}_{arch}.hsaco"
        subprocess.run(
            [
                clang,
                *DEVICE_FLAGS,
                f"--offload-arch={arch}",
                "-nogpuinc",
                "-nogpulib",
                f"-I{work / 'include'}",
                f"-I{ROOT / 'include'}",
                "-o",
                str(out),
                str(source),
            ],
            check=True,
        )
        return out.read_bytes()

    return build
