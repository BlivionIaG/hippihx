"""AMDGPU code object check. Mirror of ``hippihx_v1_load`` in tiles/v1_abi.cpp.

A slot's device code is one raw AMDGPU ELF, ``hippihx_<arch>.hsaco``,
built with one ``--offload-arch`` and ``--no-gpu-bundle-output``. The
loader refuses anything else before ``hipModuleLoadData`` sees it:

- an arch string that is not a built slot (``ERR_UNSUPPORTED_ARCH``);
- ``HSA_OVERRIDE_GFX_VERSION`` in the environment (``ERR_FOREIGN_ISA``);
- not a 64-bit little-endian AMDGPU HSA shared object, including offload
  bundles and relocatable ``.o`` files (``ERR_CODE_OBJECT``);
- an ``EF_AMDGPU_MACH`` that is not the slot's (``ERR_FOREIGN_ISA``).

Host-only, no HIP.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping
from pathlib import Path

from .fatbin import ELF_MACH, KNOWN_ARCHES
from .v1 import V1Status

ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
ELFOSABI_AMDGPU_HSA = 64
ET_DYN = 3
EM_AMDGPU = 224
EF_AMDGPU_MACH = 0xFF
ELF64_EHDR_SIZE = 64
OFFLOAD_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"

_MACH_ARCH: dict[int, str] = {mach: arch for arch, mach in ELF_MACH.items()}


def code_object_mach(image: bytes) -> int | None:
    """``EF_AMDGPU_MACH`` of a loadable code object, or None when it is not one."""

    if len(image) < ELF64_EHDR_SIZE or image[:4] != ELF_MAGIC:
        return None
    if (image[4], image[5], image[7]) != (ELFCLASS64, ELFDATA2LSB, ELFOSABI_AMDGPU_HSA):
        return None
    e_type, e_machine = struct.unpack_from("<HH", image, 16)
    if e_type != ET_DYN or e_machine != EM_AMDGPU:
        return None
    (e_flags,) = struct.unpack_from("<I", image, 48)
    return e_flags & EF_AMDGPU_MACH


def code_object_arch(image: bytes) -> str | None:
    """Slot name for a code object's mach (Later slots included), or None."""

    mach = code_object_mach(image)
    return None if mach is None else _MACH_ARCH.get(mach)


def check_code_object(
    arch: str, image: bytes, env: Mapping[str, str] | None = None
) -> V1Status:
    """What ``hippihx_v1_load_image`` returns before it reaches HIP."""

    env = os.environ if env is None else env
    if arch not in KNOWN_ARCHES:
        return V1Status.ERR_UNSUPPORTED_ARCH
    if env.get("HSA_OVERRIDE_GFX_VERSION"):
        return V1Status.ERR_FOREIGN_ISA
    mach = code_object_mach(image)
    if mach is None:
        return V1Status.ERR_CODE_OBJECT
    if mach != ELF_MACH[arch]:
        return V1Status.ERR_FOREIGN_ISA
    return V1Status.OK


def check_code_object_file(
    arch: str, path: str | Path, env: Mapping[str, str] | None = None
) -> V1Status:
    """What ``hippihx_v1_load`` returns before it reaches HIP."""

    env = os.environ if env is None else env
    if arch not in KNOWN_ARCHES:
        return V1Status.ERR_UNSUPPORTED_ARCH
    if env.get("HSA_OVERRIDE_GFX_VERSION"):
        return V1Status.ERR_FOREIGN_ISA
    try:
        image = Path(path).read_bytes()
    except OSError:
        return V1Status.ERR_CODE_OBJECT
    return check_code_object(arch, image, env)


__all__ = [
    "EM_AMDGPU",
    "OFFLOAD_BUNDLE_MAGIC",
    "check_code_object",
    "check_code_object_file",
    "code_object_arch",
    "code_object_mach",
]
