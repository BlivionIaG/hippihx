"""Public ctypes view of the V1 C ABI (``include/hippihx/v1.h``).

Serve (``rdna_extras``) loads ``libhippihx_v1.so`` through this module.
Importing it loads nothing; ``load(path)`` opens the library.
"""

from hippihx._lib.v1_ctypes import (
    Caps,
    Fabric,
    Param,
    Plan,
    ScratchSpec,
    Tensor,
    fabric_struct,
    load,
    new_plan,
    params_array,
    tensor,
)

__all__ = [
    "Caps",
    "Fabric",
    "Param",
    "Plan",
    "ScratchSpec",
    "Tensor",
    "fabric_struct",
    "load",
    "new_plan",
    "params_array",
    "tensor",
]
