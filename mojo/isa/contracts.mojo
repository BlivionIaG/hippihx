# Durable ISA contracts for the Mojo/MAX authoring tree.
#
# Emit language does not own these. Match hippihx/_lib/isa.py and
# include/hippihx/isa.hpp.
#
# Mojo objects are not produce for vllm-rdna until re-emitted to HIP
# (hipcc 7.14 → hipModuleLoad / libamdhip64) or an explicit MAX serve path
# is chosen.
#
# gfx1030 packed DOT: fdot2 / v_dot2c and sdot4. No WMMA gate.
# Never fdot2.bf16 (gfx1030 LLVM ISel abort).

alias WAVE32 = 32
alias LDS_BANKS = 32
alias LDS_BANK_BYTES = 4
alias LDS_PAD = 8
alias WMMA_GATE_GFX1030 = False
alias PACKED_DOT_FDOT2 = "fdot2"
alias PACKED_DOT_SDOT4 = "sdot4"
alias LLVM_FDOT2 = "llvm.amdgcn.fdot2"
alias LLVM_SDOT4 = "llvm.amdgcn.sdot4"
alias ISA_FDOT2 = "v_dot2_f32_f16,v_dot2c_f32_f16"
alias ISA_SDOT4 = "v_dot4_i32_i8,v_dot4c_i32_i8"

# Same order as hippihx._lib.fatbin.DOT_ARCHES / isa.MAD_MIX_ARCHES.
alias DOT_ARCHES = "gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1031,gfx1032,gfx1033,gfx1035,gfx1036"
alias MAD_MIX_ARCHES = "gfx900,gfx906,gfx1013"


fn arch_switch(arch: StaticString) -> StaticString:
    """``dot`` or ``mad_mix``. gfx1030 is the primary DOT slot.

    The arch lists are ``DOT_ARCHES`` and ``MAD_MIX_ARCHES`` above.
    """
    if arch == "gfx1030" or arch == "gfx1100" or arch == "gfx1101" or arch == "gfx1102":
        return "dot"
    if arch == "gfx1151" or arch == "gfx1031" or arch == "gfx1032" or arch == "gfx1033":
        return "dot"
    if arch == "gfx1035" or arch == "gfx1036":
        return "dot"
    if arch == "gfx900" or arch == "gfx906" or arch == "gfx1013":
        return "mad_mix"
    return "unknown"


fn refuse_dot_on_mad_mix(arch: StaticString) raises:
    """DOT objects stay off gfx900/gfx906/gfx1013."""
    if arch_switch(arch) == "mad_mix":
        raise Error(
            "do not ship gfx1030 DOT objects onto mad_mix (gfx900/gfx906/gfx1013)"
        )


fn wave_occupancy_ok(dot: Bool, wave: Int) -> Bool:
    """DOT tiles are wave32 only."""
    if dot:
        return wave == WAVE32
    return wave == WAVE32 or wave == 64
