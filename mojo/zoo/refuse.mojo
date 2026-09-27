# Shared fail-closed path for every MAX registration.
#
# Mojo objects are not dest-ready. Dest produce stays hipcc 7.14 /
# hipModuleLoad / libamdhip64 until a documented HIP re-emit or MAX-serve
# soak exists. ABI gap: MAX execute is not hippihx_v1_plan / hippihx_v1_run.
# These functions do not enqueue and do not allocate.

fn not_produce(qualname: StaticString, arch: StaticString) raises:
    """Authoring slot. No device enqueue."""
    raise Error(
        qualname,
        " Mojo object is not dest-ready (arch=",
        arch,
        "); ABI gap versus hippihx_v1_*; dest produce stays hipcc 7.14 / "
        "hipModuleLoad / libamdhip64 until a documented HIP re-emit or "
        "MAX-serve soak exists",
    )


fn refuse_dot_on_gfx900(qualname: StaticString) raises:
    """gfx1030 packed-DOT objects do not ship on the gfx900 mad_mix slot.

    gfx906 and gfx1013 are Later and are not registered. ``refuse_dot_on_mad_mix``
    in ``isa/contracts.mojo`` still rejects them.
    """
    raise Error(
        "do not ship gfx1030 DOT objects onto gfx900 mad_mix (",
        qualname,
        "); re-emit is still not this path",
    )
