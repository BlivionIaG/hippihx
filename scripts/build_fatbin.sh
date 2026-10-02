#!/usr/bin/env bash
# Build one hippihx fatbin slot. Never pass multiple arches to one tree.
# Never HSA_OVERRIDE a foreign ISA into another slot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${HIPPIHX_ARCH:-gfx1030}"
BUILD="${BUILD_DIR:-$ROOT/build-$ARCH}"

# hippihx:gen begin arch_case -- python -m hippihx._lib.codegen; do not edit
case "$ARCH" in
  gfx1030|gfx1100|gfx1101|gfx1102|gfx1200|gfx1151|gfx1031|gfx1032|gfx1033|gfx1035|gfx1036|gfx900) ;;
  gfx906)
    echo "gfx906: Later non-DOT (real Vega20/MI50). Not BC-250 — BC-250 is gfx1013 (Cyan Skillfish, also Later; not true RDNA2). Never load FA/EXL3 DOT objects" >&2
    exit 1
    ;;
  gfx1013)
    echo "gfx1013: Later. BC-250 / Cyan Skillfish is not true RDNA2 — not dest, not a portable DOT fatbin with gfx1030. Never HSA_OVERRIDE a gfx1030/Deck object onto it. Not gfx906/Vega20" >&2
    exit 1
    ;;
  *)
    echo "unknown HIPPIHX_ARCH='$ARCH' (built: gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1151 gfx1031 gfx1032 gfx1033 gfx1035 gfx1036 gfx900; Later: gfx906 gfx1013)" >&2
    exit 1
    ;;
esac
# hippihx:gen end arch_case

if [[ "${ARCH}" == *","* || "${ARCH}" == *" "* ]]; then
  echo "refusing multi-arch HIPPIHX_ARCH='$ARCH'" >&2
  exit 1
fi

EXTRA=()
if [[ "${HIPPIHX_FORCE_HOST_STUB:-}" == "1" ]]; then
  EXTRA+=(-DHIPPIHX_FORCE_HOST_STUB=ON)
fi

cmake -S "$ROOT" -B "$BUILD" -DHIPPIHX_ARCH="$ARCH" "${EXTRA[@]}"
cmake --build "$BUILD"
echo "fatbin: $BUILD/fatbin/$ARCH"
