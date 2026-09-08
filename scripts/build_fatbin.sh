#!/usr/bin/env bash
# Build one hippihx fatbin slot. Never pass multiple arches to one tree.
# Never HSA_OVERRIDE a foreign ISA into another slot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${HIPPIHX_ARCH:-gfx1030}"
BUILD="${BUILD_DIR:-$ROOT/build-$ARCH}"

case "$ARCH" in
  gfx1030|gfx1100|gfx1101|gfx1102|gfx1151|gfx1031|gfx1032|gfx1033|gfx1035|gfx1036|gfx1013|gfx900) ;;
  gfx906)
    echo "gfx906 is Later non-DOT (real Vega20/MI50), not BC-250 — not built yet" >&2
    exit 1
    ;;
  *)
    echo "unknown HIPPIHX_ARCH='$ARCH' (built DOT + gfx900; Later: gfx906)" >&2
    exit 1
    ;;
esac

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
