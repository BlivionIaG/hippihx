#!/usr/bin/env bash
# Build one hippihx fatbin slot. Never pass multiple arches to one tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${HIPPIHX_ARCH:-gfx1030}"
BUILD="${BUILD_DIR:-$ROOT/build-$ARCH}"

case "$ARCH" in
  gfx1030|gfx1100|gfx900) ;;
  *)
    echo "HIPPIHX_ARCH must be gfx1030, gfx1100, or gfx900 (got '$ARCH')" >&2
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
