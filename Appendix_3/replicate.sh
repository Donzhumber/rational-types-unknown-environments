#!/usr/bin/env bash
# ==============================================================================
# APPENDIX 3 -- ONE-CLICK LEAN VERIFICATION ENTRY POINT
#
# Run from the package root:
#   ./replicate.sh
# or from anywhere:
#   bash "/path/to/Appendix_3/replicate.sh"
# ==============================================================================
set -euo pipefail

find_pkgroot() {
  local dir="$1"
  while [[ -n "$dir" && "$dir" != "/" ]]; do
    if [[ -f "$dir/lean/Appendix3Proofs.lean" && -f "$dir/lean/lakefile.lean" ]]; then
      printf '%s' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKGROOT="$(find_pkgroot "$SCRIPT_DIR" || true)"
if [[ -z "$PKGROOT" ]]; then
  PKGROOT="$(find_pkgroot "$(pwd)" || true)"
fi
if [[ -z "$PKGROOT" ]]; then
  echo "ERROR: Could not locate Appendix_3/ (lean/Appendix3Proofs.lean missing)." >&2
  exit 1
fi

LEAN_DIR="$PKGROOT/lean"
LOG_FILE="$LEAN_DIR/build.log"

if ! command -v lake >/dev/null 2>&1; then
  echo "ERROR: lake not found. Install Lean 4 and ensure elan/lake is on PATH." >&2
  echo "Toolchain file: $LEAN_DIR/lean-toolchain" >&2
  exit 2
fi

echo "=== APPENDIX 3: LEAN FORMAL VERIFICATION ==="
echo "Package root: $PKGROOT"
echo "Lean project: $LEAN_DIR"
echo "Logging to:   $LOG_FILE"
echo

cd "$LEAN_DIR"
{
  echo "Build started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "Working dir: $(pwd)"
  echo "Lean toolchain: $(cat lean-toolchain)"
  echo
  lake build
  echo
  echo "Build finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
} 2>&1 | tee "$LOG_FILE"

echo
echo "=== VERIFICATION COMPLETE ==="
echo "Full log: $LOG_FILE"
