#!/usr/bin/env bash
# Set up epic_stock/: the unmodified upstream Epic used for the stock CPU
# baseline cells (see EGAD_plotting/PLOTS.md provenance). Clones upstream at
# the exact commit EGAD forked from and applies the only two changes those
# baselines carry, both build fixes for this hardware generation:
#   patches/epic_stock_build.patch  - CUDA architectures 86 -> 86;89
#                                     (RTX 5000 Ada is compute capability 8.9)
#   patches/cuco_thrust_guard.patch - guard cuco's thrust tuple helpers behind
#                                     THRUST_VERSION < 1.14 (CUDA 12's Thrust
#                                     already provides them)
# The 120 B YCSB record-size variant used by the record-size figure applies
# one more patch on top; upstream's default table is the 1 KB variant.
#   patches/epic_stock_120b.patch   - 100 B fields -> 12 B fields (120 B records)
#
# Usage:
#   ./setup_epic_stock.sh              # 1 KB checkout in ./epic_stock
#   ./setup_epic_stock.sh --small-records [dest]   # 120 B variant
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM=https://github.com/ShujianQian/epic-eval.git
BASE=5a7dc90539ea66306768b513146c5ff1f21121b6

SMALL=0
if [ "${1:-}" = "--small-records" ]; then SMALL=1; shift; fi
DEST="${1:-$ROOT/epic_stock}"

git clone "$UPSTREAM" "$DEST"
git -C "$DEST" checkout "$BASE"
git -C "$DEST" submodule update --init --recursive
git -C "$DEST" apply "$ROOT/patches/epic_stock_build.patch"
git -C "$DEST/third_party/cuco" apply "$ROOT/patches/cuco_thrust_guard.patch"
if [ "$SMALL" = 1 ]; then
    git -C "$DEST" apply "$ROOT/patches/epic_stock_120b.patch"
fi

echo "epic_stock ready at $DEST (upstream @ ${BASE:0:7}, record size $([ "$SMALL" = 1 ] && echo 120B || echo 1KB))."
echo "Build per upstream's README:  cmake -S $DEST -B $DEST/build -DCMAKE_BUILD_TYPE=Release && cmake --build $DEST/build -j"
