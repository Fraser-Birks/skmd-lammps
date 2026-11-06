#!/usr/bin/env bash
# CI smoke test for fix skmd (runs a 32-atom LJ system for a few steps).
# Usage: tests/smoke/run_smoke.sh <lmp-binary> <plugin.so|none> [extra lmp args...]
#   e.g. run_smoke.sh build/lmp none -k on -sf kk     (in-tree Kokkos build)
set -euo pipefail
LMP=$(realpath "$1"); PLUGIN=$2; shift 2
[ "$PLUGIN" = "none" ] || PLUGIN=$(realpath "$PLUGIN")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
cd "$WORK"
python3 "$HERE/make_dataset.py" smoke
if [ "$PLUGIN" = "none" ]; then : > pre.in; else echo "plugin load $PLUGIN" > pre.in; fi
"$LMP" -in "$HERE/in.skmd_smoke" -log log.smoke "$@"

fail() { echo "SMOKE TEST FAILED: $*"; exit 1; }
grep -q "threshold met on step 5" log.smoke   || fail "threshold did not trigger at step 5"
grep -q "SMOKE fnorm=.* b=100 triggered=1" log.smoke || fail "unexpected fix outputs"
grep -q "setting b adaptively to 5$" log.smoke || fail "adaptive b != 5"
grep -q "SMOKE adaptive b=5$" log.smoke        || fail "adaptive b not reported as 5"
test -s smoke_stats.txt                        || fail "no _stats.txt written"
grep -i "nan" smoke_stats.txt && fail "NaN in fnorm"
echo "SMOKE TEST PASSED"
