#!/usr/bin/env bash
# Top-level orchestrator for the EGAD paper plot pipeline.
#
# Layout:
#   tests/   one *.sh per test. Each script runs an epic_driver experiment
#            and writes its output log under logs/.
#   plots/   one *.py (or *.sh) per figure. Each script reads from logs/
#            and writes one or more figures under figures/.
#   logs/    test outputs (created on first run).
#   figures/ rendered figures (created on first run).
#
# This script (a) runs every test in tests/ in lexicographic order, then
# (b) runs every plot in plots/ in lexicographic order. Tests are skipped
# if their expected log already exists, unless FORCE_RERUN=1 is set.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_DIR="$DIR/tests"
PLOTS_DIR="$DIR/plots"
LOGS_DIR="$DIR/logs"
FIGURES_DIR="$DIR/figures"

mkdir -p "$LOGS_DIR" "$FIGURES_DIR"

export EGAD_LOGS_DIR="$LOGS_DIR"
export EGAD_FIGURES_DIR="$FIGURES_DIR"

# ---- Phase 0: box preflight ----
# Benchmark box convention (reference_baseline.md): hugepage reservation
# deallocated, performance governor. A reboot re-applies /etc/sysctl.conf's
# vm.nr_hugepages=50000 (100 GB locked; node 0 loses 34 GB and membind=0
# hybrid runs decay with epoch depth) and resets the governor to schedutil.
# Both silently corrupt throughput cells, so refuse to run until restored.
hp="$(grep -oP 'HugePages_Total:\s*\K\d+' /proc/meminfo || echo 0)"
gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
if [[ "${hp:-0}" -ne 0 ]]; then
    echo "!!! preflight: HugePages_Total=${hp}, expected 0."
    echo "    Run $DIR/../deallocate_hugepages.sh (safe when HugePages_Free == HugePages_Total), then retry."
    exit 1
fi
if [[ "$gov" != "performance" ]]; then
    echo "!!! preflight: cpu governor is '${gov}', expected performance."
    echo "    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee \"\$c\" >/dev/null; done"
    exit 1
fi
# Some launch environments (containers, sandboxes, tuned latency profiles)
# set prctl(PR_SET_THP_DISABLE) on their process tree. Every epic_driver we
# spawn inherits it, madvise(MADV_HUGEPAGE) silently no-ops, and the TPC-C
# growing-store huge pages never materialize (-6-12% on hybrid W=128 versus
# a normal shell). Refuse rather than record quietly-wrong cells.
thp="$(grep -oP 'THP_enabled:\s*\K\d+' /proc/self/status 2>/dev/null || echo 1)"
if [[ "${thp:-1}" -eq 0 ]]; then
    echo "!!! preflight: THP_enabled: 0 for this process (PR_SET_THP_DISABLE inherited from the launcher)."
    echo "    Re-launch through the clearing wrapper:  $DIR/tools/thp_on $0"
    exit 1
fi
# THP defrag must be 'defer' (a reboot restores the 'madvise' default). Under
# 'madvise'-mode defrag, huge-page faults in the madvised growing stores do
# SYNCHRONOUS direct compaction once node 0's free 2MB-contiguous blocks run
# out (~e285 on the deep cliff cells: compact_stall storms inside the flush
# worker's first-touch scatter writes, deck W=128 steps 8.3 -> 2.9 MTxn/s).
# 'defer' makes those faults fall back to 4KB pages and lets kcompactd catch
# up in the background: the deep runs degrade ~8% instead of collapsing, and
# e<=50 cells are unaffected (they never exhaust the contiguous free pool;
# verified identical e30-50 windows under both settings).
defrag="$(grep -oP '\[\K[^]]+' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo defer)"
if [[ "$defrag" != "defer" ]]; then
    echo "preflight: transparent_hugepage/defrag is '${defrag}'; setting 'defer'"
    echo defer | sudo -n tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
    defrag="$(grep -oP '\[\K[^]]+' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo defer)"
fi
if [[ "$defrag" != "defer" ]]; then
    echo "!!! preflight: could not set transparent_hugepage/defrag=defer (needs passwordless sudo)."
    echo "    echo defer | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
    exit 1
fi
echo "preflight OK (hugepages deallocated, performance governor, THP enabled, defrag=defer)"

echo "=== EGAD plot pipeline ==="
echo "logs    -> $LOGS_DIR"
echo "figures -> $FIGURES_DIR"
echo

# ---- Phase 1: tests ----
echo "--- Phase 1: tests ---"
shopt -s nullglob
for t in "$TESTS_DIR"/*.sh; do
    name="$(basename "$t" .sh)"
    echo ">>> [test] $name"
    bash "$t"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! test $name failed (rc=$rc); aborting"
        exit $rc
    fi
done
echo

# ---- Phase 2: plots ----
echo "--- Phase 2: plots ---"
for p in "$PLOTS_DIR"/*.py "$PLOTS_DIR"/*.sh; do
    name="$(basename "$p")"
    echo ">>> [plot] $name"
    case "$p" in
        *.py) python3 "$p" ;;
        *.sh) bash "$p" ;;
    esac
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! plot $name failed (rc=$rc); aborting"
        exit $rc
    fi
done
echo
echo "=== done ==="
