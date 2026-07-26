#!/usr/bin/env bash
# 09_tpcc_warehouse_sweep: TPCC headline plot. Sweep over W ∈ {4..128}
# (tpccdeck) and W ∈ {4..240} (tpcc/NP) for two mixes side-by-side:
#   - tpccdeck  (10/10/1/1/1, OL backlog bounded by NO=Delivery balance)
#   - tpcc      (50/50/0/0/0 = NewOrder + Payment only, the Epic-paper "NP")
# Three modes per mix: hybrid_staging, cpu_only, gpu_only. The Epic
# Fig 4 + Fig 8 analog: contention varies with W, gpu_only's HBM ceiling
# is the failure boundary, hybrid extends past it.
#
# Per-mode invocation:
#   - hybrid_staging: RUNNING_HYBRID_STAGER.md §4b.1 canonical (full env
#     var set, NUMA pin to node 0, OMP_NESTED 8,3, --hybrid_hbm_reserve_gb
#     left at default 4).
#   - cpu_only:       RUNNING_HYBRID_STAGER.md §4c.1 canonical (no env
#     vars, no NUMA, -c 24).
#   - gpu_only:       EGAD flat-index + realistic-sizing env vars only
#     (no NUMA pin, no staging env vars). Permitted to fail at high W
#     where the OL insert pool exceeds HBM.
#
# Steady-state window: e30-e50. e=50 is sufficient depth for tpccdeck
# (no cliff dynamic) AND for tpcc (no Delivery so OL stays bounded too).
#
# (6 deck W + 11 NP W) × 3 modes × 3 reps = 153 cells. ~60 min total.
#
# W=1, W=2 excluded — pre-existing Epic instability at low warehouse
# counts affects all execution modes; W ≥ 4 is rock-solid.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/tpcc_warehouse_sweep"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
# ON (hybrid record layout) for hybrid_staging; OFF (stock-Epic layout) for the
# cpu_only + gpu_only baselines. cpu_only is value-incorrect on the ON layout and
# gpu_only's ON padding inflates its HBM footprint (stock EPIC-GPU runs well past
# W=128), so both baselines must use the OFF binary. Build both:
#   ./build_binaries.sh --hybrid   (-> build)      and   ./build_binaries.sh   (-> build-off)
BINARY_ON="$EPIC_DIR/build/epic_driver"
BINARY_OFF="$EPIC_DIR/build-off/epic_driver"

MIXES=(tpccdeck tpcc)
# Per-mix warehouse lists. tpccdeck stops at 128: past ~W144 the fully
# resident static tables squeeze the autosized OL/O caches below the mix's
# reach-back working set and hybrid falls under cpu_only (disclosed in the
# paper prose). tpcc (NP) extends past stock EPIC-GPU's ceiling (last alive
# W=208, OOM from W=216 on the 32 GB card) to show hybrid continuing above
# cpu_only; the gpu_only cells at 216/240 fail by design and render as the
# line ending at 208.
WAREHOUSES_DECK=(4 8 16 32 64 128)
WAREHOUSES_NP=(4 8 16 32 64 128 160 192 208 216 240)
MODES=(hybrid_staging cpu_only gpu_only)
REPS=(1 2 3)
EPOCHS=50

run_one() {
    local mix=$1 mode=$2 w=$3 rep=$4
    local log="$OUTDIR/${mix}_w${w}_${mode}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    echo "    [run] $(basename "$log")"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local rc=0
    case "$mode" in
      hybrid_staging)
        EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1 \
        EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1 \
        EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1 \
        OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false \
        CUDA_VISIBLE_DEVICES=0 \
        numactl --physcpubind=0-11,24-35 --membind=0 \
          "$BINARY_ON" -b "$mix" -d epic -w "$w" -a 0.0 -r false -c 24 \
            -s 100000 -f false -m false -n 20000000 \
            -x gpu -e "$EPOCHS" -y hybrid_staging -z true \
            > "$log" 2>&1
        rc=$?
        ;;
      cpu_only)
        CUDA_VISIBLE_DEVICES=0 \
          "$BINARY_OFF" -b "$mix" -d epic -w "$w" -a 0.0 -r false -c 24 \
            -s 100000 -f false -m false -n 20000000 \
            -x cpu -e "$EPOCHS" -y cpu_only \
            > "$log" 2>&1
        rc=$?
        ;;
      gpu_only)
        EPIC_FLAT_AUX_INDEX=1 EPIC_FLAT_INDEX_OL=1 EPIC_MIX_REALISTIC_SIZING=1 \
        CUDA_VISIBLE_DEVICES=0 \
          "$BINARY_OFF" -b "$mix" -d epic -w "$w" -a 0.0 -r false -c 32 \
            -s 100000 -f false -m false -n 20000000 \
            -x gpu -e "$EPOCHS" -y gpu_only \
            > "$log" 2>&1
        rc=$?
        ;;
    esac

    if [[ $rc -ne 0 ]]; then
        if [[ "$mode" == "gpu_only" ]]; then
            echo "    gpu_only failed (rc=$rc) at $mix W=$w — likely HBM overflow"
        else
            echo "    FAIL ($rc): $log"
            return $rc
        fi
    fi
    sleep 1
}

for mix in "${MIXES[@]}"; do
    if [[ "$mix" == "tpcc" ]]; then WLIST=("${WAREHOUSES_NP[@]}"); else WLIST=("${WAREHOUSES_DECK[@]}"); fi
    for mode in "${MODES[@]}"; do
        for w in "${WLIST[@]}"; do
            for rep in "${REPS[@]}"; do
                run_one "$mix" "$mode" "$w" "$rep" || exit $?
            done
        done
    done
done

echo "    tpcc_warehouse_sweep: $(( (${#WAREHOUSES_DECK[@]} + ${#WAREHOUSES_NP[@]}) * ${#MODES[@]} * ${#REPS[@]} )) cells complete (or skipped)."
