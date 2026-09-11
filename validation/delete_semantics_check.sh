#!/usr/bin/env bash
# Serial-order delete semantics check (validation build). Runs the adversarial
# ycsbx workload with the host oracle: every epoch's resolved records must
# match a serial-order replay, and the GPU index must match the model's live
# set at the end. Then proves the gate is sensitive: with the order-blind
# control (EPIC_TIMELINE_ORDER_BLIND=1, the epoch-boundary semantics), the
# oracle must FAIL. Finally runs the evaluated delete workload (ycsbw) under
# the same oracle.
#
# Usage:  GPU=2 bash validation/delete_semantics_check.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-EGAD/build/epic_driver}
GPU=${GPU:-2}
SEED=${SEED:-42}
DIR=${DIR:-/dev/shm/egad_ycsb_sem}
NUMA=(numactl --physcpubind=12-23,36-47 --membind=1)
BASEENV=(EPIC_WORKLOAD_AWARE_AUTOSIZER=1 OMP_DYNAMIC=false CUDA_VISIBLE_DEVICES=$GPU EPIC_YCSB_SEED=$SEED
         EPIC_YCSB_CACHE_CAP=1500000 EPIC_TIMELINE_ORACLE=1)
ARGSX=(-b ycsbx -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
       -n 8000000 -N 1000000 -x gpu -e 12 -y hybrid_staging -z true)
ARGSW=(-b ycsbw -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
       -n 8000000 -N 1000000 -x gpu -e 10 -y hybrid_staging -z true)
run () {  # $1=log $2...=extra env, then args after --
    local log="$1"; shift
    local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    rm -rf "$DIR"; mkdir -p "$DIR"
    env "${BASEENV[@]}" "${extra[@]}" EPIC_DURABLE_STORE="$DIR" "${NUMA[@]}" "$BIN" "$@" > "$log" 2>&1
    rm -rf "$DIR"
}
fail=0
echo "Delete semantics check: GPU=$GPU seed=$SEED"
echo "----------------------------------------------------------------------"
run /tmp/sem_ycsbx.log -- "${ARGSX[@]}"
final=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' /tmp/sem_ycsbx.log | tail -1)
mapline=$(grep -aoE '\[MAP-CHECK\] (PASS|FAILED).*' /tmp/sem_ycsbx.log | tail -1)
deadline=$(grep -aoE '\[DEAD-CHECK\] (PASS|FAILED).*' /tmp/sem_ycsbx.log | tail -1)
absent=$(echo "$final" | grep -oE 'absent_ops=[0-9]+' | grep -oE '[0-9]+')
reins=$(echo "$final" | grep -oE 'reinserts=[0-9]+' | grep -oE '[0-9]+')
wins=$(echo "$final" | grep -oE 'write_inserts=[0-9]+' | grep -oE '[0-9]+')
echo "[ycsbx] $final"
echo "[ycsbx] $mapline"
echo "[ycsbx] $deadline"
# Both index paths must have run: the timeline fallback (an epoch whose inserts
# were rejected) in some epochs, the fast delete path in the others.
fallback=$(grep -ac 'resolving through the timeline' /tmp/sem_ycsbx.log)
epochs=$(grep -ac 'Running epoch' /tmp/sem_ycsbx.log)
if echo "$final" | grep -q 'PASS' && echo "$mapline" | grep -q 'PASS' && echo "$deadline" | grep -q 'PASS' \
   && [ "${absent:-0}" -gt 0 ] && [ "${reins:-0}" -gt 0 ] && [ "${wins:-0}" -gt 0 ] \
   && [ "$fallback" -gt 0 ] && [ "$fallback" -lt "$epochs" ]; then
    echo "[ycsbx] serial-order semantics: PASS (coverage: absent=$absent reinserts=$reins write_inserts=$wins; timeline fallback in $fallback of $epochs epochs, fast delete path in the others)"
else
    echo "[ycsbx] serial-order semantics: FAIL (log: /tmp/sem_ycsbx.log)"; fail=1
fi
run /tmp/sem_ycsbx_blind.log EPIC_TIMELINE_ORDER_BLIND=1 -- "${ARGSX[@]}" -e 3
blind=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' /tmp/sem_ycsbx_blind.log | tail -1)
if echo "$blind" | grep -q 'FAILED'; then
    echo "[control] order-blind resolution: oracle FAILED as required (gate SENSITIVE)"
else
    echo "[control] order-blind resolution: oracle did NOT fail (gate BLIND): $blind"; fail=1
fi
run /tmp/sem_ycsbw.log -- "${ARGSW[@]}"
finalw=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' /tmp/sem_ycsbw.log | tail -1)
echo "[ycsbw] $finalw"
echo "$finalw" | grep -q 'PASS' || { echo "[ycsbw] FAIL (log: /tmp/sem_ycsbw.log)"; fail=1; }
echo "----------------------------------------------------------------------"
[ "$fail" -eq 0 ] && echo "RESULT: delete semantics PASS" || echo "RESULT: delete semantics FAILED"
exit $fail
