#!/usr/bin/env bash
# Serial-order delete semantics check (validation build). Runs the adversarial
# ycsbx workload with the host oracle: every epoch's resolved records must
# match a serial-order replay, the GPU index must match the model's live set
# at the end, and no deleted record may be written after its delete. The
# same workload runs again on a map filled to a high occupancy, where a
# key's probe chain crosses erased slots (the fresh-insert check must hold
# there). Then proves the gate is sensitive: with the order-blind control
# (EPIC_TIMELINE_ORDER_BLIND=1, both index paths resolve as the
# epoch-boundary semantics did), the oracle must FAIL. Then runs the
# evaluated delete workload (ycsbw) under the oracle, and finally checks
# that forcing every epoch through the timeline (EPIC_TIMELINE_FORCE_GENERAL=1)
# reproduces the fast path's hashes exactly on ycsbf, ycsbi and ycsbw.
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
# High occupancy: 1.8 M of 3 M records live, the map at 48 % of its slots, so
# most probe chains cross slots that the churn's deletes erased. The cache
# holds the whole live set; the eviction path is not what this run checks.
ARGSXD=(-b ycsbx -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
        -n 3000000 -N 1800000 -x gpu -e 12 -y hybrid_staging -z true)
ARGSF=(-b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
       -n 20000000 -x gpu -e 20 -y hybrid_staging -z true)
ARGSI=(-b ycsbi -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
       -n 8000000 -N 1000000 -x gpu -e 10 -y hybrid_staging -z true)
fail=0
run () {  # $1=log $2...=extra env, then args after --; a non-zero exit fails the check
    local log="$1"; shift
    local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    rm -rf "$DIR"; mkdir -p "$DIR"
    env "${BASEENV[@]}" "${extra[@]}" EPIC_DURABLE_STORE="$DIR" "${NUMA[@]}" "$BIN" "$@" > "$log" 2>&1
    local rc=$?
    rm -rf "$DIR"
    [ "$rc" -ne 0 ] && { echo "[run] $log: exit $rc"; fail=1; }
    return 0
}
hashes () { grep -aoE 'STATE-HASH(-VALONLY|-LIVE)?\] [^=]*= 0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tr '\n' ' '; }
echo "Delete semantics check: GPU=$GPU seed=$SEED"
echo "----------------------------------------------------------------------"
check_x () {  # $1=tag $2=log: oracle, map and dead checks, coverage, both index paths
    local tag="$1" log="$2"
    local final mapline deadline absent reins wins fallback epochs
    final=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' "$log" | tail -1)
    mapline=$(grep -aoE '\[MAP-CHECK\] (PASS|FAILED).*' "$log" | tail -1)
    deadline=$(grep -aoE '\[DEAD-CHECK\] (PASS|FAILED).*' "$log" | tail -1)
    absent=$(echo "$final" | grep -oE 'absent_ops=[0-9]+' | grep -oE '[0-9]+')
    reins=$(echo "$final" | grep -oE 'reinserts=[0-9]+' | grep -oE '[0-9]+')
    wins=$(echo "$final" | grep -oE 'write_inserts=[0-9]+' | grep -oE '[0-9]+')
    echo "[$tag] $final"
    echo "[$tag] $mapline"
    echo "[$tag] $deadline"
    # Both index paths must have run: the timeline fallback (an epoch with an
    # insert of a live key or a key inserted twice) in some epochs, the fast
    # delete path in the others.
    fallback=$(grep -ac 'resolving through the timeline' "$log")
    epochs=$(grep -ac 'Running epoch' "$log")
    if echo "$final" | grep -q '\] PASS' && echo "$mapline" | grep -q '\] PASS' && echo "$deadline" | grep -q '\] PASS' \
       && [ "${absent:-0}" -gt 0 ] && [ "${reins:-0}" -gt 0 ] && [ "${wins:-0}" -gt 0 ] \
       && [ "$fallback" -gt 0 ] && [ "$fallback" -lt "$epochs" ]; then
        echo "[$tag] serial-order semantics: PASS (coverage: absent=$absent reinserts=$reins write_inserts=$wins; timeline fallback in $fallback of $epochs epochs, fast delete path in the others)"
    else
        echo "[$tag] serial-order semantics: FAIL (log: $log)"; fail=1
    fi
}
run /tmp/sem_ycsbx.log -- "${ARGSX[@]}"
check_x ycsbx /tmp/sem_ycsbx.log
run /tmp/sem_ycsbx_dense.log -- "${ARGSXD[@]}" EPIC_YCSB_CACHE_CAP=4000000
check_x ycsbx-dense /tmp/sem_ycsbx_dense.log
# Sensitivity control, three epochs (the last -e wins): both index paths
# resolve order-blind, so the oracle must reject the run.
run /tmp/sem_ycsbx_blind.log EPIC_TIMELINE_ORDER_BLIND=1 -- "${ARGSX[@]}" -e 3
blind=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' /tmp/sem_ycsbx_blind.log | tail -1)
if echo "$blind" | grep -q '\] FAILED'; then
    echo "[control] order-blind resolution: oracle FAILED as required (gate SENSITIVE)"
else
    echo "[control] order-blind resolution: oracle did NOT fail (gate BLIND): $blind"; fail=1
fi
run /tmp/sem_ycsbw.log -- "${ARGSW[@]}"
finalw=$(grep -aoE '\[TIMELINE-ORACLE\] (PASS|FAILED).*' /tmp/sem_ycsbw.log | tail -1)
echo "[ycsbw] $finalw"
echo "$finalw" | grep -q '\] PASS' || { echo "[ycsbw] FAIL (log: /tmp/sem_ycsbw.log)"; fail=1; }
# Forcing every epoch through the timeline must reproduce the fast path's
# hashes (value-only, positional, live digest) exactly.
for wl in ycsbf ycsbi ycsbw; do
    case $wl in ycsbf) args=("${ARGSF[@]}");; ycsbi) args=("${ARGSI[@]}");; ycsbw) args=("${ARGSW[@]}");; esac
    run /tmp/sem_${wl}_fast.log -- "${args[@]}"
    run /tmp/sem_${wl}_general.log EPIC_TIMELINE_FORCE_GENERAL=1 -- "${args[@]}"
    hf=$(hashes /tmp/sem_${wl}_fast.log); hg=$(hashes /tmp/sem_${wl}_general.log)
    if [ -n "$hf" ] && [ "$hf" = "$hg" ]; then
        echo "[$wl] forced timeline: hashes identical ($hf)"
    else
        echo "[$wl] forced timeline: hashes DIFFER (fast: ${hf:-none}; general: ${hg:-none})"; fail=1
    fi
done
echo "----------------------------------------------------------------------"
[ "$fail" -eq 0 ] && echo "RESULT: delete semantics PASS" || echo "RESULT: delete semantics FAILED"
exit $fail
