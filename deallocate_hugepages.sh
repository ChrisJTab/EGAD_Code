#!/usr/bin/env bash
set -euo pipefail

echo "== HugeTLB current =="
grep -i Huge /proc/meminfo || true

echo "== Setting global nr_hugepages to 0 (requires sudo) =="
sudo sh -c 'echo 0 > /proc/sys/vm/nr_hugepages' || true

# Per-NUMA-node (some systems use per-node reservation)
for f in /sys/devices/system/node/node*/hugepages/hugepages-*/nr_hugepages; do
  [ -e "$f" ] || continue
  echo "== Setting $f to 0 =="
  sudo sh -c "echo 0 > '$f'" || true
done

echo "== HugeTLB after =="
grep -i Huge /proc/meminfo || true
