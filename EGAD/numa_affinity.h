// numa_affinity.h
//
// Helper for pinning scatter/flush worker threads to node-1 physical cores.
// A naive upper-half-of-online-CPUs heuristic on HT-enabled multi-socket
// machines pins workers to HT siblings spanning both NUMA nodes (and doubles
// the OMP scatter pool's thread count). The cross-socket worker placement is
// load-bearing for throughput: the worker's CPU scatter must not contend on
// node-0 DRAM channels with the main thread's SG gather.
//
// Strategy:
//   - If libnuma reports node 1, query its CPUs and filter to one per physical
//     core via /sys/devices/system/cpu/cpu<N>/topology/core_id.
//   - Otherwise fall back to upper-half-of-online-CPUs, which on a
//     single-socket box is the best we can do without cross-NUMA placement.

#ifndef EPIC_NUMA_AFFINITY_H_
#define EPIC_NUMA_AFFINITY_H_

#include <numa.h>
#include <unistd.h>
#include <cstdio>
#include <set>
#include <vector>

namespace epic {

static inline int read_core_id_for_cpu(int cpu) {
    char path[256];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    std::FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    int core_id = -1;
    if (std::fscanf(f, "%d", &core_id) != 1) core_id = -1;
    std::fclose(f);
    return core_id;
}

static inline const std::vector<int>& scatter_worker_cpus() {
    static const std::vector<int> kCpus = []() {
        std::vector<int> cpus;
        if (numa_available() != -1 && numa_max_node() >= 1) {
            struct bitmask* node1_mask = numa_allocate_cpumask();
            if (node1_mask && numa_node_to_cpus(1, node1_mask) == 0) {
                std::set<int> seen_cores;
                int n_cpus = numa_num_configured_cpus();
                for (int cpu = 0; cpu < n_cpus; ++cpu) {
                    if (!numa_bitmask_isbitset(node1_mask, (unsigned)cpu)) continue;
                    int core_id = read_core_id_for_cpu(cpu);
                    if (core_id < 0) {
                        // sysfs unreadable -- accept the cpu rather than skip.
                        cpus.push_back(cpu);
                    } else if (seen_cores.insert(core_id).second) {
                        // First cpu we have seen for this physical core; keep it.
                        cpus.push_back(cpu);
                    }
                    // Otherwise: HT sibling of an already-kept core; skip.
                }
            }
            if (node1_mask) numa_free_cpumask(node1_mask);
        }
        if (cpus.empty()) {
            // Single-NUMA / no-libnuma fallback: upper half of online CPUs.
            long n = sysconf(_SC_NPROCESSORS_ONLN);
            if (n >= 2) {
                for (int cpu = static_cast<int>(n / 2); cpu < static_cast<int>(n); ++cpu) {
                    cpus.push_back(cpu);
                }
            } else {
                cpus.push_back(0);
            }
        }
        return cpus;
    }();
    return kCpus;
}

}  // namespace epic

#endif  // EPIC_NUMA_AFFINITY_H_
