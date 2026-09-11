//
// reclaim_flags.cuh -- shared reclaim-first eviction kernels.
//
// A stager can flag cache slots whose records are logically dead (deleted
// keys; TPC-C's delivered OrderLine rows are the same pattern) so its
// eviction pass drains them before touching live residents. The flag is a
// preference, not an override: flagged slots still honor the needed-set
// protection, so a dead record that the current epoch still reads stays
// resident until the epoch ends. A dead record dirtied in its final epoch
// is safe to evict afterwards because the writeback packs the flush set
// out of the cache at collect time, before the next eviction runs.
//

#ifndef EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH
#define EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef EPIC_CUDA_AVAILABLE

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace epic {

// Flag the cache slots of a CRID list (resolved through the flat cache
// index). Non-resident CRIDs resolve to the sentinel and need nothing.
static __global__ void k_mark_reclaim_by_crids(const uint32_t* __restrict__ crids,
                                               uint32_t n,
                                               const uint32_t* __restrict__ d_crid_to_grid,
                                               uint32_t num_units,
                                               uint32_t cap,
                                               uint8_t* __restrict__ reclaim_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t crid = crids[i];
    if (crid == 0xffffffffu || crid >= num_units) return;
    uint32_t g = d_crid_to_grid[crid];
    if (g != 0xffffffffu && g < cap) {
        reclaim_flag[g] = 1;
    }
}

// Clear the flag for a list of GRIDs. Run after eviction renames so the
// slot's new occupant starts unflagged; without it the stale flag would
// drain the new resident on the next sweep.
static __global__ void k_clear_reclaim_by_grids(const uint32_t* __restrict__ grids,
                                                uint32_t n,
                                                uint32_t cap,
                                                uint8_t* __restrict__ reclaim_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t g = grids[i];
    if (g != 0xffffffffu && g < cap) {
        reclaim_flag[g] = 0;
    }
}

// Reclaim-first victim collection: pick flagged slots only, sharing the
// out_count atomic with the FIFO scan that runs after it -- if this pass
// fills the deficit the FIFO scan early-exits, otherwise the FIFO scan
// tops up from where this left off. A picked slot is also marked in the
// epoch's needed flags, which the FIFO scan skips, so no slot is picked by
// both passes (two picks would bind two records to one slot). No cursor
// bookkeeping: reclaimed slots are off-order by design.
static __global__ void k_collect_evictions_reclaim_first(uint32_t cap,
                                                         const uint32_t* __restrict__ resident_list,
                                                         uint8_t* __restrict__ needed_flag,
                                                         const uint8_t* __restrict__ reclaim_flag,
                                                         uint32_t deficit,
                                                         uint32_t* __restrict__ out_grids,
                                                         uint32_t* __restrict__ out_crids,
                                                         uint32_t* __restrict__ out_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    if (atomicAdd(out_count, 0u) >= deficit) return;
    for (uint32_t g = tid; g < cap; g += stride) {
        if (atomicAdd(out_count, 0u) >= deficit) break;
        uint32_t crid = resident_list[g];
        if (crid == 0xffffffffu) continue;
        if (needed_flag[g]) continue;
        if (!reclaim_flag[g]) continue;
        uint32_t pos = atomicAdd(out_count, 1u);
        if (pos < deficit) {
            out_grids[pos] = g;
            out_crids[pos] = crid;
            needed_flag[g] = 1;
        }
    }
}

#ifdef EGAD_VALIDATION
// Validation build: an eviction list must not name a slot twice.
inline void checkEvictionListDistinct(const uint32_t* d_grids, uint32_t n, const char* stager)
{
    if (n < 2) return;
    thrust::device_vector<uint32_t> grids(d_grids, d_grids + n);
    thrust::sort(grids.begin(), grids.end());
    if (thrust::unique(grids.begin(), grids.end()) != grids.end()) {
        throw std::runtime_error(std::string("[EVICT-CHECK] FAILED: ") + stager +
                                 " eviction list names a cache slot twice");
    }
}
#endif

} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH
