//
// reclaim_flags.cuh -- shared reclaim-first eviction kernels.
//
// A stager can flag cache slots whose records are logically dead (deleted
// keys; TPC-C's delivered OrderLine rows are the same pattern) so its
// eviction pass drains them before touching live residents. The flag is a
// preference, not a pin: flagged slots still honor the needed-set
// protection and the in-flight-writeback pin, which is what keeps
// reclaiming race-free -- a dead record dirtied in its final epoch stays
// pinned until its writeback lands, and one that the current epoch still
// reads stays needed-protected until the epoch ends.
//

#ifndef EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH
#define EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH

#include <cstdint>

#ifdef EPIC_CUDA_AVAILABLE

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
// tops up from where this left off. No cursor bookkeeping: reclaimed
// slots are off-order by design.
static __global__ void k_collect_evictions_reclaim_first(uint32_t cap,
                                                         const uint32_t* __restrict__ resident_list,
                                                         const uint8_t* __restrict__ needed_flag,
                                                         const uint8_t* __restrict__ flush_pinned_flag,
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
        if (needed_flag[g] || flush_pinned_flag[g]) continue;
        if (!reclaim_flag[g]) continue;
        uint32_t pos = atomicAdd(out_count, 1u);
        if (pos < deficit) {
            out_grids[pos] = g;
            out_crids[pos] = crid;
        }
    }
}

} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_BENCHMARKS_RECLAIM_FLAGS_CUH
