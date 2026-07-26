//
// Created by Shujian Qian on 2023-09-15.
//

#ifndef UTIL_MEMORY_H
#define UTIL_MEMORY_H

#include <cstdlib>
#include <cstdint>
#include <cstring>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace epic {

struct FreeDelete
{
    void operator()(void *ptr)
    {
        free(ptr);
    }
};

void *allocatePinnedMemory(size_t size);

// Durable allocation for the GPU-crash-recovery experiment.
// When EPIC_DURABLE_STORE=<dir> is set, the region is backed by a MAP_SHARED
// mmap of <dir>/<name>.bin (ftruncate'd to size) + cudaHostRegister (pinning
// a tmpfs-backed mmap is supported), so the bytes survive the
// process dying. When EPIC_RECOVER_FROM=<dir> is set, the existing
// <dir>/<name>.bin is re-mapped as-is (NOT zeroed) so a fresh process can
// reload the durable state. When neither env is set, this falls straight
// through to Malloc -- the normal path is byte-identical (gated, recovery
// experiment only). `name` keys the file so the recover process re-maps the
// same region.
// pin=false skips cudaHostRegister; the growing tables (never a cudaMemcpy
// endpoint) instead get eager page-commit provisioning on create (the ship
// config), avoiding a lazy first-touch stall mid-run.
void *MallocDurable(const char *name, size_t size, bool pin = true);

#ifdef EGAD_VALIDATION
// GPU-crash fault injector (validation harness only): launches a kernel that
// performs a deterministic illegal device access, syncs (-> cudaErrorIllegalAddress),
// and terminates the process non-zero, modeling a real GPU fault that a supervisor
// detects and recovers from in a fresh process. Co-located here only because
// util_memory.cu is an already-compiled .cu (avoids adding a new build target).
void injectGpuFault();
#endif

inline void *Malloc(size_t size)
{
    void *retval = nullptr;
#ifdef EPIC_CUDA_AVAILABLE
    retval = allocatePinnedMemory(size);
#else
    retval = malloc(size);
#endif
    memset(retval, 0, size);
    return retval;
}

// Pageable variant: returns memory backed by ordinary malloc (no
// cudaHostAlloc, no upfront page commitment). The caller can still
// access the buffer normally; the OS lazily backs each touched page
// with physical RAM. Drops the memset so untouched pages stay
// uncommitted -- this is the load-bearing difference for the TPC-C
// growing-table primary stores, where allocation is sized for the
// worst-case insert mix (every txn = NewOrder, every NewOrder = 15
// OL rows) but the actual touched pages are far fewer.
//
// Cost: pages have to be page-faulted in on first access (~few us per
// 4 KB page). Steady-state access is then full memory bandwidth, same
// as pinned. Suitable for buffers that are read by host CPU only;
// NOT suitable as a direct source for cudaMemcpyAsync (which would
// fall back to a synchronous staged copy).
inline void *MallocPageable(size_t size)
{
    void *retval = malloc(size);
    // Intentionally no memset: untouched pages stay un-faulted-in.
    // The Record<X> defaults zero-initialize anyway when the executor
    // first writes to them, so callers do not depend on the buffer
    // being pre-zeroed.
    return retval;
}

// MallocPageable + madvise(MADV_HUGEPAGE) on the region. For the growing
// primary stores whose writeback walks CRIDs near-sequentially (insert
// allocation is sequential), first-touch then costs one fault per 2 MB
// instead of one per 4 KB, and the flush scatter stops paying a page-fault
// plus page-walk per row. Laziness is preserved: pages still commit only
// when touched, just at 2 MB granularity, so this is only for stores with
// DENSE touch patterns (OL/O/NO append-style writeback). Sparse random-
// update stores (stock, customer) must stay 4 KB: a 2 MB granule per
// touched row would balloon RSS by orders of magnitude.
// EPIC_NO_HUGE_GROWING=1 disables the madvise (plain MallocPageable).
inline void *MallocPageableHuge(size_t size)
{
    void *retval = MallocPageable(size);
#ifdef MADV_HUGEPAGE
    static const bool disabled = std::getenv("EPIC_NO_HUGE_GROWING") != nullptr;
    if (retval != nullptr && !disabled) {
        // Large allocations come from mmap and are page-aligned, but align
        // inward anyway so madvise never sees a partial page.
        const uintptr_t lo = (reinterpret_cast<uintptr_t>(retval) + 4095) & ~uintptr_t(4095);
        const uintptr_t hi = (reinterpret_cast<uintptr_t>(retval) + size) & ~uintptr_t(4095);
        if (hi > lo) madvise(reinterpret_cast<void *>(lo), hi - lo, MADV_HUGEPAGE);
    }
#endif
    return retval;
}

inline void FreePageable(void *ptr)
{
    free(ptr);
}

void freePinedMemory(void *ptr);
inline void Free(void *ptr)
{
#ifdef EPIC_CUDA_AVAILABLE
    freePinedMemory(ptr);
#else
    free(ptr);
#endif
}

} // namespace epic

#endif // UTIL_MEMORY_H
