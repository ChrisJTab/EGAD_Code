//
// Created by Christian Tabbah on 2025-09-22.
//

#ifndef EPIC__TPCC_HYBRID_STAGER_H
#define EPIC__TPCC_HYBRID_STAGER_H

#include <cstdint>
#include <any>
#include <benchmarks/tpcc_storage.h>
#include <scatter_gather.h>
#include <benchmarks/tpcc_txn.h>
#include "allocator.h"
#include "device_ring_types.h"
#include "crid_grid_index.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "flush_handle.h"

// cudaStream_t is forward-declared in crid_grid_index.h (included above via
// the CridGridIndex member declaration); we re-use that typedef here for
// active_stream() and prep_stream_.

namespace epic::tpcc {

// CridGridIndex and FlushHandle live in epic::ycsb (the YCSB hybrid path
// authored them; TPC-C reuses them as-is, one instance per table). Bring
// them into epic::tpcc so the rest of the file can refer to them unqualified
// — same shape as the YCSB stager.
using ::epic::ycsb::CridGridIndex;
using ::epic::ycsb::FlushHandle;


struct Timeline {
    struct Point { const char* name; int64_t us; };
    std::vector<Point> pts;
    int64_t epoch_start_us = 0;

    void reset() {
        pts.clear();
        using namespace std::chrono;
        epoch_start_us = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }
    void mark(const char* name) {
        using namespace std::chrono;
        pts.push_back({name, duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()});
    }
    void dump(uint32_t epoch) {
        if (pts.size() < 2) return;
        std::string line = "[TL] epoch=" + std::to_string(epoch);
        for (size_t i = 0; i < pts.size(); ++i) {
            int64_t rel = pts[i].us - epoch_start_us;
            if (i > 0) {
                int64_t gap = pts[i].us - pts[i-1].us;
                line += " |" + std::to_string(gap) + "| ";
            }
            line += std::string(pts[i].name) + "@" + std::to_string(rel);
        }
        epic::Logger::GetInstance().Info("{}", line);
    }
};

// Forward decl: persistent flush worker (definition in tpcc_hybrid_stager.cu).
// Each TpccHybridStager owns one of these via unique_ptr; it spawns a single
// long-lived std::thread pinned to node 1 that picks up writeback work via CV
// signaling, eliminating per-epoch thread spawn cost.
class TpccPersistentFlushWorker;

// Per-table hybrid stager for TPC-C, parametrised on the table's record / version
// types. Adapted from epic::ycsb::HybridStager (see ycsb_hybrid_stager.{h,cu}).
// TPC-C never enables field-split, so the split_field branches in the YCSB
// version are dropped here.
//
// Each writeable TPC-C table gets one TpccHybridStager instance (History gets
// none; the executor neither reads nor writes it). Constructor takes
// table-local pointers; nothing reaches into globals.
template <typename TRec, typename TVer>
class TpccHybridStager
{
public:
    // enable_reclaim_eviction activates the reclaim-first eviction pass
    // (reclaim_flags.cuh): flagged logically-dead slots are drained before
    // live residents. Two flag producers exist -- the Delivery executor
    // marks delivered OrderLine slots directly through the plumbed flag
    // pointer, and mark_reclaimable flags deleted NewOrder rows' slots --
    // both feed the same per-slot flag and eviction pre-pass.
    TpccHybridStager(Allocator& alloc, TRec* GPU_records, TVer* GPU_versions,
        TRec* CPU_records,
        uint32_t gpu_capacity, uint32_t num_units,
        bool enable_reclaim_eviction = false);

    ~TpccHybridStager();

    // Run before executor->execute()
    // If flush_handle is non-null, spawns scatter worker after SG transfer completes.
    // d_insert_keys is sentinel-padded parallel to d_all_keys, with the CRID
    // when the corresponding op was an INSERT and the sentinel otherwise. Used
    // to split admission into miss-admit (needs SG) and insert-allocate (skip
    // the SG transfer since the cache slot's data will be produced by the
    // executor's INSERT write, not the CPU primary store). May be nullptr, in
    // which case every new CRID takes the regular miss-admit path.
    //
    // Unlike the YCSB stager this does NOT take TxnArray references — the
    // CRID→GRID Remap is per-txn-type for TPC-C and lives in
    // tpcc_hybrid_remap.{h,cu}. The caller fetches the CridGridIndex
    // through crid_to_grid_index() and feeds it to the dispatch kernel.
    void prepareEpoch(uint32_t epoch,
    const uint32_t* d_all_keys, uint32_t n_all,
    const uint32_t* d_read_keys, uint32_t n_read,
    const uint32_t* d_write_keys, uint32_t n_write,
    const uint32_t* d_insert_keys, uint32_t n_insert,
    FlushHandle* flush_handle = nullptr);
    void periodicFlush(uint32_t epoch);

    // ----------------------------
    // Flush pipeline (overlap flush(E) with epoch E+1 work)
    // ----------------------------

    // Called near the start of epoch E (before eviction selection / staging).
    // If inflight == nullptr or !inflight->valid, no pins are applied.
    // This updates the internal "pinned" set used by FIFO eviction to avoid evicting
    // records that are being flushed in the background.
    void set_flush_pins_for_next_epoch(const FlushHandle* inflight);

    // Called right after execution(E) ends to prepare flush(E): collect the
    // dirty set, sort, D2D + pack into the flush SG's device buffer, clear
    // the dirty bits, and pin the flush set for epoch E+1's eviction. Fills
    // out_handle with the metadata later phases need. Does NOT start the
    // D2H + host scatter; submit_flush_worker does. The split lets the epoch
    // loop advance the durable recovery marker at the one sound point:
    // after every stager's E-1 drain (end-of-(E-1) fully durable) and
    // before any byte of epoch E reaches the Primary Store.
    void start_flush_epoch_async(uint32_t epoch, FlushHandle& out_handle);

    // Hand the prepared flush set to the persistent worker, which drives the
    // chunked D2H + host scatter into the Primary Store. Must be called
    // after start_flush_epoch_async and, in durable mode, after the marker
    // bump (the first scatter byte overwrites an end-of-(E-2) slot, which is
    // unrecoverable under a marker that still reads E). No-op when the
    // handle is invalid (empty flush set).
    void submit_flush_worker(FlushHandle& handle);

    // Called near the end of epoch E to enforce the "deadline" that flush(E-1) is complete.
    // This should block until the async flush recorded in 'handle' is fully finished.
    void sync_flush(FlushHandle& handle);

    // --- Granular async flush phase control ---
    // Call from the epoch loop at specific points to avoid contention.
    void flush_ensure_buffers(FlushHandle& handle);       // allocate SG buffers
    void flush_queue_h2d(FlushHandle& handle);            // queue H2D grids+slots (non-blocking)
    void flush_queue_pack(FlushHandle& handle);           // queue pack kernel (non-blocking)
    void flush_sync_gpu();                                // sync flush stream (H2D + pack done)

    // Cumulative PCIe payload bytes pushed by all owned ScatterGather instances
    // since constructor. Sums across the admit / async-flush instances.
    // Reported by the benchmark's end-of-run summary.
    uint64_t pcie_admit_h2d_bytes() const;
    uint64_t pcie_writeback_d2h_bytes() const;

    // The per-txn-type Remap needs per-table CRID→GRID lookups; expose the
    // index by const reference so the Remap kernels can look up against it
    // without owning the stager.
    const CridGridIndex& crid_to_grid_index() const { return crid2grid_index_; }

    // The CUDA stream prepareEpoch and the flush setup issue work on: the
    // per-stager stream created in the constructor.
    cudaStream_t active_stream() const { return prep_stream_; }

    // Cumulative cache-slot accounting since constructor. cache_slots_admitted
    // counts every CRID that was newly bound to a GRID (whether the GRID came
    // from the free-list ring or from evicting a prior CRID). cache_slots_evicted
    // counts only the eviction-driven subset — slots whose prior CRID was kicked
    // out to make room for a new one. cache_slots_admitted - cache_slots_evicted
    // equals the number of pure ring-pops. miss_admit / insert_allocate split
    // the admitted count by whether the slot's data came from an SG transfer
    // or from the executor's INSERT write.
    uint64_t cache_slots_admitted_total() const { return cache_slots_admitted_total_; }
    uint64_t cache_slots_evicted_total()  const { return cache_slots_evicted_total_; }
    uint64_t cache_slots_miss_admit_total()      const { return cache_slots_miss_admit_total_; }
    uint64_t cache_slots_insert_allocate_total() const { return cache_slots_insert_allocate_total_; }
private:
    Allocator& alloc_;

    // Cache-slot accounting. See public getters above for semantics.
    uint64_t cache_slots_admitted_total_ = 0;
    uint64_t cache_slots_evicted_total_  = 0;

    TRec* GPU_records_;
    TVer* GPU_versions_;
    TRec* CPU_records_;

    uint32_t gpu_capacity_;
    uint32_t num_units_; // CRID universe size for this table
    std::unique_ptr<ScatterGather<TRec>>      sg_record_;

    //Used for async flush
    std::unique_ptr<ScatterGather<TRec>>     sg_record_flush_;
    std::once_flag flush_sg_once_flag_;

    // Persistent flush worker (definition in .cu). Created in ctor, destroyed
    // in dtor. Per-stager: each of the 8 stagers gets its own thread.
    std::unique_ptr<TpccPersistentFlushWorker> persistent_worker_;



    // Device lists and counters
    uint32_t * d_needed_ = nullptr; // array of records that are needed for the current batch
    uint32_t * d_new_ = nullptr;    // array of records that are new to the GPU (not resident)
    uint32_t * d_resident_list_ = nullptr; // array of size gpu_capacity_ holding the CRIDs that are currently resident in GPU

    uint32_t * d_needed_count_ = nullptr;
    uint32_t * d_new_count_ = nullptr;

    uint32_t * first_needed_count_ = nullptr;


    // Free-list ring. The device-side ring functions live in
    // util_device_ring.cuh (CUDA-only); this header keeps the plain view
    // struct so it stays includable from C++ TUs.
    DeviceRingView free_list_ring_{};
    uint32_t* d_ring_buffer_ = nullptr;
    unsigned long long* d_ring_head_ = nullptr;
    unsigned long long* d_ring_tail_ = nullptr;
    // 8-byte scratch for the two-phase parallel ring_pop (see util_device_ring.cuh).
    unsigned long long* d_ring_pop_scratch_ = nullptr;

    //index to map CRID to GRID
    CridGridIndex crid2grid_index_;

    // Per-stager CUDA stream for prepareEpoch and the flush setup, created
    // in the constructor (a REGULAR stream; see the constructor comment for
    // why non-blocking would race the remap) and destroyed in the destructor.
    cudaStream_t prep_stream_ = 0;

    // FIFO eviction cursor (next GRID to consider)
    uint32_t fifo_next_grid_ = 0;
    uint8_t* d_needed_grid_flag_ = nullptr; // temporary array to mark which grids are needed

    // buildNeeded scratch (cached)
    uint32_t* d_keys_in_  = nullptr;   // size >= max n_keys seen
    uint32_t* d_keys_out_ = nullptr;   // size >= max n_keys seen
    uint32_t  keys_cap_   = 0;

    void*  d_sort_tmp_      = nullptr;
    size_t sort_tmp_bytes_  = 0;

    void*  d_unique_tmp_     = nullptr;
    size_t unique_tmp_bytes_ = 0;

    // periodicFlush scratch (cached)
    uint32_t* d_dirty_grids_ = nullptr;   // size = gpu_capacity_
    uint32_t* d_dirty_crids_ = nullptr;   // size = gpu_capacity_
    uint32_t* d_dirty_grids_slot1_ = nullptr;
    uint32_t* d_dirty_crids_slot1_ = nullptr;

    // Host periodicFlush scratch (persistent, reused)
    std::vector<uint32_t> h_dirty_grids_;
    std::vector<uint32_t> h_dirty_crids_;
    size_t h_dirty_cap_ = 0;

    // Skip-admit support. d_insert_bitmap_ is a num_units_-sized byte array
    // marking which CRIDs are insert targets in the current epoch (set from
    // d_insert_keys at the start of prepareEpoch, reset to 0 between epochs).
    // d_new_is_insert_ is a per-d_new_ flag (size = gpu_capacity_) populated
    // by looking up each new CRID in the bitmap; used by the admission step
    // to split d_new_ into the miss-admit (regular SG) and insert-allocate
    // (skip-SG) subsets. d_new_insert_count_ holds the GPU-side count of
    // insert-allocate slots in d_new_ for the current epoch.
    uint8_t*  d_insert_bitmap_  = nullptr;
    uint8_t*  d_new_is_insert_  = nullptr;
    uint32_t* d_new_insert_count_ = nullptr;
    // Cumulative split of cache_slots_admitted into miss-admit vs
    // insert-allocate, for end-of-run reporting.
    uint64_t  cache_slots_miss_admit_total_   = 0;
    uint64_t  cache_slots_insert_allocate_total_ = 0;

    // CUB select scratch (flag array + temp storage) for the
    // eviction-path stream compaction
    uint8_t* d_filter_flags_ = nullptr;  // gpu_capacity_ bytes
    void*    d_filter_cub_tmp_ = nullptr;
    size_t   d_filter_cub_tmp_bytes_ = 0;

    // Flush-set CRID sort scratch (start_flush_epoch_async). Each slot's
    // (dirty CRID -> GRID) pair list is radix-sorted ascending by CRID so
    // the flush worker's host scatter walks the Primary Store sequentially
    // instead of in GRID (ring/eviction) order. Lazily allocated to
    // gpu_capacity_ on first flush, reused every epoch.
    uint32_t* d_flush_sort_crids_ = nullptr;
    uint32_t* d_flush_sort_grids_ = nullptr;
    void*     d_flush_sort_tmp_ = nullptr;
    size_t    d_flush_sort_tmp_bytes_ = 0;
    bool      flush_sort_ready_ = false;

    // Reclaim-first eviction. When enabled, every cache slot has a 1-byte
    // flag marking its record as logically dead (a delivered OrderLine
    // row, a deleted NewOrder row); the eviction pass drains flagged
    // slots ahead of live residents, and renames clear the flag so a
    // stale 1 never drains the slot's next occupant.
    bool      reclaim_eviction_active_ = false;
public:
    // Public getter so tpcc.cpp can plumb the pointer into TpccRecords
    // for the Delivery executor to write to (delivered OL slots).
    // Returns nullptr if this stager did not enable reclaim eviction.
    uint8_t* reclaim_flag_ptr() const { return d_reclaim_flag_; }

    // Flag the cache slots of a deleted-CRID list reclaim-first (the
    // NewOrder stager's producer; Delivery's OL marking goes through the
    // plumbed flag pointer instead). No-op when reclaim eviction is
    // disabled or n is 0.
    void mark_reclaimable(const uint32_t* d_crids, uint32_t n);

    // Per-slot dirty-flag array pointers from the stager's CridGridIndex.
    // tpcc.cpp wires these into TpccRecords so the executor can mark dirty
    // inline (replacing the staging-time mark_dirty kernels). One byte per
    // cache slot for each of the two version slots.
    uint8_t* dirty_v1_ptr() const { return crid2grid_index_.device_view().grid_dirty_v1; }
    uint8_t* dirty_v2_ptr() const { return crid2grid_index_.device_view().grid_dirty_v2; }

    // One-shot pre-warm: bring the OL records of the spec's initial
    // undelivered orders (per district, NO_O_ID in [2101, 3000]) onto the
    // GPU before epoch 1. Removes the 25-epoch warmup hump where Delivery
    // would otherwise fetch each batch on demand. Caller must invoke once
    // after stager construction and table loading. Called only on the OL
    // stager (canonical default; tpcc.cpp skips it only for the recover process).
    void prewarm_initial_undelivered_orderline(uint32_t num_warehouses);

private:
    uint8_t*  d_reclaim_flag_ = nullptr;  // gpu_capacity_ bytes; 1 = logically dead, drain first

    // GPU-side stream compaction for the eviction-path miss-admit subset.
    // Replaces the previous "D2H all deficit items + CPU loop" with a single
    // CUB::DeviceSelect::Flagged on GPU, then D2H of just the filtered output.
    // d_filter_flags_ holds the inverted-is_insert flag and
    // d_filter_cub_tmp_ the CUB temp storage (both gpu_capacity-sized).
    // d_ev_cr_miss_ / d_ev_gr_miss_ are lazy-grown to the observed deficit.
    uint32_t* d_ev_cr_miss_    = nullptr;
    uint32_t* d_ev_gr_miss_    = nullptr;
    uint32_t* d_ev_miss_count_ = nullptr;
    size_t    d_ev_buffer_cap_ = 0;

    // Pre-allocated scratch buffers for prepareEpoch (avoid cudaMalloc/cudaFree per epoch)
    // Each is gpu_capacity_ * sizeof(uint32_t) bytes. Reused across non-overlapping scopes.
    uint32_t* d_scratch_a_ = nullptr;  // d_needed_grids, d_grids, d_k_grids, d_grids2, d_write_grids
    uint32_t* d_scratch_b_ = nullptr;  // d_evict_crids (long-lived)
    uint32_t* d_scratch_c_ = nullptr;  // d_evict_grids (long-lived)
    uint32_t* d_scratch_d_ = nullptr;  // CUB size-query scratch
    uint32_t* d_scratch_f_ = nullptr;  // d_k_crids
    // Small scalar scratch (4 bytes each)
    uint32_t* d_scalar_a_ = nullptr;   // used as: d_granted, d_granted2, d_miss
    uint32_t* d_scalar_b_ = nullptr;   // used as: d_cnt
    uint32_t* d_scalar_c_ = nullptr;   // used as: d_maxoffset, d_k_count

    void ensureHostDirtyCapacity(size_t n);
    // Helpers:
    void buildNeededOnDevice(const uint32_t* d_all_keys, uint32_t n_keys, uint32_t epoch, cudaStream_t stream = 0);

    // SG helpers — TPC-C uses a single ScatterGather<TRec> instance per stager
    // (no field-split path), so the YCSB-style split_field dispatch is gone.
    void sg_transfer_versions(const uint32_t* h_crids, const uint32_t* h_grids, uint32_t n, uint32_t epoch);
    void sg_sync();
    void* sg_stream(); // returns the cuda stream (as void*) of sg_record_

    // ----------------------------
    // Pinned (in-flight flush) set for eviction skipping
    // ----------------------------
    // Device flag: d_flush_pinned_flag_[grid] == 1 => grid is pinned (cannot be evicted)
    uint8_t*  d_flush_pinned_flag_ = nullptr;

    // --- reusable host vectors for pre-SG D2H (avoid per-epoch heap alloc) ---
    // Pinned host buffers for pre-SG D2H (avoids copy engine serialization with worker D2H)
    uint32_t* h_evict_crids_ = nullptr;
    uint32_t* h_evict_grids_ = nullptr;
    size_t    h_evict_cap_   = 0;

    uint8_t* d_clear_slots0_ = nullptr; // size gpu_capacity_
    uint8_t* d_clear_slots1_ = nullptr; // size gpu_capacity_

    // Mapped host memory for scalar readbacks (bypasses D2H copy engine).
    // Device pointers (d_*) stay as device memory for fast kernel atomics/CUB.
    // k_copy_to_mapped copies the result to mapped memory after the kernel.
    static constexpr int kMappedScalarCount_ = 13;  // [0..7,12] uint32_t scalars, [8..11] ring head/tail (ull)
    uint32_t* h_stager_mapped_ = nullptr;           // host block [0..12]
    uint32_t* d_mapped_needed_count_     = nullptr;  // [0]
    uint32_t* d_mapped_load_needed_count_= nullptr;  // [1]
    uint32_t* d_mapped_write_only_count_ = nullptr;  // [2]
    uint32_t* d_mapped_new_count_        = nullptr;  // [3]
    uint32_t* d_mapped_scalar_a_         = nullptr;  // [4] (d_granted, d_granted2, d_miss)
    uint32_t* d_mapped_scalar_b_         = nullptr;  // [5] (d_cnt)
    uint32_t* d_mapped_scalar_c_         = nullptr;  // [6] (d_maxoffset)
    uint32_t* d_mapped_write_unique_     = nullptr;  // [7]
    uint32_t* d_mapped_ring_head_        = nullptr;  // [8..9] (unsigned long long)
    uint32_t* d_mapped_ring_tail_        = nullptr;  // [10..11] (unsigned long long)
    uint32_t* d_mapped_new_insert_count_ = nullptr;  // [12] (skip-admit subset count)

    // Per-epoch timeline for instrumentation
    Timeline tl_;
};

}






#endif // EPIC__TPCC_HYBRID_STAGER_H