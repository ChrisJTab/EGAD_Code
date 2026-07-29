//
// Created by Christian Tabbah on 2025-09-22.
//

#ifndef EPIC__YCSB_HYBRID_STAGER_H
#define EPIC__YCSB_HYBRID_STAGER_H

#include <cstdint>
#include <any>
#include <benchmarks/ycsb_storage.h>
#include <scatter_gather.h>
#include <benchmarks/ycsb_txn.h>
#include "allocator.h"
#include "device_ring_types.h"
#include "crid_grid_index.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "flush_handle.h"

namespace epic::ycsb {



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

class HybridStager
{
public:
    // enable_reclaim_eviction activates the reclaim-first eviction pass
    // for delete-bearing mixes (reclaim_flags.cuh): deleted records'
    // slots are flagged via mark_reclaimable and drained before live
    // residents. Off for mixes without deletes (no flag storage, no
    // extra eviction pass).
    HybridStager(Allocator& alloc, YcsbRecordArrType GPU_records, YcsbVersionArrType GPU_versions,
        YcsbRecordArrType CPU_records,
        uint32_t gpu_capacity, uint32_t num_units, bool split_field,
        bool enable_reclaim_eviction = false);

    ~HybridStager();

    // Compute how many record slots can fit in the given GPU memory budget.
    // Accounts for all per-slot metadata arrays allocated by the constructor
    // (stager lists, scratch buffers, index arrays, etc.).
    // This lives here rather than in ycsb.cpp so the overhead stays in sync
    // with the actual allocations in the constructor.
    static uint32_t compute_staging_capacity(size_t usable_bytes,
                                             size_t per_unit_payload,
                                             uint32_t max_units);

    // Expose the per-slot dirty-flag arrays so HybridExecutor can mark each
    // slot as it commits a write: one byte store inside gpuWriteToTableCoop /
    // gpuWriteToTableThread, with no staging-time dirty pipeline.
    uint8_t* dirty_v1_ptr() const { return crid2grid_index_.device_view().grid_dirty_v1; }
    uint8_t* dirty_v2_ptr() const { return crid2grid_index_.device_view().grid_dirty_v2; }

    // Run before executor->execute()
    // If flush_handle is non-null, spawns scatter worker after SG transfer completes.
    // d_insert_keys is sentinel-padded parallel to d_all_keys, with the CRID
    // when the corresponding op was an INSERT and the sentinel otherwise. Used
    // to split admission into miss-admit (needs SG) and insert-allocate (skip
    // the SG transfer since the cache slot's data will be produced by the
    // executor's INSERT write, not the CPU primary store). May be nullptr, in
    // which case every new CRID takes the regular miss-admit path.
    void prepareEpoch(uint32_t epoch, TxnArray<YcsbTxnParam>& txn_array, TxnArray<YcsbExecPlan>& plan_array,
    const uint32_t* d_all_keys, uint32_t n_all,
    const uint32_t* d_read_keys, uint32_t n_read,
    const uint32_t* d_write_keys, uint32_t n_write,
    const uint32_t* d_insert_keys, uint32_t n_insert,
    FlushHandle* flush_handle = nullptr);
    void periodicFlush(uint32_t epoch);

    // Flag the cache slots of this epoch's deleted CRIDs reclaim-first.
    // Called before prepareEpoch each epoch on delete-bearing mixes; the
    // eviction pass then drains flagged slots ahead of live residents
    // once their needed-set protection and writeback pins lapse. No-op
    // when reclaim eviction is disabled or n is 0.
    void mark_reclaimable(const uint32_t* d_crids, uint32_t n);

    // ----------------------------
    // Flush pipeline (overlap flush(E) with epoch E+1 work)
    // ----------------------------

    // Called near the start of epoch E (before eviction selection / staging).
    // If inflight == nullptr or !inflight->valid, no pins are applied.
    // This updates the internal "pinned" set used by FIFO eviction to avoid evicting
    // records that are being flushed in the background.
    void set_flush_pins_for_next_epoch(const FlushHandle* inflight);

    // Called right after execution(E) ends to begin flush(E) asynchronously.
    // Fills out_handle with events/metadata needed for later sync and for pinning in epoch E+1.
    void start_flush_epoch_async(uint32_t epoch, FlushHandle& out_handle);

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
    // since constructor. Sums across the per-table-record and per-field-record
    // SGs and across the admit / async-flush instances. Reported by the
    // benchmark's end-of-run summary.
    uint64_t pcie_admit_h2d_bytes() const;
    uint64_t pcie_writeback_d2h_bytes() const;

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
    bool split_field_ = false;

    // Cache-slot accounting. See public getters above for semantics.
    uint64_t cache_slots_admitted_total_ = 0;
    uint64_t cache_slots_evicted_total_  = 0;

    YcsbRecordArrType GPU_records_;
    YcsbVersionArrType GPU_versions_;
    YcsbRecordArrType CPU_records_;

    uint32_t gpu_capacity_;
    uint32_t num_units_; // these are either full records or fields depending on split_field
    std::unique_ptr<ScatterGather<YcsbRecords>>      sg_record_;
    std::unique_ptr<ScatterGather<YcsbFieldRecords>> sg_field_;

    //Used for async flush
    std::unique_ptr<ScatterGather<YcsbRecords>>     sg_record_flush_;
    std::unique_ptr<ScatterGather<YcsbFieldRecords>> sg_field_flush_;
    std::once_flag flush_sg_once_flag_;



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

    // CUB select scratch (flag array + temp storage), allocated up
    // front so no cudaMalloc (a device-wide sync point) fires mid-staging
    uint8_t* d_filter_flags_ = nullptr;  // gpu_capacity_ bytes
    void*    d_filter_cub_tmp_ = nullptr;
    size_t   d_filter_cub_tmp_bytes_ = 0;

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
    void buildNeededOnDevice(const uint32_t* d_all_keys, uint32_t n_keys, uint32_t epoch);

    // SG helpers (dispatch to the active SG based on split_field)
    void sg_transfer_versions(const uint32_t* h_crids, const uint32_t* h_grids, uint32_t n, uint32_t epoch);
    void sg_sync();
    void* sg_stream(); // returns the cuda stream (as void*) of the active SG

    // ----------------------------
    // Pinned (in-flight flush) set for eviction skipping
    // ----------------------------
    // Device flag: d_flush_pinned_flag_[grid] == 1 => grid is pinned (cannot be evicted)
    uint8_t*  d_flush_pinned_flag_ = nullptr;

    // Reclaim-first eviction (delete-bearing mixes): per-slot flag set by
    // mark_reclaimable, preferred by the eviction pre-pass, cleared on
    // rename. nullptr when disabled.
    bool      reclaim_eviction_active_ = false;
    uint8_t*  d_reclaim_flag_ = nullptr;

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
    uint32_t* d_mapped_ring_head_        = nullptr;  // [8..9] (unsigned long long)
    uint32_t* d_mapped_ring_tail_        = nullptr;  // [10..11] (unsigned long long)
    uint32_t* d_mapped_new_insert_count_ = nullptr;  // [12] (skip-admit subset count)

    // Per-epoch timeline for instrumentation
    Timeline tl_;

    // Signal from full SG transfer completion → worker D2H+scatter
    std::atomic<bool> sg_transfer_done_{false};
    std::mutex sg_transfer_mutex_;
    std::condition_variable sg_transfer_cv_;

public:
    void signal_sg_transfer_done() {
        {
            std::lock_guard<std::mutex> lk(sg_transfer_mutex_);
            sg_transfer_done_.store(true, std::memory_order_release);
        }
        sg_transfer_cv_.notify_one();
    }
};

}






#endif // EPIC__YCSB_HYBRID_STAGER_H