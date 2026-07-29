//
// Created by Christian Tabbah on 2025-09-24.
//

#include "tpcc_hybrid_stager.h"
#include <unistd.h>
#include <gpu_txn.cuh>
#include <util_gpu_error_check.cuh>
#include <util_device_ring.cuh>
#include <scatter_gather.h>
#include <cuco/static_map.cuh>

#include <thrust/system/cuda/detail/sort.h>
#include <thrust/system/cuda/detail/unique.h>
#include <thrust/system/cuda/detail/set_operations.h>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_ptr.h>
#include <chrono>
#include <util_log.h>
#include <unordered_set>


#include <variant>
#include <stdexcept>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <cstring>
#include <numa.h>
#include "numa_affinity.h"
#include "reclaim_flags.cuh"

static void pin_current_thread_to_node1_cores()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // Node-1 physical cores via numa_affinity.h.
    for (int cpu : epic::scatter_worker_cpus()) {
        CPU_SET(cpu, &cpuset);
    }

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        throw std::runtime_error(
            std::string("pthread_setaffinity_np failed: ") + std::strerror(rc));
    }

    numa_run_on_node(1);
    numa_set_preferred(1);
}

namespace epic::tpcc {

// Persistent flush-worker thread. One per TpccHybridStager instance. Created
// in the stager constructor, pinned to node 1 ONCE, then loops: wait on CV
// for work -> run it -> signal done. This eliminates the per-epoch
// std::thread() spawn cost (a ~3 ms scheduler delay while queued workers
// wait for the main thread to block on a CUDA sync).
class TpccPersistentFlushWorker {
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_request_;
    std::condition_variable cv_done_;
    enum State { IDLE, BUSY, TERMINATE };
    State state_ = IDLE;
    std::function<void()> work_fn_;
public:
    TpccPersistentFlushWorker() {
        thread_ = std::thread([this]{ this->run(); });
    }
    ~TpccPersistentFlushWorker() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            state_ = TERMINATE;
        }
        cv_request_.notify_one();
        if (thread_.joinable()) thread_.join();
    }
    // Wait for any in-flight work to complete. Idempotent.
    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [this]{ return state_ == IDLE; });
    }
    // Submit work. Caller must ensure no in-flight work (or call wait() first).
    // Defensive: this also waits for any in-flight work before submitting.
    void submit(std::function<void()> fn) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [this]{ return state_ == IDLE; });
        work_fn_ = std::move(fn);
        state_ = BUSY;
        cv_request_.notify_one();
    }
private:
    void run() {
        // Pin once and stay on node 1 for the lifetime of the process.
        try {
            pin_current_thread_to_node1_cores();
        } catch (...) {
            // Non-fatal; worker can still run on inherited affinity.
        }
        while (true) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_request_.wait(lk, [this]{ return state_ != IDLE; });
            if (state_ == TERMINATE) break;
            auto fn = std::move(work_fn_);
            lk.unlock();
            try { if (fn) fn(); } catch (...) {}
            {
                std::lock_guard<std::mutex> lk2(mu_);
                state_ = IDLE;
            }
            cv_done_.notify_one();
        }
    }
};
}  // namespace epic::tpcc



namespace epic::tpcc {
    /* Eviction: FIFO ring-based scan over GPU slots */


#define CUB_ERR_CHECK(call) do {                          \
cudaError_t _e = (call);                              \
if (_e != cudaSuccess) {                              \
throw std::runtime_error(std::string("CUB/CUDA error: ") + cudaGetErrorString(_e)); \
}                                                     \
} while(0)

    // ------- Timer helpers --------
    struct CpuTimer {
        const char* tag;
        std::chrono::high_resolution_clock::time_point t0;
        explicit CpuTimer(const char* t) : tag(t), t0(std::chrono::high_resolution_clock::now()) {}
        ~CpuTimer() {
            using namespace std::chrono;
            auto us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
            Logger::GetInstance().Info("{}: {} us", tag, us);
        }
    };

    struct GpuTimer {
        const char* tag;
        std::chrono::high_resolution_clock::time_point t0;
        explicit GpuTimer(const char* t, cudaStream_t s = 0) : tag(t), t0(std::chrono::high_resolution_clock::now()) {}
        ~GpuTimer() {
            using namespace std::chrono;
            auto us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
            Logger::GetInstance().Info("{}: {} us", tag, us);
        }
    };

    // Each TpccHybridStager owns one record type (Record<X> for a given
    // table); GPU_records_ / GPU_versions_ / CPU_records_ are direct typed
    // pointers (no YCSB-style variant accessors).

    __global__ void init_ring_ids(uint32_t* buf, uint32_t n) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) buf[i] = i;   // GRIDs 0..n-1
    }

    __global__ void k_fill_slots(uint8_t* slots, uint32_t n, uint8_t value) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) slots[i] = value;
    }

    // Initialize the two version tags of a freshly-inserted record's cache slot.
    // An insert reuses an evicted slot and inherits the prior occupant's version
    // tags. The executor's dual-version slot picker (version1 < version2) reads
    // those tags to choose which slot to write, so leftover tags make the chosen
    // slot depend on which record previously occupied it, which is not
    // deterministic across runs. Resetting both tags to 0 makes the picker write
    // slot 2 deterministically for every insert. Placement only: the executor
    // overwrites the value of the slot it writes, and the other slot's leftover
    // value never reaches the Primary Store (writeback ships only the written
    // slot). Touches insert-allocate entries only; miss-admit entries receive
    // both tags from the admission SG transfer and must keep them.
    template <typename TRecT>
    __global__ void k_zero_insert_slot_tags(TRecT* records,
                                            const uint32_t* __restrict__ d_new,
                                            const uint8_t* __restrict__ d_new_is_insert,
                                            const uint32_t* __restrict__ d_crid_to_grid,
                                            uint32_t num_units, uint32_t n) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        if (!d_new_is_insert[i]) return;
        uint32_t crid = d_new[i];
        if (crid >= num_units) return;
        uint32_t grid = d_crid_to_grid[crid];
        if (grid == 0xffffffffu) return;
        records[grid].version1 = 0;
        records[grid].version2 = 0;
    }

    // Simple copy_if kernel: copy src[i] to out where stencil[i] == sentinel
    __global__ void k_copy_if_stencil_eq(
        const uint32_t* __restrict__ src,
        const uint32_t* __restrict__ stencil,
        uint32_t n,
        uint32_t sentinel,
        uint32_t* __restrict__ out,
        uint32_t* __restrict__ d_out_count)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        // d_needed_ (passed as `src`) is built from cub::Unique over the per-table
        // op-key array, which preserves the sentinel-padded entries that the
        // submitter writes for incomplete txns. The matching lookup in
        // d_needed_grids (passed as `stencil`) returns sentinel for those CRIDs,
        // so without the `src[i] != sentinel` filter the sentinel CRID falls
        // into d_new_, propagates through admission, and the host-side gather
        // computes `CPU_recs[sentinel]` ~ 4 GiB past the primary store. SIGSEGV.
        if (stencil[i] == sentinel && src[i] != sentinel) {
            uint32_t pos = atomicAdd(d_out_count, 1u);
            out[pos] = src[i];
        }
    }

    __global__ void k_set_grid2crid(const uint32_t* __restrict__ grids,
                                    const uint32_t* __restrict__ crids,
                                    uint32_t n,
                                    uint32_t* __restrict__ grid2crid) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) grid2crid[grids[i]] = crids[i];
    }

    // The reclaim-first eviction kernels (flag-preferring collector,
    // clear-on-rename, mark-by-CRIDs) live in reclaim_flags.cuh, shared
    // with the YCSB stager.

    // Collect eviction candidates into out arrays
    __global__ void k_fifo_collect_evictions(uint32_t start,
                                            uint32_t cap,
                                            const uint32_t* __restrict__ resident_list,
                                            const uint8_t* __restrict__ needed_flag,
                                            const uint8_t* __restrict__ flush_pinned_flag,
                                            uint32_t deficit,
                                            uint32_t* __restrict__ out_grids,
                                            uint32_t* __restrict__ out_crids,
                                            uint32_t* __restrict__ out_count,
                                            uint32_t* __restrict__ out_max_offset)
    {
        // scan offsets i = tid, tid+stride, ...
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t stride = blockDim.x * gridDim.x;
        if (atomicAdd(out_count, 0u) >= deficit) return;
        for (uint32_t i = tid; i < cap; i += stride) {
            if (atomicAdd(out_count, 0u) >= deficit) break;

            uint32_t g = start + i;
            if (g >= cap) g -= cap;

            uint32_t crid = resident_list[g];
            if (crid == 0xffffffffu) continue;     // empty
            if (needed_flag[g] || flush_pinned_flag[g]) continue;          // needed -> skip

            uint32_t pos = atomicAdd(out_count, 1u);
            if (pos < deficit) {
                out_grids[pos] = g;
                out_crids[pos] = crid;
                atomicMax(out_max_offset, i);
            }
            // once we have enough, we still might have threads running,
            // but pos<deficit prevents overflow.
        }
    }


    // TPC-C never enables split_field, so the YCSB k_promote_to_field_crid
    // kernel (which rewrites per-op record_ids into per-field CRIDs) has no
    // TPC-C counterpart.
    //
    // Note for the per-txn-type Remap kernels (tpcc_hybrid_remap.cu): do
    // NOT introduce a "skip INSERT/FULL_READ/FULL_READ_MODIFY_WRITE"
    // branch when rewriting record_id fields. The skip in the YCSB
    // k_flat_remap_txn_record_ids is load-bearing only for the split-field
    // expansion that comes after it there; see the invariant comment in
    // tpcc_hybrid_remap.cu.

    // Gather CRIDs by GRIDs from the resident list
    __global__ void k_gather_crids_by_grids(const uint32_t* __restrict__ grids,
                                            uint32_t n,
                                            const uint32_t* __restrict__ grid2crid, // d_resident_list_
                                            uint32_t* __restrict__ out_crids) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out_crids[i] = grid2crid[grids[i]];
    }

    __global__ void k_mark_flag_by_grids(const uint32_t* __restrict__ grids,
                                     uint32_t n,
                                     uint32_t cap,
                                     uint8_t* __restrict__ flag)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        uint32_t g = grids[i];
        if (g != 0xffffffffu && g < cap) {
            flag[g] = 1;
        }
    }

    // Set bitmap[crid] = 1 for every non-sentinel entry in d_insert_keys.
    // d_insert_keys is sentinel-padded parallel to d_all_keys (length n_all);
    // duplicate insert keys can't happen — the indexer mints a fresh CRID per
    // INSERT — but the kernel is duplicate-safe (idempotent stores).
    __global__ void k_set_insert_bitmap(const uint32_t* __restrict__ insert_keys,
                                        uint32_t n,
                                        uint32_t num_units,
                                        uint8_t* __restrict__ bitmap)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        uint32_t key = insert_keys[i];
        if (key != 0xffffffffu && key < num_units) {
            bitmap[key] = 1;
        }
    }

    // out[i] = (in[i] == 0) ? 1 : 0   — used to feed cub::DeviceSelect::Flagged
    // when we want to keep the entries where is_insert is 0 (the miss-admit
    // subset that needs a version SG transfer). Single dependent op per thread.
    __global__ void k_invert_flag(const uint8_t* __restrict__ in,
                                  uint8_t* __restrict__ out,
                                  uint32_t n)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = (in[i] == 0) ? 1 : 0;
    }

    // For each CRID in d_new_, look up bitmap[crid] and write the per-slot
    // is_insert flag into d_new_is_insert_. Also bumps d_insert_count by 1
    // for every flag set, giving the host a prompt count of insert-allocate
    // slots to log alongside the existing slots-admitted accounting.
    __global__ void k_lookup_insert_flag_for_new(const uint32_t* __restrict__ d_new,
                                                  uint32_t n,
                                                  const uint8_t* __restrict__ bitmap,
                                                  uint8_t* __restrict__ d_new_is_insert,
                                                  uint32_t* __restrict__ d_insert_count,
                                                  uint32_t num_units)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        uint32_t crid = d_new[i];
        // bitmap is sized num_units; sentinel-padded slots in d_all_keys can
        // propagate through dedup into d_new_, and d_new_ can hold CRIDs that
        // were valid at lookup time but exceed the bitmap's static size. The
        // partner kernel k_set_insert_bitmap already guards both cases; this
        // one must too. Unguarded reads here OOB the GPU heap and surface as
        // illegal-memory-access at the next stream sync.
        bool valid = (crid != 0xffffffffu) && (crid < num_units);
        uint8_t is_insert = valid ? bitmap[crid] : 0;
        d_new_is_insert[i] = is_insert;
        if (is_insert) atomicAdd(d_insert_count, 1u);
    }

    // Copy a device scalar to a mapped host pointer via a 1-thread kernel.
    // Bypasses the D2H copy engine — uses SM store instruction over PCIe.
    __global__ void k_copy_to_mapped(const uint32_t* __restrict__ src, uint32_t* dst) {
        if (threadIdx.x == 0) *dst = *src;
    }

    // Copy unsigned long long to mapped memory (2 x uint32_t slots)
    __global__ void k_copy_ull_to_mapped(const unsigned long long* __restrict__ src, uint32_t* dst) {
        if (threadIdx.x == 0) {
            unsigned long long val = *src;
            dst[0] = static_cast<uint32_t>(val);
            dst[1] = static_cast<uint32_t>(val >> 32);
        }
    }

    template <typename TRec, typename TVer>
    TpccHybridStager<TRec, TVer>::TpccHybridStager(Allocator& alloc,
                               TRec* GPU_records, TVer* GPU_versions,
                               TRec* CPU_records,
                               uint32_t gpu_capacity, uint32_t num_units,
                               bool enable_reclaim_eviction)
        : alloc_(alloc),
          GPU_records_(GPU_records),
          GPU_versions_(GPU_versions),
          CPU_records_(CPU_records),
          gpu_capacity_(gpu_capacity),
          num_units_(num_units),
          crid2grid_index_(alloc_, gpu_capacity_, num_units_) // capacity=gpu slots, num_units=total CRIDs
    {
        auto &logger = Logger::GetInstance();
        // allocate device memory for the various arrays
        logger.Info("Initializing TpccHybridStager with GPU capacity: " + std::to_string(gpu_capacity_));
        reclaim_eviction_active_ = enable_reclaim_eviction;
        if (reclaim_eviction_active_) {
            logger.Info("  reclaim-first eviction ENABLED for this stager");
        }

        // Each stager owns its own CUDA stream so cross-stager kernels do not
        // implicitly serialize via the default stream when the 8 stagers
        // prepare in parallel (8-way OMP parallel sections in tpcc.cpp).
        //
        // The stream is REGULAR (not cudaStreamNonBlocking) so it implicitly
        // synchronizes with the legacy default stream.  This is load-bearing:
        // bulk_insert (writes d_crid_to_grid) launches on this stream from an
        // OMP worker thread; remapTpccTxnsKernel later reads d_crid_to_grid on
        // the default stream from the master thread.  With a non-blocking
        // stream, host-side cudaStreamSynchronize from the worker does not
        // establish happens-before with the master's subsequent default-stream
        // launch, and remap can read SENTINEL for valid CRIDs (the executor
        // then reads out of bounds in the dirty/record arrays at
        // gpuExecKernel).
        gpu_err_check(cudaStreamCreate(&prep_stream_));
        logger.Info("prepareEpoch on per-stager stream");

        // Spin up the persistent flush worker (one per stager). It pins itself
        // to node 1 in its constructor body and then sleeps on a CV until
        // start_flush_epoch_async submits work to it.
        persistent_worker_ = std::make_unique<TpccPersistentFlushWorker>();

        // YCSB had a runtime check here that the variants matched split_field.
        // TPC-C has typed pointers; the type system enforces consistency.

        auto bytes_u32 = [](size_t n) { return n * sizeof(uint32_t); };

        d_needed_ = static_cast<uint32_t*>(alloc_.Allocate(bytes_u32(gpu_capacity_)));
        d_new_    = static_cast<uint32_t*>(alloc_.Allocate(bytes_u32(gpu_capacity_)));
        d_resident_list_ = static_cast<uint32_t*>(alloc_.Allocate(bytes_u32(gpu_capacity_)));

        d_needed_count_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_new_count_    = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        // malloc space for first_needed_count_ instead of using the allocator since this is on host
        first_needed_count_ = static_cast<uint32_t*>(malloc(sizeof(uint32_t)));


        gpu_err_check(cudaMemset(d_needed_count_, 0, sizeof(uint32_t)));
        gpu_err_check(cudaMemset(d_new_count_,    0, sizeof(uint32_t)));
        *first_needed_count_ = gpu_capacity_;
        // mark all empty
        gpu_err_check(cudaMemset(d_resident_list_, 0xff, bytes_u32(gpu_capacity_)));

        // allocate the device ring
        d_ring_buffer_ = static_cast<uint32_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint32_t)));
        d_ring_head_   = static_cast<unsigned long long*>(alloc_.Allocate(sizeof(unsigned long long)));
        d_ring_tail_   = static_cast<unsigned long long*>(alloc_.Allocate(sizeof(unsigned long long)));

        // reserve host vectors for dirty tracking
        h_dirty_grids_.reserve(gpu_capacity_);
        h_dirty_crids_.reserve(gpu_capacity_);

        //Initialize the buffer with GRIDs 0-gpu_capacity_-1

        {
            const int block = 256;
            const int grid = (gpu_capacity_ + block - 1) / block;
            init_ring_ids<<<grid, block>>>(d_ring_buffer_, gpu_capacity_);
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        }

        // Initialize head/tail counters
        // we start will full(all slots free): head = 0, tail = gpu_capacity_ (so tail - head = capacity)
        {
            const unsigned long long zero = 0;
            const unsigned long long cap  = static_cast<unsigned long long>(gpu_capacity_);
            gpu_err_check(cudaMemcpy(d_ring_head_, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
            gpu_err_check(cudaMemcpy(d_ring_tail_, &cap,  sizeof(unsigned long long), cudaMemcpyHostToDevice));
        }

        // 8-byte scratch used by launch_ring_pop_parallel to pass the reserved
        // head index from Phase 1 (reservation) to Phase 2 (parallel copy).
        gpu_err_check(cudaMalloc(&d_ring_pop_scratch_, sizeof(unsigned long long)));

        // prepare the device ring struct
        free_list_ring_.buffer = d_ring_buffer_;
        free_list_ring_.head   = d_ring_head_;
        free_list_ring_.tail   = d_ring_tail_;
        free_list_ring_.capacity = gpu_capacity_;
        free_list_ring_.pop_scratch_start = d_ring_pop_scratch_;

        // init the scatter-gather (single instance per stager — TPC-C has no
        // field-split branch).
        sg_record_ = std::make_unique<ScatterGather<TRec>>();
        {
            typename ScatterGather<TRec>::Options opt{};
            opt.own_stream = true;
            opt.chunk_mode = true;
            opt.async_writeback = false;
            sg_record_->init(opt);
        }
        sg_record_flush_ = std::make_unique<ScatterGather<TRec>>();

        // initialize FIFO data structures
        d_needed_grid_flag_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));
        gpu_err_check(cudaMemset(d_needed_grid_flag_, 0, gpu_capacity_ * sizeof(uint8_t)));

        // periodicFlush scratch (cached) - allocate once
        d_dirty_grids_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_dirty_crids_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        //second scratch for grids for the second slot, will combine into d_dirty_grids_
        d_dirty_grids_slot1_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_dirty_crids_slot1_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));

        // Skip-admit scratch.
        d_insert_bitmap_   = static_cast<uint8_t* >(alloc_.Allocate(num_units_   * sizeof(uint8_t)));
        d_new_is_insert_   = static_cast<uint8_t* >(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));
        d_new_insert_count_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        gpu_err_check(cudaMemset(d_insert_bitmap_,    0, num_units_   * sizeof(uint8_t)));
        gpu_err_check(cudaMemset(d_new_is_insert_,    0, gpu_capacity_ * sizeof(uint8_t)));
        gpu_err_check(cudaMemset(d_new_insert_count_, 0, sizeof(uint32_t)));

        d_flush_pinned_flag_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));
        gpu_err_check(cudaMemset(d_flush_pinned_flag_, 0, gpu_capacity_ * sizeof(uint8_t)));

        // Pre-allocated scratch buffers for prepareEpoch (avoid cudaMalloc/cudaFree per epoch)
        d_scratch_a_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_scratch_b_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_scratch_c_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_scratch_d_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_scratch_f_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * gpu_capacity_));
        d_filter_flags_ = static_cast<uint8_t*>(alloc_.Allocate(sizeof(uint8_t) * gpu_capacity_));

        // Reclaim-first eviction: allocate the per-slot flag, initialized
        // to 0 (= live). The Delivery executor and mark_reclaimable set
        // it; the eviction pre-pass prefers flagged slots.
        if (reclaim_eviction_active_) {
            d_reclaim_flag_ = static_cast<uint8_t*>(alloc_.Allocate(sizeof(uint8_t) * gpu_capacity_));
            gpu_err_check(cudaMemset(d_reclaim_flag_, 0, sizeof(uint8_t) * gpu_capacity_));
        }

        // Pre-allocate the CUB temp buffer used by the eviction-path stream
        // compaction at max capacity, avoiding cudaMalloc (a global device
        // barrier) during staging
        {
            size_t tmp_bytes = 0;
            cub::DeviceSelect::Flagged(nullptr, tmp_bytes,
                                       d_scratch_b_, d_filter_flags_, d_scratch_d_, d_scalar_b_,
                                       (int)gpu_capacity_, 0);
            gpu_err_check(cudaMalloc(&d_filter_cub_tmp_, tmp_bytes));
            d_filter_cub_tmp_bytes_ = tmp_bytes;
        }

        // GPU-side eviction-path stream compaction. Only the count scalar is
        // pre-allocated (4 bytes). The crids/grids output buffers are lazy-grown
        // in the eviction block based on the actual observed deficit (which is
        // bounded by gpu_capacity_ but in practice << it). The keep-flag buffer
        // and CUB temp storage are the d_filter_flags_/d_filter_cub_tmp_
        // pair allocated above (both sized to gpu_capacity_).
        d_ev_miss_count_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_scalar_a_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_scalar_b_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_scalar_c_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));

        d_clear_slots0_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));
        d_clear_slots1_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));

        // Mapped memory for scalar readbacks (bypasses D2H copy engine)
        gpu_err_check(cudaHostAlloc(&h_stager_mapped_, kMappedScalarCount_ * sizeof(uint32_t), cudaHostAllocMapped));
        memset(h_stager_mapped_, 0, kMappedScalarCount_ * sizeof(uint32_t));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_needed_count_,      &h_stager_mapped_[0], 0));
        // Slots [1] (d_mapped_load_needed_count_), [2] (d_mapped_write_only_count_),
        // [7] (d_mapped_write_unique_) were the readbacks for the unwired write_only
        // optimization and the staging-time mark_dirty pipeline (both removed). Slots
        // intentionally unbound; keep kMappedScalarCount_ at 13 so the offsets of
        // [3..12] stay stable.
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_new_count_,         &h_stager_mapped_[3], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_a_,          &h_stager_mapped_[4], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_b_,          &h_stager_mapped_[5], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_c_,          &h_stager_mapped_[6], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_ring_head_,         &h_stager_mapped_[8], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_ring_tail_,         &h_stager_mapped_[10], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_new_insert_count_,  &h_stager_mapped_[12], 0));

        logger.Info("TpccHybridStager initialization complete.");

        // Warmup launches to eliminate first-launch overhead (JIT/driver
        // setup adds ~4ms to a kernel's first invocation).
        {
            gpu_err_check(cudaMemsetAsync(d_scalar_b_, 0, sizeof(uint32_t), 0));
            gpu_err_check(cudaMemsetAsync(d_scalar_c_, 0, sizeof(uint32_t), 0));
            // Warmup k_fifo_collect_evictions
            k_fifo_collect_evictions<<<1, 1>>>(0, gpu_capacity_,
                                                d_resident_list_, d_needed_grid_flag_,
                                                d_flush_pinned_flag_,
                                                0,  // deficit=0 → kernel exits immediately
                                                d_scratch_c_, d_scratch_b_,
                                                d_scalar_b_, d_scalar_c_);
            // Warmup bulk_update_crid (first called at epoch ~100, eviction transition).
            // k_flat_rename_crids unconditionally writes d_crid_to_grid[oldc]=SENT
            // then d_crid_to_grid[newc]=grid; with zeroed scratch, both crids are
            // 0 and grid is 0, so d_crid_to_grid[0] is left at 0 instead of the
            // sentinel set by CridGridIndex's ctor. Restore it here. Without this,
            // tables whose CRID-0 is in active use (Warehouse at W=1) silently
            // pretend the CRID is cached without a matching d_resident_list_ entry,
            // and periodicFlush eventually feeds a sentinel CRID into SG.
            crid2grid_index_.bulk_update_crid(d_scratch_b_, d_scratch_c_, d_scratch_a_, 1);
            gpu_err_check(cudaStreamSynchronize(0));
            gpu_err_check(cudaMemset(const_cast<uint32_t*>(crid2grid_index_.device_view().d_crid_to_grid),
                                     0xff, num_units_ * sizeof(uint32_t)));
            logger.Info("Warmup launches complete.");
        }

    }

        template <typename TRec, typename TVer>
        void TpccHybridStager<TRec, TVer>::buildNeededOnDevice(const uint32_t* d_all_keys, uint32_t n_keys, uint32_t epoch, cudaStream_t stream)
    {
        if (n_keys == 0) {
            // Use cudaMemsetAsync on the per-stager stream; synchronous
            // cudaMemcpy here would be a global cross-stream sync point and
            // serialize the 8 concurrently-preparing stagers.
            gpu_err_check(cudaMemsetAsync(d_needed_count_, 0, sizeof(uint32_t), stream));
            return;
        }

        GpuTimer tg("build_needed(gpu)");

        // ---- ensure cached key buffers large enough ----
        if (n_keys > keys_cap_) {
            if (d_keys_in_)  alloc_.Free(d_keys_in_);
            if (d_keys_out_) alloc_.Free(d_keys_out_);

            d_keys_in_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * n_keys));
            d_keys_out_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * n_keys));
            keys_cap_   = n_keys;

            // reset cached temp sizes so we re-query below
            if (d_sort_tmp_)   { cudaFree(d_sort_tmp_);   d_sort_tmp_ = nullptr;   sort_tmp_bytes_ = 0; }
            if (d_unique_tmp_) { cudaFree(d_unique_tmp_); d_unique_tmp_ = nullptr; unique_tmp_bytes_ = 0; }
        }

        // ---- copy keys to cached input ----
        {
            GpuTimer tc("build_needed: memcpy keys");
            gpu_err_check(cudaMemcpyAsync(d_keys_in_, d_all_keys, sizeof(uint32_t) * n_keys, cudaMemcpyDeviceToDevice, stream));
        }

        // ---- sort (CUB, cached temp) ----
        {
            GpuTimer ts("build_needed: sort");

            // CUB temp_bytes scales with n_keys; query per call, grow cache if needed
            size_t sort_bytes = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes,
                                           d_keys_in_, d_keys_out_,
                                           n_keys,
                                           /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t)*8, stream);
            if (sort_bytes > sort_tmp_bytes_) {
                if (d_sort_tmp_) cudaFree(d_sort_tmp_);
                gpu_err_check(cudaMalloc(&d_sort_tmp_, sort_bytes));
                sort_tmp_bytes_ = sort_bytes;
            }

            cub::DeviceRadixSort::SortKeys(d_sort_tmp_, sort_bytes,
                                           d_keys_in_, d_keys_out_,
                                           n_keys,
                                           /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t)*8, stream);
        gpu_err_check(cudaStreamSynchronize(stream));
        }

        // ---- unique (CUB, cached temp) ----
        // unique output goes back into d_keys_in_ (ping-pong reuse)
        {
            GpuTimer tu("build_needed: unique");

            size_t unique_bytes = 0;
            cub::DeviceSelect::Unique(nullptr, unique_bytes,
                                      d_keys_out_, d_keys_in_,
                                      d_needed_count_,  // device scalar for count
                                      n_keys, stream);
            if (unique_bytes > unique_tmp_bytes_) {
                if (d_unique_tmp_) cudaFree(d_unique_tmp_);
                gpu_err_check(cudaMalloc(&d_unique_tmp_, unique_bytes));
                unique_tmp_bytes_ = unique_bytes;
            }

            cub::DeviceSelect::Unique(d_unique_tmp_, unique_bytes,
                                      d_keys_out_, d_keys_in_,
                                      d_needed_count_,
                                      n_keys, stream);
        }

        // ---- pull unique_count to host for capacity check ----
        uint32_t unique_count = 0;
        k_copy_to_mapped<<<1, 1, 0, stream>>>(d_needed_count_, d_mapped_needed_count_);
        gpu_err_check(cudaStreamSynchronize(stream));
        unique_count = h_stager_mapped_[0];

        if (unique_count > gpu_capacity_) {
            throw std::runtime_error("TpccHybridStager<TRec, TVer>::buildNeededOnDevice: unique needed keys exceed GPU capacity.");
        }

        // ---- epoch==1 resize d_needed_ ----
        if (epoch == 1) {
            alloc_.Free(d_needed_);
            *first_needed_count_ = unique_count * 2;
            d_needed_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * (*first_needed_count_)));
        }

        // ---- store canonical needed list from d_keys_in_ ----
        // cudaMemcpyAsync on per-stager stream (avoid global sync from synchronous Memcpy).
        gpu_err_check(cudaMemcpyAsync(d_needed_, d_keys_in_,
                                 unique_count * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice, stream));

        // d_needed_count_ is already set on device by CUB Unique
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::sg_transfer_versions(const uint32_t* h_crids, const uint32_t* h_grids, uint32_t n, uint32_t epoch)
    {
        if (n == 0) {
            return;
        }
            sg_record_->transfer_versions(h_crids, h_grids, epoch, n,
                CPU_records_,
                GPU_records_);


    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::sg_sync()
    {
        sg_record_->sync();
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::prewarm_initial_undelivered_orderline(uint32_t num_warehouses)
    {
        auto& logger = Logger::GetInstance();

        // Per the TPC-C spec (Clause 4.3.3.1): each district initially holds
        // 3000 orders; orders with O_ID in [2101, 3000] are undelivered. With
        // 15 OL slots per order (worst-case ol_cnt), that is 900 × 15 = 13500
        // OL CRIDs per district that Delivery will eventually want. The CPU
        // primary loader assigns OL CRIDs in (w, d, o, ol) row-major order,
        // so for each (w, d) pair the undelivered range is contiguous:
        //   start = ((w-1)*10 + (d-1)) * 45000 + 2100*15
        //   count = 13500
        constexpr uint32_t districts_per_w   = 10;
        constexpr uint32_t orders_per_d      = 3000;
        constexpr uint32_t ol_per_o          = 15;
        constexpr uint32_t per_district_size = orders_per_d * ol_per_o;       // 45000
        constexpr uint32_t undelivered_first_o_idx = 2100;                    // O_ID 2101 -> idx 2100
        constexpr uint32_t undelivered_n_orders    = 900;
        constexpr uint32_t per_district_ol         = undelivered_n_orders * ol_per_o; // 13500

        const uint64_t total_ol = uint64_t(num_warehouses) * districts_per_w * per_district_ol;
        if (total_ol >= gpu_capacity_) {
            logger.Warn("OL pre-warm: requested {} CRIDs >= gpu_capacity_ {}; skipping",
                        total_ol, gpu_capacity_);
            return;
        }
        const uint32_t n = static_cast<uint32_t>(total_ol);
        logger.Info("OL pre-warm: warming {} CRIDs ({} MB record data)",
                    n, (uint64_t(n) * sizeof(TRec)) / (1024 * 1024));

        // 1. Build the host CRID list for the undelivered backlog.
        std::vector<uint32_t> h_crids;
        h_crids.reserve(n);
        for (uint32_t w = 0; w < num_warehouses; ++w) {
            for (uint32_t d = 0; d < districts_per_w; ++d) {
                const uint32_t base = (w * districts_per_w + d) * per_district_size
                                    + undelivered_first_o_idx * ol_per_o;
                for (uint32_t i = 0; i < per_district_ol; ++i) {
                    h_crids.push_back(base + i);
                }
            }
        }

        // 2. Allocate device CRID list and GRID output, copy CRIDs to device.
        uint32_t* d_pw_crids = nullptr;
        uint32_t* d_pw_grids = nullptr;
        uint32_t* d_pw_granted = nullptr;
        gpu_err_check(cudaMalloc(&d_pw_crids,   sizeof(uint32_t) * n));
        gpu_err_check(cudaMalloc(&d_pw_grids,   sizeof(uint32_t) * n));
        gpu_err_check(cudaMalloc(&d_pw_granted, sizeof(uint32_t)));
        gpu_err_check(cudaMemcpy(d_pw_crids, h_crids.data(),
                                 sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

        // 3. Pop n GRIDs from the free list ring.
        launch_ring_pop_parallel(free_list_ring_, n, d_pw_grids, d_pw_granted, 0);
        gpu_err_check(cudaDeviceSynchronize());
        uint32_t h_granted = 0;
        gpu_err_check(cudaMemcpy(&h_granted, d_pw_granted,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (h_granted != n) {
            logger.Error("OL pre-warm: ring_pop granted {} < {} requested; aborting",
                         h_granted, n);
            cudaFree(d_pw_crids);
            cudaFree(d_pw_grids);
            cudaFree(d_pw_granted);
            return;
        }

        // 4. Install (CRID, GRID) mappings in the flat index.
        crid2grid_index_.bulk_insert(d_pw_crids, d_pw_grids, n, 0);
        gpu_err_check(cudaDeviceSynchronize());

        // 5. Update resident_list[grid] = crid for each pre-warmed slot.
        {
            constexpr int B = 256;
            int G = (n + B - 1) / B;
            k_set_grid2crid<<<G, B>>>(d_pw_grids, d_pw_crids, n, d_resident_list_);
            gpu_err_check(cudaPeekAtLastError());
            gpu_err_check(cudaDeviceSynchronize());
        }

        // 6. Pull GRIDs back to host so we can pass plain host arrays into
        //    sg_transfer_versions (which expects host-side arrays per the
        //    existing prepareEpoch admit path).
        std::vector<uint32_t> h_grids(n);
        gpu_err_check(cudaMemcpy(h_grids.data(), d_pw_grids,
                                 sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));

        // 7. CPU-gather + H2D + GPU-scatter the records. Epoch 0 is the
        //    placeholder pre-epoch-1 marker.
        sg_transfer_versions(h_crids.data(), h_grids.data(), n, /*epoch=*/0);
        sg_sync();

        // 8. Pre-warmed slots start live (reclaim_flag = 0). The
        //    buffer was zeroed in the constructor and we have not touched
        //    it here, so this is already correct -- no extra work needed.

        cudaFree(d_pw_crids);
        cudaFree(d_pw_grids);
        cudaFree(d_pw_granted);

        logger.Info("OL pre-warm complete: {} CRIDs admitted to GPU cache", n);
    }

    template <typename TRec, typename TVer>
    void* TpccHybridStager<TRec, TVer>::sg_stream()
    {
        return sg_record_->get_stream();
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::prepareEpoch(uint32_t epoch,
        const uint32_t* d_all_keys, uint32_t n_all,
        const uint32_t* d_read_keys, uint32_t n_read,
        const uint32_t* d_write_keys, uint32_t n_write,
        const uint32_t* d_insert_keys, uint32_t n_insert,
        FlushHandle* flush_handle)
    {
        CpuTimer tc("prepareEpoch(cpu)");
        // Per-stager CUDA stream (see the constructor comment for why it is
        // a regular stream).
        const cudaStream_t s = active_stream();
        tl_.reset();
        tl_.mark("start");

        // Reset the per-CRID insert bitmap and re-populate from d_insert_keys.
        // Each non-sentinel entry in d_insert_keys is the CRID of an INSERT
        // op, so bitmap[crid] = 1 marks the CRID as a fresh-insert target this
        // epoch. We use this bitmap below to split d_new_ into the miss-admit
        // subset (regular SG transfer from CPU primary store) and the
        // insert-allocate subset (skip the SG transfer; the executor will
        // produce the cache slot's data via its INSERT write).
        if (d_insert_keys && n_all > 0) {
            GpuTimer tg("prepareEpoch:set_insert_bitmap(gpu)");
            gpu_err_check(cudaMemsetAsync(d_insert_bitmap_, 0, num_units_ * sizeof(uint8_t), s));
            if (n_insert > 0) {
                constexpr int B = 256;
                int G = (n_all + B - 1) / B;
                k_set_insert_bitmap<<<G, B, 0, s>>>(d_insert_keys, n_all, num_units_, d_insert_bitmap_);
                gpu_err_check(cudaPeekAtLastError());
            }
            gpu_err_check(cudaStreamSynchronize(s));
        }

        {
            tl_.mark("build_needed_start");
            GpuTimer tg("build_needed(gpu)");
            buildNeededOnDevice(d_all_keys, n_all, epoch, s);
        gpu_err_check(cudaStreamSynchronize(s));
             // get h_needed_count

        }
        uint32_t h_needed_count = 0;
        // Copy via kernel to mapped memory (bypasses D2H copy engine)
        k_copy_to_mapped<<<1, 1, 0, s>>>(d_needed_count_, d_mapped_needed_count_);
        gpu_err_check(cudaStreamSynchronize(s));
        h_needed_count = h_stager_mapped_[0];
        Logger::GetInstance().Info("Needed count for epoch " + std::to_string(epoch) + " is " + std::to_string(h_needed_count));

        uint32_t h_new_count = 0;
        {
        // 2. Clear needed grid flags and mark needed grids(only needed for FIFO and histogram)
        gpu_err_check(cudaMemsetAsync(d_needed_grid_flag_, 0, gpu_capacity_ * sizeof(uint8_t), s));
        // lookup needed CRIDs --> GRIDs
            uint32_t* d_needed_grids = d_scratch_a_;
        {
            tl_.mark("fifo_mark_start");
            GpuTimer tg("fifo:lookup_and_mark_needed_grids(gpu)");
            // combine the above two steps into one kernel to save some GPU round-trips and also allow touch-based optimizations in the index
            crid2grid_index_.bulk_lookup_crid_to_grid_and_mark_needed(
                d_needed_, d_needed_grids, h_needed_count, d_needed_grid_flag_, gpu_capacity_, s);
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
        }

        {
            GpuTimer tg("index:touch_needed_grids(gpu)");
            crid2grid_index_.touch_by_grids(d_needed_grids, h_needed_count, epoch, s);
        }
        // Then, we mark needed grids in the flag array


        // build d_new_ as "needed CRIDs whose grid lookup missed"
        {
            tl_.mark("fifo_select_start");
            GpuTimer tg("fifo:select_new_crids(gpu)");

            auto cr_begin   = thrust::device_pointer_cast(d_needed_);
            auto cr_end     = cr_begin + h_needed_count;
            auto grid_begin = thrust::device_pointer_cast(d_needed_grids);
            auto new_begin  = thrust::device_pointer_cast(d_new_);

            // Use custom kernel instead of thrust (avoids hidden cudaDeviceSynchronize)
            gpu_err_check(cudaMemsetAsync(d_new_count_, 0, sizeof(uint32_t), s));
            if (h_needed_count > 0) {
                constexpr int B = 256;
                int G = (h_needed_count + B - 1) / B;
                k_copy_if_stencil_eq<<<G, B, 0, s>>>(
                    d_needed_, d_needed_grids, h_needed_count, 0xffffffffu,
                    d_new_, d_new_count_);
                gpu_err_check(cudaPeekAtLastError());
            }
            gpu_err_check(cudaStreamSynchronize(s));
            k_copy_to_mapped<<<1, 1, 0, s>>>(d_new_count_, d_mapped_new_count_);
            gpu_err_check(cudaStreamSynchronize(s));
            h_new_count = h_stager_mapped_[3];
            gpu_err_check(cudaMemcpyAsync(d_new_count_, &h_new_count, sizeof(uint32_t), cudaMemcpyHostToDevice, s));
        }
        }

        // Phase 2b: split d_new_ into miss-admit and insert-allocate subsets
        // by looking up each new CRID in the per-CRID insert bitmap built at
        // the top of this function. d_new_is_insert_[i] = 1 iff d_new_[i]'s
        // op was an INSERT this epoch. Phase 2c will use that flag to skip
        // the SG H2D for the insert subset.
        uint32_t h_new_insert_count = 0;
        if (h_new_count > 0 && d_insert_keys && n_insert > 0) {
            GpuTimer tg("prepareEpoch:lookup_new_is_insert(gpu)");
            gpu_err_check(cudaMemsetAsync(d_new_insert_count_, 0, sizeof(uint32_t), s));
            constexpr int B = 256;
            int G = (h_new_count + B - 1) / B;
            k_lookup_insert_flag_for_new<<<G, B, 0, s>>>(
                d_new_, h_new_count, d_insert_bitmap_, d_new_is_insert_, d_new_insert_count_, num_units_);
            gpu_err_check(cudaPeekAtLastError());
            k_copy_to_mapped<<<1, 1, 0, s>>>(d_new_insert_count_, d_mapped_new_insert_count_);
            gpu_err_check(cudaStreamSynchronize(s));
            h_new_insert_count = h_stager_mapped_[12];
        } else if (h_new_count > 0) {
            // No insert keys this epoch — clear the flag array so Phase 2c's
            // miss-admit path sees zeros for every new entry.
            gpu_err_check(cudaMemsetAsync(d_new_is_insert_, 0, h_new_count * sizeof(uint8_t), s));
            gpu_err_check(cudaStreamSynchronize(s));
        }
        const uint32_t h_new_miss_count =
            (h_new_count >= h_new_insert_count) ? (h_new_count - h_new_insert_count) : 0;
        if (h_new_count > 0) {
            Logger::GetInstance().Info("new={} split: miss-admit={} insert-allocate={}",
                                       h_new_count, h_new_miss_count, h_new_insert_count);
        }
        unsigned long long h_head = 0, h_tail = 0;
        k_copy_ull_to_mapped<<<1, 1, 0, s>>>(d_ring_head_, d_mapped_ring_head_);
        k_copy_ull_to_mapped<<<1, 1, 0, s>>>(d_ring_tail_, d_mapped_ring_tail_);
        gpu_err_check(cudaStreamSynchronize(s));
        h_head = static_cast<unsigned long long>(h_stager_mapped_[8]) | (static_cast<unsigned long long>(h_stager_mapped_[9]) << 32);
        h_tail = static_cast<unsigned long long>(h_stager_mapped_[10]) | (static_cast<unsigned long long>(h_stager_mapped_[11]) << 32);
        uint32_t h_free_slots = static_cast<uint32_t>(h_tail - h_head);

        if (h_new_count > 0)
        {
            // Cache-slot accounting. Every CRID in d_new_ gets bound to a GRID
            // here — whether the GRID came from the free-list ring (no eviction
            // case below) or by evicting a prior CRID (eviction case). The
            // eviction case adds the deficit to cache_slots_evicted_total_.
            // We also split the admitted count into miss-admit (regular SG)
            // and insert-allocate (skip-SG) using the partition computed above.
            cache_slots_admitted_total_ += h_new_count;
            cache_slots_miss_admit_total_      += h_new_miss_count;
            cache_slots_insert_allocate_total_ += h_new_insert_count;
            if (h_new_count > h_free_slots) {
                cache_slots_evicted_total_ += (h_new_count - h_free_slots);
            }

            if (h_new_count <= h_free_slots)
            {
                // remove h_new_count slots from the ring (ring-pop)
                uint32_t* d_grids = d_scratch_a_;
                uint32_t* d_granted = d_scalar_a_;
                tl_.mark("admit_ringpop_start");
                {
                    GpuTimer tg("admit_only:ring_pop(gpu)");
                    auto s = reinterpret_cast<cudaStream_t>(sg_stream());
                    launch_ring_pop_parallel(free_list_ring_, h_new_count, d_grids, d_granted, s);
                    gpu_err_check(cudaPeekAtLastError());
                    gpu_err_check(cudaStreamSynchronize(s)); // sync SG stream, not stream 0
                }

                uint32_t h_granted = 0;
                k_copy_to_mapped<<<1, 1, 0, s>>>(d_granted, d_mapped_scalar_a_);
                gpu_err_check(cudaStreamSynchronize(s));
                h_granted = h_stager_mapped_[4];
                if (h_granted != h_new_count)
                {
                    throw std::runtime_error("TpccHybridStager: granted slots != requested slots, unexpected!");
                }
                tl_.mark("admit_bulk_insert_start");
                {

                    GpuTimer tg("admit_only:index_bulk_insert(gpu)");
                    // bulk insert CRID --> GRID and mark last epoch bit
                    crid2grid_index_.bulk_insert(d_new_, d_grids, h_new_count, s);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
                }


                tl_.mark("admit_upd_resident_start");
                {
                    GpuTimer tg("admit_only:update_resident_list(gpu)");
                    constexpr int block = 256;
                    int grid = (h_new_count + block - 1) / block;
                    k_set_grid2crid<<<grid, block, 0, s>>>(d_grids, d_new_, h_new_count, d_resident_list_);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
                }

                // Fast-path: when every admitted slot is an insert
                // (h_new_insert_count == h_new_count), miss_n is guaranteed
                // to be 0 by construction. Skip the D2H + CPU compaction +
                // SG transfer entirely.
                if (h_new_insert_count == h_new_count) {
                    Logger::GetInstance().Info("Admitting {} new records to GPU (miss-admit=0, all insert-allocate, fast-path).",
                                               h_granted);
                } else {
                    // transfer records (skip-admit on the insert subset)
                    std::vector<uint32_t> h_crids(h_granted);
                    std::vector<uint32_t> h_grids(h_granted);
                    std::vector<uint8_t> h_is_insert(h_granted);
                    tl_.mark("admit_d2h_start");
                    // Batch the three D2H copies with a single sync (mirrors
                    // the eviction path's pre_sg_d2h_start block); per-copy
                    // syncs would serialize the transfers and pay the sync
                    // overhead three times.
                    gpu_err_check(cudaMemcpyAsync(h_crids.data(), d_new_, sizeof(uint32_t) * h_granted, cudaMemcpyDeviceToHost, s));
                    gpu_err_check(cudaMemcpyAsync(h_grids.data(), d_grids, sizeof(uint32_t) * h_granted, cudaMemcpyDeviceToHost, s));
                    gpu_err_check(cudaMemcpyAsync(h_is_insert.data(), d_new_is_insert_,
                                                  sizeof(uint8_t) * h_granted, cudaMemcpyDeviceToHost, s));
                    tl_.mark("admit_sync_start");
                    gpu_err_check(cudaStreamSynchronize(s));
                    tl_.mark("admit_d2h_done");

                    // Build the miss-admit subset: skip every entry with is_insert=1.
                    // The mappings + resident list have already been installed for
                    // every entry above, so the cache slots for inserts are bound
                    // to their CRIDs even without an SG transfer; the executor's
                    // INSERT write will produce the slot's data, the dirty bit
                    // (driven by d_write_keys, which contains insert CRIDs) is set
                    // by the existing classify+mark step, and writeback ships it.
                    std::vector<uint32_t> h_crids_miss; h_crids_miss.reserve(h_granted);
                    std::vector<uint32_t> h_grids_miss; h_grids_miss.reserve(h_granted);
                    for (uint32_t i = 0; i < h_granted; ++i) {
                        if (!h_is_insert[i]) {
                            h_crids_miss.push_back(h_crids[i]);
                            h_grids_miss.push_back(h_grids[i]);
                        }
                    }
                    const uint32_t miss_n = static_cast<uint32_t>(h_crids_miss.size());
                    tl_.mark("admit_sg_xfer_start");
                    {
                        CpuTimer t_all("admit_only:scatter_gather_transfer(end2end)");
                        GpuTimer tg("admit_only:scatter_gather_transfer(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                        Logger::GetInstance().Info("Admitting {} new records to GPU (miss-admit={}, insert-allocate={} skipped).",
                                                   h_granted, miss_n, h_granted - miss_n);
                        if (miss_n > 0) {
                            sg_transfer_versions(h_crids_miss.data(), h_grids_miss.data(), miss_n, epoch);
                            sg_sync();
                        }
                    }
                }


            } else
            {
                Logger::GetInstance().Info("Evictions needed: new_count={} free_slots={}", h_new_count, h_free_slots);

                // else, we have to evict some old records first
                const uint32_t deficit = h_new_count - h_free_slots; // slots to evict
                // look up GRIDs for old CRIDs

                // FIFO Eviction based on ring order
                uint32_t* d_evict_crids = d_scratch_b_;
                uint32_t* d_evict_grids = d_scratch_c_;
                    uint32_t* d_cnt = d_scalar_b_;
                    uint32_t* d_maxoffset = d_scalar_c_;
                gpu_err_check(cudaMemsetAsync(d_cnt, 0, sizeof(uint32_t), s));
                gpu_err_check(cudaMemsetAsync(d_maxoffset, 0, sizeof(uint32_t), s));
                {
                    tl_.mark("fifo_evict_start");
                    GpuTimer tg("fifo:collect_evictions(gpu)");
                    constexpr int B = 256;
                    int G = 256;
                    // Reclaim-first pre-pass: drain flagged dead slots
                    // (delivered OL rows, deleted NO rows) first. Shares
                    // d_cnt with the FIFO scan below: if this fills the
                    // deficit, the FIFO scan early-exits at its first
                    // atomic check. If it doesn't, the FIFO scan continues
                    // filling from where this left off.
                    if (reclaim_eviction_active_) {
                        k_collect_evictions_reclaim_first<<<G, B, 0, s>>>(
                            gpu_capacity_,
                            d_resident_list_, d_needed_grid_flag_,
                            d_flush_pinned_flag_, d_reclaim_flag_,
                            deficit,
                            d_evict_grids, d_evict_crids,
                            d_cnt);
                        gpu_err_check(cudaPeekAtLastError());
                    }
                    // d_flush_pinned_flag_ is passed unconditionally: with
                    // overlap flush off it is all zeros and the kernel's
                    // pinned check is a no-op.
                    k_fifo_collect_evictions<<<G, B, 0, s>>>(fifo_next_grid_, gpu_capacity_,
                                                        d_resident_list_, d_needed_grid_flag_,
                                                        d_flush_pinned_flag_,
                                                        deficit,
                                                        d_evict_grids, d_evict_crids,
                                                        d_cnt, d_maxoffset);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
                }

                uint32_t h_cnt=0, h_maxoffset=0;
                k_copy_to_mapped<<<1, 1, 0, s>>>(d_cnt, d_mapped_scalar_b_);
                k_copy_to_mapped<<<1, 1, 0, s>>>(d_maxoffset, d_mapped_scalar_c_);
                gpu_err_check(cudaStreamSynchronize(s));
                h_cnt = h_stager_mapped_[5];
                h_maxoffset = h_stager_mapped_[6];


                if (h_cnt < deficit)
                {
                    throw std::runtime_error("TpccHybridStager: FIFO eviction could not find enough evictable entries!");
                }

                // advance cursor
                fifo_next_grid_ = fifo_next_grid_ + (h_maxoffset + 1);
                fifo_next_grid_ %= gpu_capacity_;






                // (end FIFO eviction)

                // No eviction-time writeback is needed: the periodic
                // flush (both sync and async) writes back all dirty records
                // every epoch, so evicted records are never dirty.
                // --------------------------------------------------------------

                // line up the new CRIDs, we need exactly deficit new admissions that won't use the free list
                // use the first deficit crids in d_new_
                uint32_t* d_new_first = d_new_;

                {
                    tl_.mark("bulk_update_start");
                    GpuTimer tg("evict:bulk_update_crid(gpu)");
                    // replace old crids from map, then insert newCRID ->evictedGRID
                    crid2grid_index_.bulk_update_crid(d_evict_crids, d_new_first, d_evict_grids, deficit, s);
        gpu_err_check(cudaStreamSynchronize(s));
                }
                {
                    tl_.mark("update_resident_start");
                    GpuTimer tg("evict:update_resident_list_for_renamed(gpu)");
                    constexpr int block = 256;
                    int grid = (deficit + block - 1) / block;
                    k_set_grid2crid<<<grid, block, 0, s>>>(d_evict_grids, d_new_first, deficit, d_resident_list_);
                    gpu_err_check(cudaPeekAtLastError());
                    // The renamed slots have new occupants that are not
                    // logically dead. Clear their reclaim flag so the next
                    // eviction pass does not drain them before Delivery or
                    // a delete has actually marked them.
                    if (reclaim_eviction_active_) {
                        k_clear_reclaim_by_grids<<<grid, block, 0, s>>>(
                            d_evict_grids, deficit, gpu_capacity_, d_reclaim_flag_);
                        gpu_err_check(cudaPeekAtLastError());
                    }
        gpu_err_check(cudaStreamSynchronize(s));
                }

                // if there are still some NEW left that could use the free list, do so
                const uint32_t still_new = h_new_count - deficit;
                if (still_new > 0)
                {


                    // Pop `still_new` GRIDs
                    uint32_t* d_grids2 = d_scratch_a_;
                    uint32_t* d_granted2 = d_scalar_a_;
                    {
                        GpuTimer tg("evict:ring_pop2(gpu)");
                        auto s = reinterpret_cast<cudaStream_t>(sg_stream());
                        launch_ring_pop_parallel(free_list_ring_, still_new, d_grids2, d_granted2, s);
                        gpu_err_check(cudaPeekAtLastError());
                        gpu_err_check(cudaStreamSynchronize(s)); // sync SG stream, not stream 0
                    }


                    gpu_err_check(cudaPeekAtLastError());
                    uint32_t h_granted2 = 0;
                    k_copy_to_mapped<<<1, 1, 0, s>>>(d_granted2, d_mapped_scalar_a_);
                    gpu_err_check(cudaStreamSynchronize(s));
                    h_granted2 = h_stager_mapped_[4];
                    if (h_granted2 != still_new) {
                        Logger::GetInstance().Error("Second ring pop granted {} < {}", h_granted2, still_new);
                    }

                    {
                        GpuTimer tg("evict:index_insert_from_ring(gpu)");
                        // The remaining NEW CRIDs start at d_new_ + deficit
                        crid2grid_index_.bulk_insert(d_new_ + deficit, d_grids2, h_granted2, s);
        gpu_err_check(cudaStreamSynchronize(s));
                    }


                    {
                        GpuTimer tg("evict:update_resident_list_from_ring(gpu)");
                        constexpr int block = 256;
                        int grid = (h_granted2 + block - 1) / block;
                        k_set_grid2crid<<<grid, block, 0, s>>>(d_grids2, d_new_ + deficit, h_granted2, d_resident_list_);
                        gpu_err_check(cudaPeekAtLastError());
                        // Same reasoning as the prior rename site -- new
                        // occupants are live, so reset the flag for these
                        // slots too.
                        if (reclaim_eviction_active_) {
                            k_clear_reclaim_by_grids<<<grid, block, 0, s>>>(
                                d_grids2, h_granted2, gpu_capacity_, d_reclaim_flag_);
                            gpu_err_check(cudaPeekAtLastError());
                        }
        gpu_err_check(cudaStreamSynchronize(s));
                    }

                    // Fast-path: when every admitted slot is an insert, the
                    // still_new subset is also all inserts → miss_n2=0. Skip
                    // the D2H + CPU loop + SG transfer entirely.
                    if (h_new_insert_count == h_new_count) {
                        Logger::GetInstance().Info("Admitting {} additional new records to GPU after eviction (miss-admit=0, all insert-allocate, fast-path).",
                                                   h_granted2);
                    } else {
                        // Transfers for these too (skip-admit on the insert subset)
                        std::vector<uint32_t> h_cr(h_granted2), h_gr(h_granted2);
                        std::vector<uint8_t> h_ins(h_granted2);
                        gpu_err_check(cudaMemcpyAsync(h_cr.data(), d_new_ + deficit, sizeof(uint32_t)*h_granted2, cudaMemcpyDeviceToHost, s));
                        gpu_err_check(cudaStreamSynchronize(s));
                        gpu_err_check(cudaMemcpyAsync(h_gr.data(), d_grids2,         sizeof(uint32_t)*h_granted2, cudaMemcpyDeviceToHost, s));
                        gpu_err_check(cudaStreamSynchronize(s));
                        gpu_err_check(cudaMemcpyAsync(h_ins.data(), d_new_is_insert_ + deficit,
                                                      sizeof(uint8_t)*h_granted2, cudaMemcpyDeviceToHost, s));
                        gpu_err_check(cudaStreamSynchronize(s));

                        // Compact to miss-admit subset.
                        std::vector<uint32_t> h_cr_miss; h_cr_miss.reserve(h_granted2);
                        std::vector<uint32_t> h_gr_miss; h_gr_miss.reserve(h_granted2);
                        for (uint32_t i = 0; i < h_granted2; ++i) {
                            if (!h_ins[i]) {
                                h_cr_miss.push_back(h_cr[i]);
                                h_gr_miss.push_back(h_gr[i]);
                            }
                        }
                        const uint32_t miss_n2 = static_cast<uint32_t>(h_cr_miss.size());
                        {
                            CpuTimer t_all("evict:scatter_gather_transfer2(end2end)");
                            GpuTimer tg("evict:scatter_gather_transfer2(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                            Logger::GetInstance().Info("Admitting {} additional new records to GPU after eviction (miss-admit={}, insert-allocate={} skipped).",
                                                       h_granted2, miss_n2, h_granted2 - miss_n2);
                            if (miss_n2 > 0) {
                                sg_transfer_versions(h_cr_miss.data(), h_gr_miss.data(), miss_n2, epoch);
                                sg_sync();
                            }
                        }
                    }

                }

                // transfer records for the "deficit" which reused the evicted slots
                {
                    tl_.mark("pre_sg_d2h_start");

                    // Fast-path: when every admitted slot is an insert, the
                    // deficit subset is also all inserts → miss_ev=0. Skip
                    // the pinned-buffer realloc, 4× D2H copies, deficit-sized
                    // CPU compaction loop, and SG transfer entirely. For OL
                    // in default tpcc this saves ~1.2 ms/epoch.
                    if (h_new_insert_count == h_new_count) {
                        tl_.mark("sg_transfer_start");
                        Logger::GetInstance().Info("Filling {} evicted slots with new records (miss-admit=0, all insert-allocate, fast-path).",
                                                   deficit);
                        tl_.mark("sg_transfer_end");
                    } else {
                        // GPU-side stream compaction: invert the is_insert flag
                        // on GPU, run cub::DeviceSelect::Flagged for crids and
                        // grids, then D2H only the count + filtered arrays
                        // (~3.5 MB for OL at w=128, no host filter loop; the
                        // naive D2H-everything + host filter costs ~9 MB of
                        // D2H plus a deficit-length CPU pass).
                        //
                        // Sub-phase markers (kept for plot symmetry):
                        //   d2h_launch_start  → start of GPU filter + D2H block
                        //   d2h_sync_start    → before the count-D2H sync
                        //   d2h_sync_done     → after count is on host
                        //   cpu_compact_done  → after the data-arrays D2H sync
                        tl_.mark("d2h_launch_start");

                        // Lazy-grow the device output buffers (cudaMalloc fires only
                        // when deficit grows beyond cap; with 2x headroom this is
                        // typically once during warmup).
                        if (deficit > d_ev_buffer_cap_) {
                            if (d_ev_cr_miss_) cudaFree(d_ev_cr_miss_);
                            if (d_ev_gr_miss_) cudaFree(d_ev_gr_miss_);
                            size_t new_cap = static_cast<size_t>(deficit) * 2;
                            if (new_cap > gpu_capacity_) new_cap = gpu_capacity_;
                            gpu_err_check(cudaMalloc(&d_ev_cr_miss_, sizeof(uint32_t) * new_cap));
                            gpu_err_check(cudaMalloc(&d_ev_gr_miss_, sizeof(uint32_t) * new_cap));
                            d_ev_buffer_cap_ = new_cap;
                        }

                        // Step 1: invert is_insert into d_filter_flags_ (1 = keep, miss-admit)
                        constexpr int B = 256;
                        const int G = (deficit + B - 1) / B;
                        k_invert_flag<<<G, B, 0, s>>>(d_new_is_insert_, d_filter_flags_, deficit);
                        gpu_err_check(cudaPeekAtLastError());

                        // Step 2: CUB DeviceSelect::Flagged for crids (uses d_filter_cub_tmp_)
                        {
                            size_t tmp_bytes = d_filter_cub_tmp_bytes_;
                            cub::DeviceSelect::Flagged(d_filter_cub_tmp_, tmp_bytes,
                                                       d_new_first, d_filter_flags_,
                                                       d_ev_cr_miss_, d_ev_miss_count_,
                                                       (int)deficit, s);
                        }
                        // Step 3: CUB DeviceSelect::Flagged for grids (same flags, same count)
                        {
                            size_t tmp_bytes = d_filter_cub_tmp_bytes_;
                            cub::DeviceSelect::Flagged(d_filter_cub_tmp_, tmp_bytes,
                                                       d_evict_grids, d_filter_flags_,
                                                       d_ev_gr_miss_, d_ev_miss_count_,
                                                       (int)deficit, s);
                        }

                        // Step 4: D2H the count first (tiny — ~µs sync)
                        uint32_t miss_ev_host = 0;
                        gpu_err_check(cudaMemcpyAsync(&miss_ev_host, d_ev_miss_count_,
                                                      sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
                        tl_.mark("d2h_sync_start");
                        gpu_err_check(cudaStreamSynchronize(s));
                        tl_.mark("d2h_sync_done");
                        const uint32_t miss_ev = miss_ev_host;

                        // Step 5: D2H exactly miss_ev items of crids and grids into the
                        // pinned host buffers. Grow if needed (rare — first epoch only).
                        if (miss_ev > h_evict_cap_) {
                            if (h_evict_crids_) cudaFreeHost(h_evict_crids_);
                            if (h_evict_grids_) cudaFreeHost(h_evict_grids_);
                            size_t new_cap = miss_ev * 2;
                            gpu_err_check(cudaHostAlloc(&h_evict_crids_, sizeof(uint32_t) * new_cap, cudaHostAllocDefault));
                            gpu_err_check(cudaHostAlloc(&h_evict_grids_, sizeof(uint32_t) * new_cap, cudaHostAllocDefault));
                            h_evict_cap_ = new_cap;
                        }
                        if (miss_ev > 0) {
                            gpu_err_check(cudaMemcpyAsync(h_evict_crids_, d_ev_cr_miss_,
                                                          sizeof(uint32_t)*miss_ev, cudaMemcpyDeviceToHost, s));
                            gpu_err_check(cudaMemcpyAsync(h_evict_grids_, d_ev_gr_miss_,
                                                          sizeof(uint32_t)*miss_ev, cudaMemcpyDeviceToHost, s));
                            gpu_err_check(cudaStreamSynchronize(s));
                        }
                        tl_.mark("cpu_compact_done");  // kept for plot symmetry — no CPU work here

                        {
                            tl_.mark("sg_transfer_start");
                            CpuTimer t_all("evict:scatter_gather_transfer_evicted(end2end)");
                            GpuTimer tg("evict:scatter_gather_transfer_evicted(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                            Logger::GetInstance().Info("Filling {} evicted slots with new records (miss-admit={}, insert-allocate={} skipped).",
                                                       deficit, miss_ev, deficit - miss_ev);
                            if (miss_ev > 0) {
                                sg_transfer_versions(h_evict_crids_, h_evict_grids_, miss_ev, epoch);
                                sg_sync();
                            }
                        }
                        tl_.mark("sg_transfer_end");
                    }

                    // (scatter handled by worker thread from start_flush_epoch_async)
                }


            }
        } else
        {

            auto &logger = Logger::GetInstance();
            Logger::GetInstance().Info("[SG] No new records to admit for this epoch.");
            Logger::GetInstance().Info("[SG] SG.transfer.total: {} us", 0);
        }

        // Dirty marking happens in the executor: gpuWriteToTableCoop sets
        // grid_dirty_v[slot][record_id] = 1 inline whenever it commits a
        // record_b write, using the per-table dirty arrays plumbed via
        // TpccRecords (populated by tpcc.cpp from each stager's
        // CridGridIndex::device_view). There is no staging-time dirty
        // pipeline.

        // Reset the version tags of freshly-inserted records' cache slots so the
        // executor's dual-version slot picker is deterministic (see
        // k_zero_insert_slot_tags). Every CRID in d_new_ is bound to a GRID by
        // now; this touches only insert-allocate entries (miss-admit keep their
        // SG-transferred tags). Placement-only change; record values are
        // unaffected.
        if (h_new_count > 0) {
            auto zit_view = crid2grid_index_.device_view();
            constexpr int ZB = 256;
            int ZG = (h_new_count + ZB - 1) / ZB;
            k_zero_insert_slot_tags<<<ZG, ZB, 0, s>>>(
                GPU_records_, d_new_, d_new_is_insert_,
                zit_view.d_crid_to_grid, zit_view.num_units, h_new_count);
            gpu_err_check(cudaPeekAtLastError());
            gpu_err_check(cudaStreamSynchronize(s));
        }

        // Reset count scalars for next epoch
        tl_.mark("wipe_start");
        {
            uint32_t z = 0;
            gpu_err_check(cudaMemcpyAsync(d_needed_count_, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, s));
            gpu_err_check(cudaMemcpyAsync(d_new_count_,    &z, sizeof(uint32_t), cudaMemcpyHostToDevice, s));

            // ensure all async operations are done
            gpu_err_check(cudaStreamSynchronize(s));
        }

        // The rebuild_start timeline marker stays for plot symmetry (the
        // overlap plots parse it); the flat cache index needs no periodic
        // rebuild.
        tl_.mark("rebuild_start");

        {
            {
                tl_.mark("promote_start");
                GpuTimer tg("prepareEpoch:promote_to_field_crid(gpu)");
                // No TPC-C promote step (that is a split-field-only YCSB
                // stage); the [TL] promote markers stay for plot symmetry.
            }


            // The CRID->GRID Remap is per-txn-type for TPC-C (5 different
            // TxnParams shapes) and lives in tpcc_hybrid_remap.{h,cu}. The
            // caller fetches our CridGridIndex via crid_to_grid_index() and
            // feeds it to the per-txn-type dispatch kernel.

            tl_.mark("end");
            tl_.dump(epoch);

        }



    }
    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::ensureHostDirtyCapacity(size_t n)
    {
        if (h_dirty_grids_.size() < n) h_dirty_grids_.resize(n);
        if (h_dirty_crids_.size() < n) h_dirty_crids_.resize(n);

        h_dirty_cap_ = std::max(h_dirty_cap_, n);
    }
    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::periodicFlush(uint32_t epoch)
    {
            constexpr uint32_t kFlushInterval = 1;
            if (epoch % kFlushInterval == 0) {

                // Device buffer for all dirty GRIDs (worst-case capacity)
                uint32_t num_dirty0 = 0, num_dirty1 = 0;
                {
                    GpuTimer tg("periodic_flush:collect_dirty_grids(gpu)");

                    num_dirty0 = crid2grid_index_.collect_dirty_grids_v1(d_dirty_grids_);
                    num_dirty1 = crid2grid_index_.collect_dirty_grids_v2(d_dirty_grids_slot1_);

        gpu_err_check(cudaStreamSynchronize(0));

                    Logger::GetInstance().Info(
                        "periodicFlush: found slot0={} slot1={} dirty GRIDs to flush at epoch {}",
                        num_dirty0, num_dirty1, epoch);
                }

                uint32_t num_dirty = num_dirty0 + num_dirty1;
                if (num_dirty == 0) {
                    Logger::GetInstance().Info("periodicFlush: no dirty GRIDs to flush at epoch {}", epoch);
                    return;
                }

                if (num_dirty > 0) {


                    {
                        // No classify step needed: dirty is guaranteed to be
                        // in the current slot, and the slot index follows
                        // directly from which dirty array (v1/v2) the GRID
                        // came from.
                    }

                    {
                        CpuTimer t("periodic_flush:host_buffer_resize");
                        ensureHostDirtyCapacity(num_dirty);
                    }

                    std::vector<uint8_t> h_slots(num_dirty);

                    // slot0 portion
                    if (num_dirty0 > 0) {
                        GpuTimer tg0("periodic_flush:copy_slot0_dirty_grids_to_host(gpu->cpu)");
                        gpu_err_check(cudaMemcpy(h_dirty_grids_.data(),
                                                 d_dirty_grids_,
                                                 sizeof(uint32_t) * num_dirty0,
                                                 cudaMemcpyDeviceToHost));
                    }

                    // slot1 portion
                    if (num_dirty1 > 0) {
                        GpuTimer tg1("periodic_flush:copy_slot1_dirty_grids_to_host(gpu->cpu)");
                        gpu_err_check(cudaMemcpy(h_dirty_grids_.data() + num_dirty0,
                                                 d_dirty_grids_slot1_,
                                                 sizeof(uint32_t) * num_dirty1,
                                                 cudaMemcpyDeviceToHost));
                    }
                    // slots known directly from source list
                    std::fill(h_slots.begin(), h_slots.begin() + num_dirty0, static_cast<uint8_t>(0));
                    std::fill(h_slots.begin() + num_dirty0, h_slots.end(), static_cast<uint8_t>(1));


                    // gather CRIDs by GRIDs
                    {
                        GpuTimer tg("periodic_flush:gather_crids_by_grids(gpu)");
                        if (num_dirty0 > 0) {
                            constexpr int B=256; int G=(num_dirty0 + B - 1)/B;
                            k_gather_crids_by_grids<<<G,B>>>(d_dirty_grids_, num_dirty0, d_resident_list_, d_dirty_crids_);
                            gpu_err_check(cudaPeekAtLastError());
                        }
                        if (num_dirty1 > 0) {
                            constexpr int B=256; int G=(num_dirty1 + B - 1)/B;
                            k_gather_crids_by_grids<<<G,B>>>(d_dirty_grids_slot1_, num_dirty1, d_resident_list_, d_dirty_crids_slot1_);
                            gpu_err_check(cudaPeekAtLastError());
                        }
        gpu_err_check(cudaStreamSynchronize(0));
                    }

                    // copy CRIDs to host
                    {
                        GpuTimer tg("periodic_flush:copy_crids_to_host(gpu->cpu)");
                        if (num_dirty0 > 0) {
                            gpu_err_check(cudaMemcpy(h_dirty_crids_.data(),
                                                     d_dirty_crids_,
                                                     sizeof(uint32_t) * num_dirty0,
                                                     cudaMemcpyDeviceToHost));
                        }
                        if (num_dirty1 > 0) {
                            gpu_err_check(cudaMemcpy(h_dirty_crids_.data() + num_dirty0,
                                                     d_dirty_crids_slot1_,
                                                     sizeof(uint32_t) * num_dirty1,
                                                     cudaMemcpyDeviceToHost));
                        }
        gpu_err_check(cudaStreamSynchronize(0));
                    }

                    // Flush GPU→CPU for all dirty GRIDs
                    {
                        GpuTimer tg("periodic_flush:writeback_dirty(gpu->cpu)");
                            sg_record_->writeback_versions(h_dirty_grids_.data(), h_dirty_crids_.data(), h_slots.data(), num_dirty,
                                                         GPU_records_,
                                                         CPU_records_,
                                                         gpu_capacity_);

                    }


                    // Clear dirty bits for those GRIDs
                    {
                        GpuTimer tg("periodic_flush:clear_dirty_bits(gpu)");

                        constexpr int B = 256;

                        if (num_dirty0 > 0) {
                            int G0 = (num_dirty0 + B - 1) / B;
                            k_fill_slots<<<G0, B>>>(d_clear_slots0_, num_dirty0, (uint8_t)0);
                            gpu_err_check(cudaPeekAtLastError());
                            // clear slot0 dirty bits for slot0 grids
                            crid2grid_index_.clear_dirty_by_grids(d_dirty_grids_, d_clear_slots0_, num_dirty0);
                        }

                        if (num_dirty1 > 0) {
                            int G1 = (num_dirty1 + B - 1) / B;
                            k_fill_slots<<<G1, B>>>(d_clear_slots1_, num_dirty1, (uint8_t)1);
                            gpu_err_check(cudaPeekAtLastError());
                            // clear slot1 dirty bits for slot1 grids
                            crid2grid_index_.clear_dirty_by_grids(d_dirty_grids_slot1_, d_clear_slots1_, num_dirty1);
                        }

        gpu_err_check(cudaStreamSynchronize(0));
                    }
                }
        }
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::mark_reclaimable(const uint32_t* d_crids, uint32_t n)
    {
        if (!reclaim_eviction_active_ || n == 0 || d_crids == nullptr) return;
        auto dv = crid2grid_index_.device_view();
        constexpr int B = 256;
        int G = (n + B - 1) / B;
        k_mark_reclaim_by_crids<<<G, B, 0, prep_stream_>>>(d_crids, n,
            dv.d_crid_to_grid, dv.num_units, gpu_capacity_, d_reclaim_flag_);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(prep_stream_));
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::set_flush_pins_for_next_epoch(const FlushHandle* inflight)
    {
        // Everything on the per-stager stream: this runs inside the 8-way
        // parallel flush sections, and a synchronous cudaMemset (like a
        // synchronous cudaMemcpy) is a device-wide sync point that must not
        // fire concurrently from 8 threads (same hazard class as the
        // prepareEpoch parallelization; see the stream comment in the ctor).
        const cudaStream_t s = active_stream();

        // Clear pins from previous epoch
        gpu_err_check(cudaMemsetAsync(d_flush_pinned_flag_, 0, gpu_capacity_ * sizeof(uint8_t), s));

        if (!inflight || !inflight->valid || inflight->n == 0) {
            gpu_err_check(cudaStreamSynchronize(s));
            return;
        }

        const uint32_t n = inflight->n;

        // Mark directly from the handle's device buffer (already on GPU from start_flush_epoch_async)
        constexpr int B = 256;
        int G = (n + B - 1) / B;
        k_mark_flag_by_grids<<<G, B, 0, s>>>(inflight->d_grids, n, gpu_capacity_, d_flush_pinned_flag_);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
    }



    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::sync_flush(FlushHandle& h) {
        // Persistent worker path: wait on the persistent worker.
        if (persistent_worker_) persistent_worker_->wait();
        if (h.worker.joinable()) h.worker.join();   // legacy: vestigial, no-op now

        // The flush set's dirty bits were already cleared at collect time
        // inside start_flush_epoch_async (see the comment there).

        // Keep device buffers allocated for reuse (freed by destructor or reset)
        h.n = 0;

        h.valid = false;
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::start_flush_epoch_async(uint32_t epoch, FlushHandle& out_handle)
    {
        // Defensive: drain any in-flight work before reusing the handle.
        // Persistent worker path: wait on the persistent worker.
        if (persistent_worker_) persistent_worker_->wait();
        if (out_handle.worker.joinable()) {
            out_handle.worker.join();   // legacy: vestigial
        }
        out_handle.valid = false;
        // Don't clear vectors or device buffers — reuse their capacity from previous epoch.
        out_handle.n = 0;

        // Per-stager stream for the flush setup too, so cross-stager
        // flush_async calls do not serialize via the legacy default stream.
        const cudaStream_t s = active_stream();

        // -------------------------
        // 1) Collect dirty GRIDs on device (synchronous list construction)
        // -------------------------
        uint32_t num_dirty0 = 0;
        uint32_t num_dirty1 = 0;
        {
            GpuTimer tg("flush_async:collect_dirty_grids(gpu)");
            num_dirty0 = crid2grid_index_.collect_dirty_grids_v1(d_dirty_grids_, s);
            num_dirty1 = crid2grid_index_.collect_dirty_grids_v2(d_dirty_grids_slot1_, s);
            // IMPORTANT: we need num_dirty now to size host vectors correctly.
            // collect_dirty_grids_v1/v2 sync internally on `s`, so num_dirty0
            // and num_dirty1 are valid host-side here. No outer sync needed.
        }

        Logger::GetInstance().Info("start_flush_epoch_async: epoch {} dirty_grids slot0={} slot1={}", epoch, num_dirty0, num_dirty1);

        if (num_dirty0 == 0 && num_dirty1 == 0) {
            // Nothing to do; handle stays invalid
            return;
        }

        // -------------------------
        // 2) Gather CRIDs for these GRIDs (device), classify curr slot if needed
        // -------------------------
        {
            GpuTimer tg("flush_async:gather_crids_by_grids(gpu)");
            constexpr int B = 256;
            if (num_dirty0 > 0)
            {
                int G = (num_dirty0 + B - 1) / B;

                k_gather_crids_by_grids<<<G, B, 0, s>>>(d_dirty_grids_, num_dirty0,
                                                  d_resident_list_, d_dirty_crids_);
            }
            if (num_dirty1 > 0)
            {
                int G1 = (num_dirty1 + B - 1) / B;
                k_gather_crids_by_grids<<<G1, B, 0, s>>>(d_dirty_grids_slot1_, num_dirty1,
                                                  d_resident_list_, d_dirty_crids_slot1_);
            }
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
        }

        // -------------------------
        // 2b) Sort each slot's (CRID -> GRID) pairs ascending by CRID.
        // The flush worker's host scatter walks h_crids in order; GRID order
        // (DeviceSelect output = ring/eviction order) is shuffled for the
        // static update tables once GRIDs recycle (measured: stock/customer
        // frac_seq < 0.02, mean |dCRID| ~ 600K at deck W=128 e40), which
        // costs a TLB/page walk per row on the pageable Primary Store.
        // Sorting restores a monotone walk. Slots are sorted independently
        // so the downstream [slot0... | slot1...] segment structure (h_slots
        // fills, pack pairing) is preserved.
        // -------------------------
        if (num_dirty0 > 1 || num_dirty1 > 1) {
            GpuTimer tg("flush_async:sort_flush_set(gpu)");
            if (!flush_sort_ready_) {
                gpu_err_check(cudaMalloc(&d_flush_sort_crids_, sizeof(uint32_t) * gpu_capacity_));
                gpu_err_check(cudaMalloc(&d_flush_sort_grids_, sizeof(uint32_t) * gpu_capacity_));
                size_t bytes = 0;
                cub::DeviceRadixSort::SortPairs(nullptr, bytes,
                    d_dirty_crids_, d_flush_sort_crids_,
                    d_dirty_grids_, d_flush_sort_grids_,
                    (int)gpu_capacity_, 0, 32, s);
                gpu_err_check(cudaMalloc(&d_flush_sort_tmp_, bytes));
                d_flush_sort_tmp_bytes_ = bytes;
                flush_sort_ready_ = true;
            }
            auto sort_slot = [&](uint32_t* d_crids, uint32_t* d_grids, uint32_t n) {
                if (n <= 1) return;
                size_t bytes = d_flush_sort_tmp_bytes_;
                cub::DeviceRadixSort::SortPairs(d_flush_sort_tmp_, bytes,
                    d_crids, d_flush_sort_crids_,
                    d_grids, d_flush_sort_grids_,
                    (int)n, 0, 32, s);
                // Copy sorted pairs back so every downstream consumer
                // (handle D2D, h_crids D2H) reads the canonical buffers.
                gpu_err_check(cudaMemcpyAsync(d_crids, d_flush_sort_crids_,
                                              sizeof(uint32_t) * n, cudaMemcpyDeviceToDevice, s));
                gpu_err_check(cudaMemcpyAsync(d_grids, d_flush_sort_grids_,
                                              sizeof(uint32_t) * n, cudaMemcpyDeviceToDevice, s));
            };
            sort_slot(d_dirty_crids_,       d_dirty_grids_,       num_dirty0);
            sort_slot(d_dirty_crids_slot1_, d_dirty_grids_slot1_, num_dirty1);
            // No host sync needed: all downstream consumers are on stream s.
        }

        const uint32_t num_dirty = num_dirty0 + num_dirty1;
        {
            GpuTimer tg("flush_async:alloc_handle_device_buffers(gpu)");
            if (num_dirty > out_handle.d_cap) {
                // Grow to the worst case (gpu_capacity_ = max possible dirty grids)
                // so this realloc fires at most once per stager and never again.
                // cudaMalloc/cudaFree are global GPU sync points and become very
                // expensive when prepareEpoch runs on per-stager streams in
                // parallel (observed up to 2.7 ms for a single realloc). Going
                // straight to gpu_capacity_ caps the cost at one warmup hit.
                const uint32_t target_cap = std::max(num_dirty, gpu_capacity_);
                if (out_handle.d_grids) cudaFree(out_handle.d_grids);
                if (out_handle.d_slots) cudaFree(out_handle.d_slots);
                gpu_err_check(cudaMalloc(&out_handle.d_grids, sizeof(uint32_t) * target_cap));
                gpu_err_check(cudaMalloc(&out_handle.d_slots, sizeof(uint8_t)  * target_cap));
                out_handle.d_cap = target_cap;
            }
            out_handle.n = num_dirty;
        }

        {
            GpuTimer tg("flush_async:copy_dirty_info_to_handle(gpu)");
            if (num_dirty0 > 0) {
                gpu_err_check(cudaMemcpyAsync(out_handle.d_grids,
                                         d_dirty_grids_,
                                         sizeof(uint32_t) * num_dirty0,
                                         cudaMemcpyDeviceToDevice, s));
            }
            if (num_dirty1 > 0) {
                gpu_err_check(cudaMemcpyAsync(out_handle.d_grids + num_dirty0,
                                         d_dirty_grids_slot1_,
                                         sizeof(uint32_t) * num_dirty1,
                                         cudaMemcpyDeviceToDevice, s));
            }
        }

        {
            GpuTimer tg("flush_async:init_handle_slots(gpu)");
            constexpr int B = 256;
            if (num_dirty0 > 0) {
                int G0 = (num_dirty0 + B - 1) / B;
                k_fill_slots<<<G0, B, 0, s>>>(out_handle.d_slots, num_dirty0, (uint8_t)0);
            }
            if (num_dirty1 > 0) {
                int G1 = (num_dirty1 + B - 1) / B;
                k_fill_slots<<<G1, B, 0, s>>>(out_handle.d_slots + num_dirty0, num_dirty1, (uint8_t)1);
            }
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(s));
        }

        // Clear the flush set's dirty bits here, at collect time, rather
        // than after the writeback drains in sync_flush. Two reasons.
        // (1) A deferred clear would run inside the fused flush block as a
        // kernel plus stream sync per stager on the critical path; here it
        // is stream-ordered behind the collect and rides the existing
        // h_crids sync. (2) A deferred clear runs after the NEXT epoch's
        // exec, so it is only correct while the whole flush set stays
        // pinned until then; a GRID recycled in between would have its
        // freshly-set executor dirty mark erased when the writer slot
        // collides with the flushed slot, silently dropping that record's
        // write from every future flush (reachable through the YCSB
        // stager's backpressure fallback, which legitimately drops pins
        // mid-epoch). Clearing at collect removes the coupling in both
        // stagers.
        crid2grid_index_.clear_dirty_by_grids(out_handle.d_grids, out_handle.d_slots, num_dirty, s);

        // -------------------------
        // 3) Copy flush set to host vectors inside the handle
        // -------------------------

        {
            // Resize host buffers (no-op after first epoch since capacity is
            // retained). h_grids is not populated (every consumer reads
            // handle.d_grids); only h_crids and h_slots are read downstream
            // by the worker scatter.
            {
                CpuTimer t("flush_async:host_buffer_resize");
                out_handle.h_crids.resize(num_dirty);
                out_handle.h_slots.resize(num_dirty);
            }

            // h_crids D2H: cudaMemcpyAsync + a single sync below; h_crids
            // only (h_grids is not read downstream).
            GpuTimer tg("flush_async:copy_flush_set_to_host(gpu->cpu)");
            gpu_err_check(cudaMemcpyAsync(out_handle.h_crids.data(),
                                          d_dirty_crids_,
                                          sizeof(uint32_t) * num_dirty0,
                                          cudaMemcpyDeviceToHost, s));
            gpu_err_check(cudaMemcpyAsync(out_handle.h_crids.data() + num_dirty0,
                                          d_dirty_crids_slot1_,
                                          sizeof(uint32_t) * num_dirty1,
                                          cudaMemcpyDeviceToHost, s));

            // fill slots in-place (no temp vector allocation)
            std::fill(out_handle.h_slots.begin(), out_handle.h_slots.begin() + num_dirty0, static_cast<uint8_t>(0));
            std::fill(out_handle.h_slots.begin() + num_dirty0, out_handle.h_slots.begin() + num_dirty, static_cast<uint8_t>(1));
            gpu_err_check(cudaStreamSynchronize(s));
        }

        // -------------------------
        // 5) D2D + pack on flush stream, then worker does chunked D2H+scatter
        // -------------------------
        flush_ensure_buffers(out_handle);
        {
            CpuTimer t("flush_async:gpu_sync");
            flush_queue_h2d(out_handle);
            flush_queue_pack(out_handle);
            flush_sync_gpu();
        }

        // valid must be set BEFORE set_flush_pins_for_next_epoch
        // because that function early-returns on !valid, silently leaving the
        // pin flag all-zero. With pins not actually set, eviction can pick
        // GRIDs holding E-1's dirty data, the GRID is reused for a new CRID,
        // and sync_flush(E-1) later erases the new CRID's executor write.
        out_handle.valid = true;

        // Set flush pins now -- no contention, GPU idle, d_grids ready. The
        // worker itself is handed the flush set by submit_flush_worker: right
        // after this call on non-durable runs, or after the epoch loop
        // advances the durable recovery marker in durable mode.
        set_flush_pins_for_next_epoch(&out_handle);
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::submit_flush_worker(FlushHandle& out_handle)
    {
        if (!out_handle.valid || out_handle.n == 0) return;
        CpuTimer t("flush_async:spawn_worker");
        // PERSISTENT WORKER PATH: submit() to the pre-pinned worker
        // thread instead of spawning std::thread per epoch.
        persistent_worker_->submit([this, &out_handle]() {
            try {
                CpuTimer t("flush_async:worker_total");
                const size_t flush_n = out_handle.n;
                Logger::GetInstance().Info("worker: n_grids={}", flush_n);

                auto w_t0 = std::chrono::high_resolution_clock::now();

                // Single-call writeback: wb_chunked_d2h_scatter handles
                // its own internal chunking (auto-tuned bytes-per-chunk).
                sg_record_flush_->wb_async_ensure(flush_n);
                Logger::GetInstance().Info("[SG][timeline] scatter_start");
                sg_record_flush_->wb_chunked_d2h_scatter(
                    out_handle.h_crids.data(), out_handle.h_slots.data(),
                    flush_n, CPU_records_);
                Logger::GetInstance().Info("[SG][timeline] scatter_end");
                auto w_t1 = std::chrono::high_resolution_clock::now();

                auto wus = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count(); };
                Logger::GetInstance().Info("worker_phases: total={} addr={}",
                    wus(w_t0,w_t1), (void*)this);

            } catch (const std::exception& e) {
                Logger::GetInstance().Error("flush_async worker threw: {}", e.what());
            }
        });
    }

    // --- Granular async flush phase control ---

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::flush_ensure_buffers(FlushHandle& handle)
    {
        std::call_once(flush_sg_once_flag_, [this]()
        {
                typename ScatterGather<TRec>::Options opt_flush{};
                opt_flush.own_stream = true;
                opt_flush.chunk_mode = false;
                opt_flush.async_writeback = true;
                sg_record_flush_->init(opt_flush, /*stream=*/nullptr);
            
        });

        sg_record_flush_->wb_async_ensure(handle.n);
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::flush_queue_h2d(FlushHandle& handle)
    {
        // Use D2D copy from handle's device buffers (skip host round-trip)
        sg_record_flush_->wb_async_queue_d2d(handle.d_grids, handle.d_slots, handle.n);
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::flush_queue_pack(FlushHandle& handle)
    {
        sg_record_flush_->wb_async_queue_pack(handle.n, GPU_records_);
    }

    template <typename TRec, typename TVer>
    void TpccHybridStager<TRec, TVer>::flush_sync_gpu()
    {
        sg_record_flush_->wb_async_sync();
    }

    template <typename TRec, typename TVer>
    uint64_t TpccHybridStager<TRec, TVer>::pcie_admit_h2d_bytes() const
    {
        uint64_t total = 0;
        if (sg_record_)        total += sg_record_->admit_h2d_payload_bytes();
        if (sg_record_flush_)  total += sg_record_flush_->admit_h2d_payload_bytes();
        return total;
    }

    template <typename TRec, typename TVer>
    uint64_t TpccHybridStager<TRec, TVer>::pcie_writeback_d2h_bytes() const
    {
        uint64_t total = 0;
        if (sg_record_)        total += sg_record_->writeback_d2h_payload_bytes();
        if (sg_record_flush_)  total += sg_record_flush_->writeback_d2h_payload_bytes();
        return total;
    }

    // Destructor: release allocator-owned and cudaMalloc-owned buffers.
    template <typename TRec, typename TVer>
    TpccHybridStager<TRec, TVer>::~TpccHybridStager()
    {
        auto free_alloc = [this](void*& p){
            if (p) { alloc_.Free(p); p = nullptr; }
        };
        auto free_cuda = [](void*& p){
            if (p) { cudaFree(p); p = nullptr; }
        };

        // allocator-owned buffers
        free_alloc(reinterpret_cast<void*&>(d_needed_));
        free_alloc(reinterpret_cast<void*&>(d_new_));
        free_alloc(reinterpret_cast<void*&>(d_resident_list_));

        free_alloc(reinterpret_cast<void*&>(d_needed_count_));
        free_alloc(reinterpret_cast<void*&>(d_new_count_));

        free_alloc(reinterpret_cast<void*&>(d_ring_buffer_));
        free_alloc(reinterpret_cast<void*&>(d_ring_head_));
        free_alloc(reinterpret_cast<void*&>(d_ring_tail_));

        free_alloc(reinterpret_cast<void*&>(d_needed_grid_flag_));

        free_alloc(reinterpret_cast<void*&>(d_dirty_grids_));
        free_alloc(reinterpret_cast<void*&>(d_dirty_crids_));
        free_alloc(reinterpret_cast<void*&>(d_dirty_grids_slot1_));
        free_alloc(reinterpret_cast<void*&>(d_dirty_crids_slot1_));

        free_alloc(reinterpret_cast<void*&>(d_keys_in_));
        free_alloc(reinterpret_cast<void*&>(d_keys_out_));

        // cudaMalloc-owned temp buffers
        free_cuda(reinterpret_cast<void*&>(d_ring_pop_scratch_));
        free_cuda(reinterpret_cast<void*&>(d_sort_tmp_));
        free_cuda(reinterpret_cast<void*&>(d_unique_tmp_));

        free_alloc(reinterpret_cast<void*&>(d_insert_bitmap_));
        free_alloc(reinterpret_cast<void*&>(d_new_is_insert_));
        free_alloc(reinterpret_cast<void*&>(d_new_insert_count_));

        free_cuda(reinterpret_cast<void*&>(d_flush_pinned_flag_));

        if (h_evict_crids_) { cudaFreeHost(h_evict_crids_); h_evict_crids_ = nullptr; }
        if (h_evict_grids_) { cudaFreeHost(h_evict_grids_); h_evict_grids_ = nullptr; }
        // GPU-side eviction-path stream-compaction lazy buffers
        free_cuda(reinterpret_cast<void*&>(d_ev_cr_miss_));
        free_cuda(reinterpret_cast<void*&>(d_ev_gr_miss_));
        free_alloc(reinterpret_cast<void*&>(d_ev_miss_count_));

        // Reclaim-first eviction per-slot flag
        free_alloc(reinterpret_cast<void*&>(d_reclaim_flag_));

        free_alloc(reinterpret_cast<void*&>(d_clear_slots0_));
        free_alloc(reinterpret_cast<void*&>(d_clear_slots1_));

        // host malloc
        if (first_needed_count_) {
            free(first_needed_count_);
            first_needed_count_ = nullptr;
        }

        // per-stager prepareEpoch stream
        if (prep_stream_ != 0) {
            cudaStreamDestroy(prep_stream_);
            prep_stream_ = 0;
        }
    }

    // Explicit instantiations — one per writeable TPC-C table. The template
    // method bodies live in this TU; this list tells the linker which
    // (TRec, TVer) pairs to materialize so callers in tpcc.cpp resolve.
    // History gets no stager (the executor doesn't read or write it).
    template class TpccHybridStager<Record<WarehouseValue>,  Version<WarehouseValue>>;
    template class TpccHybridStager<Record<DistrictValue>,   Version<DistrictValue>>;
    template class TpccHybridStager<Record<CustomerValue>,   Version<CustomerValue>>;
    template class TpccHybridStager<Record<ItemValue>,       Version<ItemValue>>;
    template class TpccHybridStager<Record<StockValue>,      Version<StockValue>>;
    template class TpccHybridStager<Record<NewOrderValue>,   Version<NewOrderValue>>;
    template class TpccHybridStager<Record<OrderValue>,      Version<OrderValue>>;
    template class TpccHybridStager<Record<OrderLineValue>,  Version<OrderLineValue>>;

} // namespace epic::tpcc

