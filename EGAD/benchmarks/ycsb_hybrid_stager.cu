//
// Created by Christian Tabbah on 2025-09-24.
//

// ycsb_hybrid_stager.cu
#include "ycsb_hybrid_stager.h"
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


namespace epic::ycsb {
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

    // Return nullptr if variant doesn't hold that type.
    static inline YcsbRecords* as_records(const YcsbRecordArrType& v) {
        if (auto p = std::get_if<YcsbRecords*>(&v)) return *p;
        return nullptr;
    }
    static inline YcsbVersions* as_versions(const YcsbVersionArrType& v) {
        if (auto p = std::get_if<YcsbVersions*>(&v)) return *p;
        return nullptr;
    }
    static inline YcsbFieldRecords* as_field_records(const YcsbRecordArrType& v) {
        if (auto p = std::get_if<YcsbFieldRecords*>(&v)) return *p;
        return nullptr;
    }
    static inline YcsbFieldVersions* as_field_versions(const YcsbVersionArrType& v) {
        if (auto p = std::get_if<YcsbFieldVersions*>(&v)) return *p;
        return nullptr;
    }

    __global__ void init_ring_ids(uint32_t* buf, uint32_t n) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) buf[i] = i;   // GRIDs 0..n-1
    }

    __global__ void k_fill_slots(uint8_t* slots, uint32_t n, uint8_t value) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) slots[i] = value;
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
        if (stencil[i] == sentinel) {
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


    // Kernel: for non-INSERT ops, turn record-level CRID into field-level CRID.
    __global__ void k_promote_to_field_crid(GpuTxnArray txns, uint32_t num_txns) {
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_txns) return;

        auto* base = txns.getTxn(tid);
        auto* p    = reinterpret_cast<YcsbTxnParam*>(base->data);

    #pragma unroll
        for (int i = 0; i < 10; ++i) {
            auto op = p->ops[i];
            // INSERT, FULL_READ, and FULL_READ_MODIFY_WRITE all touch ALL 10 fields of the
            // logical record, so promoting record_ids[i] to a single FCRID (one of the 10)
            // would lose the other 9 field identities. For these ops, record_ids[i] must
            // stay as the logical record id R so a subsequent pass can expand it into 10
            // per-field GRIDs (see fill_insert_field_grids / fill_full_read_field_grids).
            if (op == YcsbOpType::INSERT) continue;
            if (op == YcsbOpType::FULL_READ) continue;
            if (op == YcsbOpType::FULL_READ_MODIFY_WRITE) continue;
            // promote record CRID -> field CRID (single-field ops only)
            //   field_crid = record_crid * 10 + field_id (0..9)
            p->record_ids[i] = p->record_ids[i] * 10u + static_cast<uint32_t>(p->field_ids[i]);
        }
    }

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

    // For each CRID in d_new_, look up bitmap[crid] and write the per-slot
    // is_insert flag into d_new_is_insert_. Also bumps d_insert_count by 1
    // for every flag set, giving the host a prompt count of insert-allocate
    // slots to log alongside the existing slots-admitted accounting.
    __global__ void k_lookup_insert_flag_for_new(const uint32_t* __restrict__ d_new,
                                                  uint32_t n,
                                                  const uint8_t* __restrict__ bitmap,
                                                  uint8_t* __restrict__ d_new_is_insert,
                                                  uint32_t* __restrict__ d_insert_count)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        uint32_t crid = d_new[i];
        uint8_t is_insert = bitmap[crid];
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

    // Zero both dual-version tags of every
    // insert-allocate cache slot before execute, so the executor's slot picker
    // (version1<version2) deterministically writes the same slot for each fresh
    // insert regardless of the evicted predecessor's leftover tags. Without it,
    // insert-bearing YCSB (ycsbi) lands inserts in run-to-run non-deterministic
    // slots (value-equivalent but byte-divergent). Touches insert-allocate
    // entries only; miss-admit entries keep the tags written by their admission
    // SG transfer. Placement-only: record values are untouched, so the
    // values-only digest is unchanged.
    template <typename RecT>
    __global__ void k_zero_insert_slot_tags(RecT* records,
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




    uint32_t HybridStager::compute_staging_capacity(size_t usable_bytes,
                                                     size_t per_unit_payload,
                                                     uint32_t max_units)
    {
        // Per-slot overhead: one entry per gpu_capacity for each array the
        // constructor allocates.  Keep this in sync with the alloc_.Allocate()
        // calls below.
        //
        // 17 × uint32_t arrays:
        //   d_needed_, d_new_, d_resident_list_, d_ring_buffer_,
        //   d_dirty_grids_, d_dirty_crids_, d_dirty_grids_slot1_, d_dirty_crids_slot1_,
        //   d_scratch_a_..d_ and _f_ (5 arrays),
        //   plus four spare arrays of headroom in the estimate
        //
        // 7 × uint8_t arrays:
        //   d_needed_grid_flag_, d_new_is_insert_, d_flush_pinned_flag_,
        //   d_filter_flags_, d_clear_slots0_, d_clear_slots1_,
        //   plus one spare array of headroom in the estimate
        //
        // CridGridIndex per-capacity (created inside the constructor):
        //   grid_epoch_last (uint32_t), grid_dirty_v1 (uint8_t), grid_dirty_v2 (uint8_t)

        constexpr size_t stager_u32_arrays = 17;
        constexpr size_t stager_u8_arrays  = 7;
        constexpr size_t index_per_slot    = sizeof(uint32_t) + 2 * sizeof(uint8_t); // 6 bytes

        constexpr size_t per_slot_overhead =
            stager_u32_arrays * sizeof(uint32_t) +   // 68
            stager_u8_arrays  * sizeof(uint8_t)  +   // 7
            index_per_slot;                           // 6
        // Total: 81 bytes per slot

        uint64_t capacity = usable_bytes / (per_unit_payload + per_slot_overhead);
        if (capacity > max_units) capacity = max_units;
        return static_cast<uint32_t>(capacity);
    }

    HybridStager::HybridStager(Allocator& alloc,
                               YcsbRecordArrType GPU_records, YcsbVersionArrType GPU_versions,
                               YcsbRecordArrType CPU_records,
                               uint32_t gpu_capacity, uint32_t num_units, bool split_field)
        : alloc_(alloc),
          split_field_(split_field),
          GPU_records_(GPU_records),
          GPU_versions_(GPU_versions),
          CPU_records_(CPU_records),
          gpu_capacity_(gpu_capacity),
          num_units_(num_units),
          crid2grid_index_(alloc_, gpu_capacity_, num_units_) // capacity=gpu slots, num_units=total CRIDs
    {
        auto &logger = Logger::GetInstance();
        // allocate device memory for the various arrays
        logger.Info("Initializing HybridStager with GPU capacity: " + std::to_string(gpu_capacity_));

        // Make sure variants match split mode
        if (split_field_)
        {
            if (!as_field_records(GPU_records_) || !as_field_versions(GPU_versions_) ||
                !as_field_records(CPU_records_))
            {
                throw std::runtime_error("HybridStager: split_field=true but records/versions are not field-split variants.");
            }
        } else
        {
            if (!as_records(GPU_records_) || !as_versions(GPU_versions_) ||
                !as_records(CPU_records_))
            {
                throw std::runtime_error("HybridStager: split_field=false but records/versions are not non-split variants.");
            }
        }

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

        //init the right scatter gather
        if (split_field_)
        {
            sg_field_ = std::make_unique<ScatterGather<YcsbFieldRecords>>();
            ScatterGather<YcsbFieldRecords>::Options opt{};
            opt.own_stream = true;
            opt.chunk_mode = true;
            opt.async_writeback = false;
            sg_field_->init(opt);

            sg_field_flush_ = std::make_unique<ScatterGather<YcsbFieldRecords>>();
        } else
        {
            sg_record_ = std::make_unique<ScatterGather<YcsbRecords>>();
            ScatterGather<YcsbRecords>::Options opt{};
            opt.own_stream = true;
            opt.chunk_mode = true;
            opt.async_writeback = false;
            sg_record_->init(opt);

            sg_record_flush_ = std::make_unique<ScatterGather<YcsbRecords>>();
        }

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

        // Pre-allocate the CUB select temp buffer at max capacity so no
        // cudaMalloc (a device-wide sync point) fires during staging
        {
            size_t tmp_bytes = 0;
            cub::DeviceSelect::Flagged(nullptr, tmp_bytes,
                                       d_scratch_b_, d_filter_flags_, d_scratch_d_, d_scalar_b_,
                                       (int)gpu_capacity_, 0);
            gpu_err_check(cudaMalloc(&d_filter_cub_tmp_, tmp_bytes));
            d_filter_cub_tmp_bytes_ = tmp_bytes;
        }
        d_scalar_a_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_scalar_b_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));
        d_scalar_c_  = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t)));

        d_clear_slots0_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));
        d_clear_slots1_ = static_cast<uint8_t*>(alloc_.Allocate(gpu_capacity_ * sizeof(uint8_t)));

        // Mapped memory for scalar readbacks (bypasses D2H copy engine)
        gpu_err_check(cudaHostAlloc(&h_stager_mapped_, kMappedScalarCount_ * sizeof(uint32_t), cudaHostAllocMapped));
        memset(h_stager_mapped_, 0, kMappedScalarCount_ * sizeof(uint32_t));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_needed_count_,      &h_stager_mapped_[0], 0));
        // Slots [1] (d_mapped_load_needed_count_) and [2] (d_mapped_write_only_count_)
        // were the readbacks for the unwired write_only optimization
        // (buildNeedsLoadOnDevice + buildWriteOnlyFromTouchAndNeeds, both
        // removed). Slots intentionally unbound; keep kMappedScalarCount_ at
        // 13 so the offsets of [3..12] stay stable.
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_new_count_,         &h_stager_mapped_[3], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_a_,          &h_stager_mapped_[4], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_b_,          &h_stager_mapped_[5], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_scalar_c_,          &h_stager_mapped_[6], 0));
        // [7] (d_mapped_write_unique_) was the staging-time mark_dirty pipeline's
        // unique-count readback. Slot left intentionally unbound now that the
        // executor handles dirty marking inline; keep the kMappedScalarCount_
        // size unchanged so the layout stays compatible with anything reading
        // h_stager_mapped_[12] (d_mapped_new_insert_count_).
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_ring_head_,         &h_stager_mapped_[8], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_ring_tail_,         &h_stager_mapped_[10], 0));
        gpu_err_check(cudaHostGetDevicePointer(&d_mapped_new_insert_count_,  &h_stager_mapped_[12], 0));

        logger.Info("HybridStager initialization complete.");

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
            // Warmup bulk_update_crid (first called at epoch ~100, eviction transition)
            crid2grid_index_.bulk_update_crid(d_scratch_b_, d_scratch_c_, d_scratch_a_, 1);
            // The bulk_update_crid warmup runs k_flat_rename_crids with zeroed
            // scratch (oldc=newc=grid=0), which leaves d_crid_to_grid[0]=0
            // instead of the sentinel. That would alias CRID 0 onto GRID 0 (a
            // later admission reuses still-free GRID 0 for a different CRID,
            // so two CRIDs share one cache slot), so restore the cache index
            // to all-sentinel here.
            {
                auto dv = crid2grid_index_.device_view();
                gpu_err_check(cudaMemsetAsync(const_cast<uint32_t*>(dv.d_crid_to_grid),
                    0xff, static_cast<size_t>(dv.num_units) * sizeof(uint32_t), 0));
            }
            gpu_err_check(cudaStreamSynchronize(0));
            logger.Info("Warmup launches complete.");
        }

    }

        void HybridStager::buildNeededOnDevice(const uint32_t* d_all_keys, uint32_t n_keys, uint32_t epoch)
    {
        if (n_keys == 0) {
            uint32_t z = 0;
            gpu_err_check(cudaMemcpy(d_needed_count_, &z, sizeof(uint32_t), cudaMemcpyHostToDevice));
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
            gpu_err_check(cudaMemcpyAsync(d_keys_in_, d_all_keys, sizeof(uint32_t) * n_keys, cudaMemcpyDeviceToDevice, 0));
        }

        // ---- sort (CUB, cached temp) ----
        {
            GpuTimer ts("build_needed: sort");

            if (sort_tmp_bytes_ == 0) {
                // sizing call
                cub::DeviceRadixSort::SortKeys(nullptr, sort_tmp_bytes_,
                                               d_keys_in_, d_keys_out_,
                                               n_keys);
                gpu_err_check(cudaMalloc(&d_sort_tmp_, sort_tmp_bytes_));
            }

            cub::DeviceRadixSort::SortKeys(d_sort_tmp_, sort_tmp_bytes_,
                                           d_keys_in_, d_keys_out_,
                                           n_keys);
        gpu_err_check(cudaStreamSynchronize(0));
        }

        // ---- unique (CUB, cached temp) ----
        // unique output goes back into d_keys_in_ (ping-pong reuse)
        {
            GpuTimer tu("build_needed: unique");

            if (unique_tmp_bytes_ == 0) {
                cub::DeviceSelect::Unique(nullptr, unique_tmp_bytes_,
                                          d_keys_out_, d_keys_in_,
                                          d_needed_count_,  // device scalar for count
                                          n_keys);
                gpu_err_check(cudaMalloc(&d_unique_tmp_, unique_tmp_bytes_));
            }

            cub::DeviceSelect::Unique(d_unique_tmp_, unique_tmp_bytes_,
                                      d_keys_out_, d_keys_in_,
                                      d_needed_count_,
                                      n_keys);
        }

        // ---- pull unique_count to host for capacity check ----
        uint32_t unique_count = 0;
        k_copy_to_mapped<<<1, 1>>>(d_needed_count_, d_mapped_needed_count_);
        gpu_err_check(cudaStreamSynchronize(0));
        unique_count = h_stager_mapped_[0];

        if (unique_count > gpu_capacity_) {
            throw std::runtime_error("HybridStager::buildNeededOnDevice: unique needed keys exceed GPU capacity.");
        }

        // ---- epoch==1 resize d_needed_ ----
        if (epoch == 1) {
            alloc_.Free(d_needed_);
            *first_needed_count_ = unique_count * 2;
            d_needed_ = static_cast<uint32_t*>(alloc_.Allocate(sizeof(uint32_t) * (*first_needed_count_)));
        }

        // ---- store canonical needed list from d_keys_in_ ----
        gpu_err_check(cudaMemcpy(d_needed_, d_keys_in_,
                                 unique_count * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice));

        // d_needed_count_ is already set on device by CUB Unique
    }

    void HybridStager::sg_transfer_versions(const uint32_t* h_crids, const uint32_t* h_grids, uint32_t n, uint32_t epoch)
    {
        if (n == 0) {
            return;
        }
        if (split_field_) {
            sg_field_->transfer_versions(h_crids, h_grids, epoch, n,
                as_field_records(CPU_records_),
                as_field_records(GPU_records_));
        } else {
            sg_record_->transfer_versions(h_crids, h_grids, epoch, n,
                as_records(CPU_records_),
                as_records(GPU_records_));
        }

    }

    void HybridStager::sg_sync()
    {
        if (split_field_) sg_field_->sync();
        else              sg_record_->sync();
    }

    void* HybridStager::sg_stream()
    {
        return split_field_ ? sg_field_->get_stream() : sg_record_->get_stream();
    }

    void HybridStager::prepareEpoch(uint32_t epoch, TxnArray<YcsbTxnParam>& txn_array, TxnArray<YcsbExecPlan>& plan_array,
        const uint32_t* d_all_keys, uint32_t n_all,
        const uint32_t* d_read_keys, uint32_t n_read,
        const uint32_t* d_write_keys, uint32_t n_write,
        const uint32_t* d_insert_keys, uint32_t n_insert,
        FlushHandle* flush_handle)
    {
        CpuTimer tc("prepareEpoch(cpu)");
        tl_.reset();
        tl_.mark("start");

        // Reset the per-CRID insert bitmap and re-populate from d_insert_keys.
        // Each non-sentinel entry in d_insert_keys is the CRID of an INSERT
        // op, so bitmap[crid] = 1 marks the CRID as a fresh-insert target this
        // epoch. We use this bitmap below to split d_new_ into the miss-admit
        // subset (regular SG transfer from CPU primary store) and the
        // insert-allocate subset (skip the SG transfer; the executor will
        // produce the cache slot's data via its INSERT write).
        // Fast-path: if there are no inserts this epoch (e.g. ycsbf), skip the
        // bitmap zero + populate. The lookup kernel below and the SG admission
        // sites further down also short-circuit on n_insert==0, so the bitmap
        // contents are not read this epoch and stale data is harmless.
        if (d_insert_keys && n_all > 0 && n_insert > 0) {
            GpuTimer tg("prepareEpoch:set_insert_bitmap(gpu)");
            gpu_err_check(cudaMemsetAsync(d_insert_bitmap_, 0, num_units_ * sizeof(uint8_t), 0));
            constexpr int B = 256;
            int G = (n_all + B - 1) / B;
            k_set_insert_bitmap<<<G, B>>>(d_insert_keys, n_all, num_units_, d_insert_bitmap_);
            gpu_err_check(cudaPeekAtLastError());
            gpu_err_check(cudaStreamSynchronize(0));
        }


        // 1. Build the d_needed_ array from exec_params.
        {
            tl_.mark("build_needed_start");
            GpuTimer tg("build_needed(gpu)");
            buildNeededOnDevice(d_all_keys, n_all, epoch);
            gpu_err_check(cudaStreamSynchronize(0));
        }
        uint32_t h_needed_count = 0;
        k_copy_to_mapped<<<1, 1>>>(d_needed_count_, d_mapped_needed_count_);
        gpu_err_check(cudaStreamSynchronize(0));
        h_needed_count = h_stager_mapped_[0];
        Logger::GetInstance().Info("Needed count for epoch " + std::to_string(epoch) + " is " + std::to_string(h_needed_count));

        uint32_t h_new_count = 0;
        {
        // 2. Clear needed grid flags and mark needed grids(only needed for FIFO and histogram)
        gpu_err_check(cudaMemsetAsync(d_needed_grid_flag_, 0, gpu_capacity_ * sizeof(uint8_t), 0));
        // lookup needed CRIDs --> GRIDs
            uint32_t* d_needed_grids = d_scratch_a_;
        {
            tl_.mark("fifo_mark_start");
            GpuTimer tg("fifo:lookup_and_mark_needed_grids(gpu)");
            // combine the above two steps into one kernel to save some GPU round-trips and also allow touch-based optimizations in the index
            crid2grid_index_.bulk_lookup_crid_to_grid_and_mark_needed(
                d_needed_, d_needed_grids, h_needed_count, d_needed_grid_flag_, gpu_capacity_);
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        }

        {
            GpuTimer tg("index:touch_needed_grids(gpu)");
            // TODO (cleanup): crid2grid_index_.touch_by_grids only writes
            // grid_epoch_last, which no eviction or other consumer reads.
            // The cursor-based FIFO eviction does not use it. Remove this
            // call, the touch_by_grids method, the k_touch_by_grids kernel,
            // and the grid_epoch_last array if the LRU policy is not coming back.
            crid2grid_index_.touch_by_grids(d_needed_grids, h_needed_count, epoch);
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
            gpu_err_check(cudaMemsetAsync(d_new_count_, 0, sizeof(uint32_t), 0));
            {
                constexpr int B = 256;
                int G = (h_needed_count + B - 1) / B;
                k_copy_if_stencil_eq<<<G, B>>>(
                    d_needed_, d_needed_grids, h_needed_count, 0xffffffffu,
                    d_new_, d_new_count_);
                gpu_err_check(cudaPeekAtLastError());
            }
            gpu_err_check(cudaStreamSynchronize(0));
            k_copy_to_mapped<<<1, 1>>>(d_new_count_, d_mapped_new_count_);
            gpu_err_check(cudaStreamSynchronize(0));
            h_new_count = h_stager_mapped_[3];
            gpu_err_check(cudaMemcpyAsync(d_new_count_, &h_new_count, sizeof(uint32_t), cudaMemcpyHostToDevice, 0));
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
            gpu_err_check(cudaMemsetAsync(d_new_insert_count_, 0, sizeof(uint32_t), 0));
            constexpr int B = 256;
            int G = (h_new_count + B - 1) / B;
            k_lookup_insert_flag_for_new<<<G, B>>>(
                d_new_, h_new_count, d_insert_bitmap_, d_new_is_insert_, d_new_insert_count_);
            gpu_err_check(cudaPeekAtLastError());
            k_copy_to_mapped<<<1, 1>>>(d_new_insert_count_, d_mapped_new_insert_count_);
            gpu_err_check(cudaStreamSynchronize(0));
            h_new_insert_count = h_stager_mapped_[12];
        }
        // (When n_insert==0 the SG admission sites short-circuit directly,
        // so d_new_is_insert_ is never read in that case.)
        const uint32_t h_new_miss_count =
            (h_new_count >= h_new_insert_count) ? (h_new_count - h_new_insert_count) : 0;
        if (h_new_count > 0) {
            Logger::GetInstance().Info("new={} split: miss-admit={} insert-allocate={}",
                                       h_new_count, h_new_miss_count, h_new_insert_count);
        }
        unsigned long long h_head = 0, h_tail = 0;
        k_copy_ull_to_mapped<<<1, 1>>>(d_ring_head_, d_mapped_ring_head_);
        k_copy_ull_to_mapped<<<1, 1>>>(d_ring_tail_, d_mapped_ring_tail_);
        gpu_err_check(cudaStreamSynchronize(0));
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
                {
                    GpuTimer tg("admit_only:ring_pop(gpu)");
                    auto s = reinterpret_cast<cudaStream_t>(sg_stream());
                    launch_ring_pop_parallel(free_list_ring_, h_new_count, d_grids, d_granted, s);
                    gpu_err_check(cudaPeekAtLastError());
                    gpu_err_check(cudaStreamSynchronize(s)); // sync SG stream, not stream 0
                }

                uint32_t h_granted = 0;
                k_copy_to_mapped<<<1, 1>>>(d_granted, d_mapped_scalar_a_);
                gpu_err_check(cudaStreamSynchronize(0));
                h_granted = h_stager_mapped_[4];
                if (h_granted != h_new_count)
                {
                    throw std::runtime_error("HybridStager: granted slots != requested slots, unexpected!");
                }
                {

                    GpuTimer tg("admit_only:index_bulk_insert(gpu)");
                    // bulk insert CRID --> GRID and mark last epoch bit
                    crid2grid_index_.bulk_insert(d_new_, d_grids, h_new_count);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                }


                {
                    GpuTimer tg("admit_only:update_resident_list(gpu)");
                    constexpr int block = 256;
                    int grid = (h_new_count + block - 1) / block;
                    k_set_grid2crid<<<grid, block>>>(d_grids, d_new_, h_new_count, d_resident_list_);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                }

                // transfer records (skip-admit on the insert subset)
                std::vector<uint32_t> h_crids(h_granted);
                std::vector<uint32_t> h_grids(h_granted);
                gpu_err_check(cudaMemcpyAsync(h_crids.data(), d_new_, sizeof(uint32_t) * h_granted, cudaMemcpyDeviceToHost, 0));
                gpu_err_check(cudaStreamSynchronize(0));
                gpu_err_check(cudaMemcpyAsync(h_grids.data(), d_grids, sizeof(uint32_t) * h_granted, cudaMemcpyDeviceToHost, 0));
                gpu_err_check(cudaStreamSynchronize(0));

                // Fast-path: when there are no inserts this epoch, the
                // miss-admit subset IS the full set, so skip the per-entry
                // is_insert copy + walk. Otherwise build the miss-admit subset
                // by skipping every entry with is_insert=1 — the mappings +
                // resident list have already been installed for every entry,
                // so cache slots for inserts are bound to their CRIDs without
                // an SG transfer; the executor's INSERT write produces the
                // slot's data and writeback ships it.
                std::vector<uint32_t> h_crids_miss;
                std::vector<uint32_t> h_grids_miss;
                const uint32_t* miss_crids_ptr = h_crids.data();
                const uint32_t* miss_grids_ptr = h_grids.data();
                uint32_t miss_n = h_granted;
                if (n_insert > 0) {
                    std::vector<uint8_t> h_is_insert(h_granted);
                    gpu_err_check(cudaMemcpyAsync(h_is_insert.data(), d_new_is_insert_,
                                                  sizeof(uint8_t) * h_granted, cudaMemcpyDeviceToHost, 0));
                    gpu_err_check(cudaStreamSynchronize(0));
                    h_crids_miss.reserve(h_granted);
                    h_grids_miss.reserve(h_granted);
                    for (uint32_t i = 0; i < h_granted; ++i) {
                        if (!h_is_insert[i]) {
                            h_crids_miss.push_back(h_crids[i]);
                            h_grids_miss.push_back(h_grids[i]);
                        }
                    }
                    miss_n = static_cast<uint32_t>(h_crids_miss.size());
                    miss_crids_ptr = h_crids_miss.data();
                    miss_grids_ptr = h_grids_miss.data();
                }
                {
                    CpuTimer t_all("admit_only:scatter_gather_transfer(end2end)");
                    GpuTimer tg("admit_only:scatter_gather_transfer(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                    Logger::GetInstance().Info("Admitting {} new records to GPU (miss-admit={}, insert-allocate={} skipped).",
                                               h_granted, miss_n, h_granted - miss_n);
                    if (miss_n > 0) {
                        sg_transfer_versions(miss_crids_ptr, miss_grids_ptr, miss_n, epoch);
                        sg_sync();
                    }
                }


            } else
            {
                Logger::GetInstance().Info("Evictions needed: new_count={} free_slots={}", h_new_count, h_free_slots);

                // else, we have to evict some old records first
                const uint32_t deficit = h_new_count - h_free_slots; // slots to evict
                // look up GRIDs for old CRIDs

                // FIFO Eviction based on ring order

                // Create an array of whether a grid is in the needed set or not
                uint32_t* d_evict_crids = d_scratch_b_;
                uint32_t* d_evict_grids = d_scratch_c_;

                    uint32_t* d_cnt = d_scalar_b_;
                    uint32_t* d_maxoffset = d_scalar_c_;
                gpu_err_check(cudaMemsetAsync(d_cnt, 0, sizeof(uint32_t), 0));
                gpu_err_check(cudaMemsetAsync(d_maxoffset, 0, sizeof(uint32_t), 0));
                {
                    tl_.mark("fifo_evict_start");
                    GpuTimer tg("fifo:collect_evictions(gpu)");
                    constexpr int B = 256;
                    int G = 256;
                    // Note that we pass in d_flush_pinned_flag regardless of whether its flush_overlap_enabled
                    // Because even if its not, its just 0s and we are just ||ing it in the kernel.
                    k_fifo_collect_evictions<<<G, B>>>(fifo_next_grid_, gpu_capacity_,
                                                        d_resident_list_, d_needed_grid_flag_,
                                                        d_flush_pinned_flag_,
                                                        deficit,
                                                        d_evict_grids, d_evict_crids,
                                                        d_cnt, d_maxoffset);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                }

                uint32_t h_cnt=0, h_maxoffset=0;
                k_copy_to_mapped<<<1, 1>>>(d_cnt, d_mapped_scalar_b_);
                k_copy_to_mapped<<<1, 1>>>(d_maxoffset, d_mapped_scalar_c_);
                gpu_err_check(cudaStreamSynchronize(0));
                h_cnt = h_stager_mapped_[5];
                h_maxoffset = h_stager_mapped_[6];


                if (h_cnt < deficit)
                {
                    // Backpressure fallback for very small cache ratios: the
                    // in-flight writeback's pinned GRIDs plus this epoch's
                    // needed set can exceed the evictable pool (seen at a 5%
                    // cache: ~479K pinned of 1M slots). The pins only protect
                    // flush data until the worker's scatter lands, so wait for
                    // the worker, drop the pins, and collect once more. Any
                    // configuration that finds enough candidates on the first
                    // pass never enters this branch. The sg-transfer gate must
                    // be released first: the worker parks on sg_transfer_cv_
                    // between its two scatter halves waiting for this epoch's
                    // admit transfer, which only runs after eviction, so
                    // joining without signaling would deadlock. Signaling
                    // early just lets the second half overlap the upcoming
                    // gather (the pause is a contention optimization, not a
                    // correctness gate); this epoch's flush spawn re-arms the
                    // flag for its own worker.
                    if (flush_handle && flush_handle->valid && flush_handle->worker.joinable())
                    {
                        Logger::GetInstance().Info(
                            "fifo eviction short ({} evictable, {} needed): waiting for in-flight writeback, dropping its pins, retrying",
                            h_cnt, deficit);
                        signal_sg_transfer_done();
                        flush_handle->worker.join();
                        gpu_err_check(cudaMemset(d_flush_pinned_flag_, 0, gpu_capacity_ * sizeof(uint8_t)));
                        gpu_err_check(cudaMemsetAsync(d_cnt, 0, sizeof(uint32_t), 0));
                        gpu_err_check(cudaMemsetAsync(d_maxoffset, 0, sizeof(uint32_t), 0));
                        {
                            GpuTimer tg("fifo:collect_evictions_retry(gpu)");
                            constexpr int B = 256;
                            int G = 256;
                            k_fifo_collect_evictions<<<G, B>>>(fifo_next_grid_, gpu_capacity_,
                                                                d_resident_list_, d_needed_grid_flag_,
                                                                d_flush_pinned_flag_,
                                                                deficit,
                                                                d_evict_grids, d_evict_crids,
                                                                d_cnt, d_maxoffset);
                            gpu_err_check(cudaPeekAtLastError());
                            gpu_err_check(cudaStreamSynchronize(0));
                        }
                        k_copy_to_mapped<<<1, 1>>>(d_cnt, d_mapped_scalar_b_);
                        k_copy_to_mapped<<<1, 1>>>(d_maxoffset, d_mapped_scalar_c_);
                        gpu_err_check(cudaStreamSynchronize(0));
                        h_cnt = h_stager_mapped_[5];
                        h_maxoffset = h_stager_mapped_[6];
                    }
                    if (h_cnt < deficit)
                    {
                        throw std::runtime_error("HybridStager: FIFO eviction could not find enough evictable entries!");
                    }
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
                    crid2grid_index_.bulk_update_crid(d_evict_crids, d_new_first, d_evict_grids, deficit);
        gpu_err_check(cudaStreamSynchronize(0));
                }
                {
                    tl_.mark("update_resident_start");
                    GpuTimer tg("evict:update_resident_list_for_renamed(gpu)");
                    constexpr int block = 256;
                    int grid = (deficit + block - 1) / block;
                    k_set_grid2crid<<<grid, block>>>(d_evict_grids, d_new_first, deficit, d_resident_list_);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                }

                // if there are still some NEW left that could use the free list, do so
                const uint32_t still_new = h_new_count - deficit;
                if (still_new > 0)
                {


                    // Pop `still_needed` GRIDs
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
                    k_copy_to_mapped<<<1, 1>>>(d_granted2, d_mapped_scalar_a_);
                    gpu_err_check(cudaStreamSynchronize(0));
                    h_granted2 = h_stager_mapped_[4];
                    if (h_granted2 != still_new) {
                        Logger::GetInstance().Error("Second ring pop granted {} < {}", h_granted2, still_new);
                    }

                    {
                        GpuTimer tg("evict:index_insert_from_ring(gpu)");
                        // The remaining NEW CRIDs start at d_new_ + deficit
                        crid2grid_index_.bulk_insert(d_new_ + deficit, d_grids2, h_granted2);
        gpu_err_check(cudaStreamSynchronize(0));
                    }


                    {
                        GpuTimer tg("evict:update_resident_list_from_ring(gpu)");
                        constexpr int block = 256;
                        int grid = (h_granted2 + block - 1) / block;
                        k_set_grid2crid<<<grid, block>>>(d_grids2, d_new_ + deficit, h_granted2, d_resident_list_);
                        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                    }

                    // Transfers for these too (skip-admit on the insert subset)
                    std::vector<uint32_t> h_cr(h_granted2), h_gr(h_granted2);
                    gpu_err_check(cudaMemcpyAsync(h_cr.data(), d_new_ + deficit, sizeof(uint32_t)*h_granted2, cudaMemcpyDeviceToHost, 0));
                    gpu_err_check(cudaStreamSynchronize(0));
                    gpu_err_check(cudaMemcpyAsync(h_gr.data(), d_grids2,         sizeof(uint32_t)*h_granted2, cudaMemcpyDeviceToHost, 0));
                    gpu_err_check(cudaStreamSynchronize(0));

                    // Fast-path on n_insert==0 (see no-eviction site above).
                    std::vector<uint32_t> h_cr_miss;
                    std::vector<uint32_t> h_gr_miss;
                    const uint32_t* miss2_crids_ptr = h_cr.data();
                    const uint32_t* miss2_grids_ptr = h_gr.data();
                    uint32_t miss_n2 = h_granted2;
                    if (n_insert > 0) {
                        std::vector<uint8_t> h_ins(h_granted2);
                        gpu_err_check(cudaMemcpyAsync(h_ins.data(), d_new_is_insert_ + deficit,
                                                      sizeof(uint8_t)*h_granted2, cudaMemcpyDeviceToHost, 0));
                        gpu_err_check(cudaStreamSynchronize(0));
                        h_cr_miss.reserve(h_granted2);
                        h_gr_miss.reserve(h_granted2);
                        for (uint32_t i = 0; i < h_granted2; ++i) {
                            if (!h_ins[i]) {
                                h_cr_miss.push_back(h_cr[i]);
                                h_gr_miss.push_back(h_gr[i]);
                            }
                        }
                        miss_n2 = static_cast<uint32_t>(h_cr_miss.size());
                        miss2_crids_ptr = h_cr_miss.data();
                        miss2_grids_ptr = h_gr_miss.data();
                    }
                    {
                        CpuTimer t_all("evict:scatter_gather_transfer2(end2end)");
                        GpuTimer tg("evict:scatter_gather_transfer2(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                        Logger::GetInstance().Info("Admitting {} additional new records to GPU after eviction (miss-admit={}, insert-allocate={} skipped).",
                                                   h_granted2, miss_n2, h_granted2 - miss_n2);
                        if (miss_n2 > 0) {
                            sg_transfer_versions(miss2_crids_ptr, miss2_grids_ptr, miss_n2, epoch);
                            sg_sync();
                        }
                    }

                }

                // transfer records for the "deficit" which reused the evicted slots
                {
                    tl_.mark("pre_sg_d2h_start");
                    // Ensure pinned buffers are large enough (allocate 2x to avoid repeated realloc)
                    if (deficit > h_evict_cap_) {
                        if (h_evict_crids_) cudaFreeHost(h_evict_crids_);
                        if (h_evict_grids_) cudaFreeHost(h_evict_grids_);
                        size_t new_cap = deficit * 2;
                        gpu_err_check(cudaHostAlloc(&h_evict_crids_, sizeof(uint32_t) * new_cap, cudaHostAllocDefault));
                        gpu_err_check(cudaHostAlloc(&h_evict_grids_, sizeof(uint32_t) * new_cap, cudaHostAllocDefault));
                        h_evict_cap_ = new_cap;
                    }

                    // Pinned D2H: truly async, batched with single sync
                    gpu_err_check(cudaMemcpyAsync(h_evict_crids_, d_new_first, sizeof(uint32_t)*deficit, cudaMemcpyDeviceToHost, 0));
                    gpu_err_check(cudaMemcpyAsync(h_evict_grids_, d_evict_grids, sizeof(uint32_t)*deficit, cudaMemcpyDeviceToHost, 0));
                    gpu_err_check(cudaStreamSynchronize(0));

                    // Fast-path on n_insert==0 (see no-eviction site above).
                    // d_new_is_insert_[0..deficit) lines up with d_new_first / d_evict_grids.
                    std::vector<uint32_t> h_ev_cr_miss;
                    std::vector<uint32_t> h_ev_gr_miss;
                    const uint32_t* miss_ev_crids_ptr = h_evict_crids_;
                    const uint32_t* miss_ev_grids_ptr = h_evict_grids_;
                    uint32_t miss_ev = deficit;
                    if (n_insert > 0) {
                        std::vector<uint8_t> h_ev_ins(deficit);
                        gpu_err_check(cudaMemcpyAsync(h_ev_ins.data(), d_new_is_insert_,
                                                      sizeof(uint8_t)*deficit, cudaMemcpyDeviceToHost, 0));
                        gpu_err_check(cudaStreamSynchronize(0));
                        h_ev_cr_miss.reserve(deficit);
                        h_ev_gr_miss.reserve(deficit);
                        for (uint32_t i = 0; i < deficit; ++i) {
                            if (!h_ev_ins[i]) {
                                h_ev_cr_miss.push_back(h_evict_crids_[i]);
                                h_ev_gr_miss.push_back(h_evict_grids_[i]);
                            }
                        }
                        miss_ev = static_cast<uint32_t>(h_ev_cr_miss.size());
                        miss_ev_crids_ptr = h_ev_cr_miss.data();
                        miss_ev_grids_ptr = h_ev_gr_miss.data();
                    }

                    {
                        tl_.mark("sg_transfer_start");
                        CpuTimer t_all("evict:scatter_gather_transfer_evicted(end2end)");
                        GpuTimer tg("evict:scatter_gather_transfer_evicted(gpu)", reinterpret_cast<cudaStream_t>(sg_stream()));
                        Logger::GetInstance().Info("Filling {} evicted slots with new records (miss-admit={}, insert-allocate={} skipped).",
                                                   deficit, miss_ev, deficit - miss_ev);
                        if (miss_ev > 0) {
                            sg_transfer_versions(miss_ev_crids_ptr, miss_ev_grids_ptr, miss_ev, epoch);
                            sg_sync();
                        }
                    }
                    tl_.mark("sg_transfer_end");

                    // (scatter handled by worker thread from start_flush_epoch_async)
                }


            }
        } else
        {
            Logger::GetInstance().Info("[SG] No new records to admit for this epoch.");
            Logger::GetInstance().Info("[SG] SG.transfer.total: {} us", 0);
        }

        // Signal worker that SG transfer is complete (worker sleeps until this fires)
        signal_sg_transfer_done();

        // Dirty marking happens in the executor: gpuWriteToTableCoop /
        // gpuWriteToTableThread set grid_dirty_v[slot][record_id] = 1 inline
        // whenever they commit a record_b write (see
        // HybridExecutor::set_dirty_arrays in ycsb_hybrid_executor.h).
        // There is no staging-time dirty pipeline.

        // Reset count scalars for next epoch
        tl_.mark("wipe_start");
        {
            uint32_t z = 0;
            gpu_err_check(cudaMemcpyAsync(d_needed_count_, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, 0));
            gpu_err_check(cudaMemcpyAsync(d_new_count_,    &z, sizeof(uint32_t), cudaMemcpyHostToDevice, 0));

            // ensure all async operations are done
            gpu_err_check(cudaStreamSynchronize(0));
        }

        // The rebuild_start timeline marker stays for plot symmetry (the
        // overlap plots parse it); the flat cache index needs no periodic
        // rebuild.
        tl_.mark("rebuild_start");

        {
            {
                tl_.mark("promote_start");
                GpuTimer tg("prepareEpoch:promote_to_field_crid(gpu)");
                if (split_field_) {
                    constexpr int B = 256;
                    int G = (txn_array.num_txns + B - 1) / B;
                    k_promote_to_field_crid<<<G, B>>>(GpuTxnArray(txn_array), txn_array.num_txns);
                    gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
                }
            }


            // Now remap CRIDs -> GRIDs.
            // Split mode: skips INSERT/FULL_READ/FULL_RMW (their per-field GRIDs are
            //   filled by fill_{insert,full_read}_field_grids below).
            // Non-split mode: remaps every op; executor reads record_ids[i] as a single GRID.
            {
                tl_.mark("remap_start");
                GpuTimer tg("prepareEpoch:remap_txn_record_ids(gpu)");
                crid2grid_index_.remap_txn_record_ids(txn_array, txn_array.num_txns, gpu_capacity_,
                                                       /*split_field=*/split_field_);
        gpu_err_check(cudaStreamSynchronize(0));
            }

            {
                GpuTimer tg("prepareEpoch:fill_insert_field_grids(gpu)");
                if (split_field_) {
                    // in HybridStager::prepareEpoch(...) after admissions/renames and before execute():
                    crid2grid_index_.fill_insert_field_grids(txn_array, plan_array, txn_array.num_txns);
                }
            }

            {
                GpuTimer tg("prepareEpoch:fill_full_read_field_grids(gpu)");
                if (split_field_) {
                    // FULL_READ / FULL_READ_MODIFY_WRITE were skipped by promote+remap,
                    // so record_ids[i] still holds logical R. Expand into 10 per-field
                    // GRIDs via d_crid_to_grid[R*10 + j], stored in the plan.
                    crid2grid_index_.fill_full_read_field_grids(txn_array, plan_array, txn_array.num_txns);
                }
            }

            // Zero insert-allocate slot tags before execute so insert
            // placement is deterministic. Every
            // new CRID is bound to a GRID by now; this touches only the
            // insert-allocate subset (d_new_is_insert_). ycsbf has no inserts
            // -> h_new_insert_count==0 -> no-op (and baseline-equivalent).
            if (h_new_insert_count > 0) {
                auto zv = crid2grid_index_.device_view();
                constexpr int ZB = 256;
                int ZG = (h_new_count + ZB - 1) / ZB;
                if (split_field_) {
                    k_zero_insert_slot_tags<<<ZG, ZB>>>(
                        as_field_records(GPU_records_), d_new_, d_new_is_insert_,
                        zv.d_crid_to_grid, zv.num_units, h_new_count);
                } else {
                    k_zero_insert_slot_tags<<<ZG, ZB>>>(
                        as_records(GPU_records_), d_new_, d_new_is_insert_,
                        zv.d_crid_to_grid, zv.num_units, h_new_count);
                }
                gpu_err_check(cudaPeekAtLastError());
                gpu_err_check(cudaStreamSynchronize(0));
            }

            tl_.mark("end");
            tl_.dump(epoch);

        }



    }
    void HybridStager::ensureHostDirtyCapacity(size_t n)
    {
        if (h_dirty_grids_.size() < n) h_dirty_grids_.resize(n);
        if (h_dirty_crids_.size() < n) h_dirty_crids_.resize(n);

        h_dirty_cap_ = std::max(h_dirty_cap_, n);
    }
    void HybridStager::periodicFlush(uint32_t epoch)
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
                        if (split_field_) {
                            sg_field_->writeback_versions(h_dirty_grids_.data(), h_dirty_crids_.data(), h_slots.data(), num_dirty,
                                                         as_field_records(GPU_records_),
                                                         as_field_records(CPU_records_),
                                                         gpu_capacity_);
                        } else {
                            sg_record_->writeback_versions(h_dirty_grids_.data(), h_dirty_crids_.data(), h_slots.data(), num_dirty,
                                                         as_records(GPU_records_),
                                                         as_records(CPU_records_),
                                                         gpu_capacity_);
                        }
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

    void HybridStager::set_flush_pins_for_next_epoch(const FlushHandle* inflight)
    {
        // Clear pins from previous epoch
        gpu_err_check(cudaMemset(d_flush_pinned_flag_, 0, gpu_capacity_ * sizeof(uint8_t)));

        if (!inflight || !inflight->valid || inflight->n == 0) return;

        const uint32_t n = inflight->n;

        // Mark directly from the handle's device buffer (already on GPU from start_flush_epoch_async)
        constexpr int B = 256;
        int G = (n + B - 1) / B;
        k_mark_flag_by_grids<<<G, B>>>(inflight->d_grids, n, gpu_capacity_, d_flush_pinned_flag_);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
    }



    void HybridStager::sync_flush(FlushHandle& h) {
        if (h.worker.joinable()) h.worker.join();

        // The flush set's dirty bits were already cleared at collect time
        // inside start_flush_epoch_async (see the comment there).

        // Keep device buffers allocated for reuse (freed by destructor or reset)
        h.n = 0;

        h.valid = false;
    }

    void HybridStager::start_flush_epoch_async(uint32_t epoch, FlushHandle& out_handle)
    {
        // Defensive: drain any still-running worker before reusing the handle.
        if (out_handle.worker.joinable()) {
            out_handle.worker.join();
        }
        out_handle.valid = false;
        // Don't clear vectors or device buffers — reuse their capacity from previous epoch.
        out_handle.n = 0;

        // -------------------------
        // 1) Collect dirty GRIDs on device (synchronous list construction)
        // -------------------------
        uint32_t num_dirty0 = 0;
        uint32_t num_dirty1 = 0;
        {
            GpuTimer tg("flush_async:collect_dirty_grids(gpu)");
            num_dirty0 = crid2grid_index_.collect_dirty_grids_v1(d_dirty_grids_);
            num_dirty1 = crid2grid_index_.collect_dirty_grids_v2(d_dirty_grids_slot1_);
            // IMPORTANT: we need num_dirty now to size host vectors correctly.
            // This sync is happening at *end of epoch* before next epoch starts,
            // so it doesn't ruin overlap between epochs.
        gpu_err_check(cudaStreamSynchronize(0));
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

                k_gather_crids_by_grids<<<G, B>>>(d_dirty_grids_, num_dirty0,
                                                  d_resident_list_, d_dirty_crids_);
            }
            if (num_dirty1 > 0)
            {
                int G1 = (num_dirty1 + B - 1) / B;
                k_gather_crids_by_grids<<<G1, B>>>(d_dirty_grids_slot1_, num_dirty1,
                                                  d_resident_list_, d_dirty_crids_slot1_);
            }
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        }

        const uint32_t num_dirty = num_dirty0 + num_dirty1;
        {
            GpuTimer tg("flush_async:alloc_handle_device_buffers(gpu)");
            if (num_dirty > out_handle.d_cap) {
                if (out_handle.d_grids) cudaFree(out_handle.d_grids);
                if (out_handle.d_slots) cudaFree(out_handle.d_slots);
                gpu_err_check(cudaMalloc(&out_handle.d_grids, sizeof(uint32_t) * num_dirty));
                gpu_err_check(cudaMalloc(&out_handle.d_slots, sizeof(uint8_t)  * num_dirty));
                out_handle.d_cap = num_dirty;
            }
            out_handle.n = num_dirty;
        }

        {
            GpuTimer tg("flush_async:copy_dirty_info_to_handle(gpu)");
            if (num_dirty0 > 0) {
                gpu_err_check(cudaMemcpy(out_handle.d_grids,
                                         d_dirty_grids_,
                                         sizeof(uint32_t) * num_dirty0,
                                         cudaMemcpyDeviceToDevice));
            }
            if (num_dirty1 > 0) {
                gpu_err_check(cudaMemcpy(out_handle.d_grids + num_dirty0,
                                         d_dirty_grids_slot1_,
                                         sizeof(uint32_t) * num_dirty1,
                                         cudaMemcpyDeviceToDevice));
            }
        }

        {
            GpuTimer tg("flush_async:init_handle_slots(gpu)");
            constexpr int B = 256;
            if (num_dirty0 > 0) {
                int G0 = (num_dirty0 + B - 1) / B;
                k_fill_slots<<<G0, B>>>(out_handle.d_slots, num_dirty0, (uint8_t)0);
            }
            if (num_dirty1 > 0) {
                int G1 = (num_dirty1 + B - 1) / B;
                k_fill_slots<<<G1, B>>>(out_handle.d_slots + num_dirty0, num_dirty1, (uint8_t)1);
            }
            gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        }

        // Clear the flush set's dirty bits here, at collect time, rather
        // than after the writeback drains in sync_flush. A deferred clear
        // runs after the NEXT epoch's exec, so it is only correct while the
        // whole flush set stays pinned until then — and the backpressure
        // fallback in prepareEpoch legitimately drops the pins mid-epoch.
        // A GRID recycled after that drop would have its freshly-set
        // executor dirty mark erased when the writer slot collides with the
        // flushed slot, silently dropping that record's write from every
        // future flush. Clearing here, before this epoch's eviction can
        // recycle anything, closes that window.
        crid2grid_index_.clear_dirty_by_grids(out_handle.d_grids, out_handle.d_slots, num_dirty);
        gpu_err_check(cudaStreamSynchronize(0));

        // -------------------------
        // 3) Copy flush set to host vectors inside the handle
        // -------------------------

        {
            // h_grids is not populated (every consumer reads handle.d_grids);
            // only h_crids/h_slots are read downstream.
            {
                CpuTimer t("flush_async:host_buffer_resize");
                out_handle.h_crids.resize(num_dirty);
                out_handle.h_slots.resize(num_dirty);
            }

            GpuTimer tg("flush_async:copy_flush_set_to_host(gpu->cpu)");
            gpu_err_check(cudaMemcpyAsync(out_handle.h_crids.data(),
                                          d_dirty_crids_,
                                          sizeof(uint32_t) * num_dirty0,
                                          cudaMemcpyDeviceToHost, 0));
            gpu_err_check(cudaMemcpyAsync(out_handle.h_crids.data() + num_dirty0,
                                          d_dirty_crids_slot1_,
                                          sizeof(uint32_t) * num_dirty1,
                                          cudaMemcpyDeviceToHost, 0));

            // fill slots in-place (no temp vector allocation)
            std::fill(out_handle.h_slots.begin(), out_handle.h_slots.begin() + num_dirty0, static_cast<uint8_t>(0));
            std::fill(out_handle.h_slots.begin() + num_dirty0, out_handle.h_slots.begin() + num_dirty, static_cast<uint8_t>(1));
            gpu_err_check(cudaStreamSynchronize(0));
        }

        // -------------------------
        // 5) D2D + pack on flush stream, then worker does chunked D2H+scatter
        // -------------------------
        const bool split_field = split_field_;

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
        sg_transfer_done_.store(false, std::memory_order_release);
        out_handle.valid = true;

        // Set flush pins before spawning worker -- no contention, GPU idle, d_grids ready
        set_flush_pins_for_next_epoch(&out_handle);
        {
            CpuTimer t("flush_async:spawn_worker");
            auto t_pre_spawn = std::chrono::high_resolution_clock::now();
            out_handle.worker = std::thread([this, &out_handle, split_field, t_pre_spawn]() {
                try {
                    auto t_entry = std::chrono::high_resolution_clock::now();
                    CpuTimer t("flush_async:worker_total");
                    pin_current_thread_to_node1_cores();
                    auto t_post_pin = std::chrono::high_resolution_clock::now();
                    const size_t flush_n = out_handle.n;
                    Logger::GetInstance().Info("worker: n_grids={}  spawn_to_entry_us={}  entry_to_pin_us={}",
                        flush_n,
                        std::chrono::duration_cast<std::chrono::microseconds>(t_entry - t_pre_spawn).count(),
                        std::chrono::duration_cast<std::chrono::microseconds>(t_post_pin - t_entry).count());

                    auto w_t0 = std::chrono::high_resolution_clock::now();

                    if (split_field) {
                        // Normal chunked D2H+scatter
                        sg_field_flush_->wb_async_ensure(flush_n);
                        const size_t half = flush_n / 2;

                        // First half: chunked D2H+scatter (overlaps with main thread indexing/submission)
                        Logger::GetInstance().Info("[SG][timeline] scatter_start");
                        sg_field_flush_->wb_chunked_d2h_scatter(
                            out_handle.h_crids.data(), out_handle.h_slots.data(),
                            flush_n, as_field_records(CPU_records_),
                            0, half);
                        auto w_t1 = std::chrono::high_resolution_clock::now();

                        // Wait for SG transfer to complete
                        {
                            std::unique_lock<std::mutex> lk(sg_transfer_mutex_);
                            sg_transfer_cv_.wait(lk, [this]{ return sg_transfer_done_.load(std::memory_order_acquire); });
                        }
                        auto w_t2 = std::chrono::high_resolution_clock::now();

                        // Second half: chunked D2H+scatter (after SG, no DRAM contention)
                        sg_field_flush_->wb_chunked_d2h_scatter(
                            out_handle.h_crids.data(), out_handle.h_slots.data(),
                            flush_n, as_field_records(CPU_records_),
                            half, flush_n - half);
                        Logger::GetInstance().Info("[SG][timeline] scatter_end");
                        auto w_t3 = std::chrono::high_resolution_clock::now();

                        auto wus = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count(); };
                        Logger::GetInstance().Info("worker_phases: half1={} sg_wait={} half2={} total={}",
                            wus(w_t0,w_t1), wus(w_t1,w_t2), wus(w_t2,w_t3), wus(w_t0,w_t3));
                    } else {
                        // Mirror of split branch using sg_record_flush_ on YcsbRecords.
                        sg_record_flush_->wb_async_ensure(flush_n);
                        const size_t half = flush_n / 2;

                        Logger::GetInstance().Info("[SG][timeline] scatter_start");
                        sg_record_flush_->wb_chunked_d2h_scatter(
                            out_handle.h_crids.data(), out_handle.h_slots.data(),
                            flush_n, as_records(CPU_records_),
                            0, half);
                        auto w_t1 = std::chrono::high_resolution_clock::now();

                        {
                            std::unique_lock<std::mutex> lk(sg_transfer_mutex_);
                            sg_transfer_cv_.wait(lk, [this]{ return sg_transfer_done_.load(std::memory_order_acquire); });
                        }
                        auto w_t2 = std::chrono::high_resolution_clock::now();

                        sg_record_flush_->wb_chunked_d2h_scatter(
                            out_handle.h_crids.data(), out_handle.h_slots.data(),
                            flush_n, as_records(CPU_records_),
                            half, flush_n - half);
                        Logger::GetInstance().Info("[SG][timeline] scatter_end");
                        auto w_t3 = std::chrono::high_resolution_clock::now();

                        auto wus = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count(); };
                        Logger::GetInstance().Info("worker_phases: half1={} sg_wait={} half2={} total={}",
                            wus(w_t0,w_t1), wus(w_t1,w_t2), wus(w_t2,w_t3), wus(w_t0,w_t3));
                    }

                } catch (const std::exception& e) {
                    Logger::GetInstance().Error("flush_async worker threw: {}", e.what());
                }
            });
            auto t_post_spawn = std::chrono::high_resolution_clock::now();
            Logger::GetInstance().Info("main: spawn_ctor_us={}",
                std::chrono::duration_cast<std::chrono::microseconds>(t_post_spawn - t_pre_spawn).count());
        }

    }

    // --- Granular async flush phase control ---

    void HybridStager::flush_ensure_buffers(FlushHandle& handle)
    {
        std::call_once(flush_sg_once_flag_, [this]()
        {
            if (split_field_)
            {
                ScatterGather<YcsbFieldRecords>::Options opt_flush{};
                opt_flush.own_stream = true;
                opt_flush.chunk_mode = false;
                opt_flush.async_writeback = true;
                sg_field_flush_->init(opt_flush, /*stream=*/nullptr);
            } else
            {
                ScatterGather<YcsbRecords>::Options opt_flush{};
                opt_flush.own_stream = true;
                opt_flush.chunk_mode = false;
                opt_flush.async_writeback = true;
                sg_record_flush_->init(opt_flush, /*stream=*/nullptr);
            }
        });

        if (split_field_)
            sg_field_flush_->wb_async_ensure(handle.n);
        else
            sg_record_flush_->wb_async_ensure(handle.n);
    }

    void HybridStager::flush_queue_h2d(FlushHandle& handle)
    {
        // Use D2D copy from handle's device buffers (skip host round-trip)
        if (split_field_)
            sg_field_flush_->wb_async_queue_d2d(handle.d_grids, handle.d_slots, handle.n);
        else
            sg_record_flush_->wb_async_queue_d2d(handle.d_grids, handle.d_slots, handle.n);
    }

    void HybridStager::flush_queue_pack(FlushHandle& handle)
    {
        if (split_field_)
            sg_field_flush_->wb_async_queue_pack(handle.n, as_field_records(GPU_records_));
        else
            sg_record_flush_->wb_async_queue_pack(handle.n, as_records(GPU_records_));
    }

    void HybridStager::flush_sync_gpu()
    {
        if (split_field_)
            sg_field_flush_->wb_async_sync();
        else
            sg_record_flush_->wb_async_sync();
    }

    uint64_t HybridStager::pcie_admit_h2d_bytes() const
    {
        uint64_t total = 0;
        if (sg_record_)        total += sg_record_->admit_h2d_payload_bytes();
        if (sg_field_)         total += sg_field_->admit_h2d_payload_bytes();
        if (sg_record_flush_)  total += sg_record_flush_->admit_h2d_payload_bytes();
        if (sg_field_flush_)   total += sg_field_flush_->admit_h2d_payload_bytes();
        return total;
    }

    uint64_t HybridStager::pcie_writeback_d2h_bytes() const
    {
        uint64_t total = 0;
        if (sg_record_)        total += sg_record_->writeback_d2h_payload_bytes();
        if (sg_field_)         total += sg_field_->writeback_d2h_payload_bytes();
        if (sg_record_flush_)  total += sg_record_flush_->writeback_d2h_payload_bytes();
        if (sg_field_flush_)   total += sg_field_flush_->writeback_d2h_payload_bytes();
        return total;
    }

    // Destructor: release allocator-owned and cudaMalloc-owned buffers.
    HybridStager::~HybridStager()
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

        free_alloc(reinterpret_cast<void*&>(d_clear_slots0_));
        free_alloc(reinterpret_cast<void*&>(d_clear_slots1_));

        // host malloc
        if (first_needed_count_) {
            free(first_needed_count_);
            first_needed_count_ = nullptr;
        }
    }


} // namespace epic::ycsb

