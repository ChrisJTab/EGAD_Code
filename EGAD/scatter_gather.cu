//
// Created by Christian Tabbah on 2025-10-31
//

#ifdef EPIC_CUDA_AVAILABLE

#include "scatter_gather.h"
#include <cuda.h>
#include <cstring>
#include <unistd.h> // for usleep
#include <cstdlib>
#include <thread>
#include <algorithm>
#include <benchmarks/ycsb_storage.h>
#include <benchmarks/tpcc_storage.h>  // required for the TPC-C explicit instantiations at end of file
#include <iostream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <util_log.h>
#include <chrono>
#include <type_traits>
#include <atomic>
#include <allocator.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <numeric>

#include <util_gpu_error_check.cuh>

#include "gpu_allocator.h"
#include "numa_affinity.h"
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#ifdef _OPENMP

static inline void set_affinity_to_node1(cpu_set_t* out_old_mask) {
    // Save old
    CPU_ZERO(out_old_mask);
    int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), out_old_mask);
    if (rc != 0) throw std::runtime_error(std::string("pthread_getaffinity_np failed: ") + std::strerror(rc));

    // Set new (node-1 physical cores via numa_affinity.h)
    cpu_set_t new_mask;
    CPU_ZERO(&new_mask);
    for (int cpu : epic::scatter_worker_cpus()) CPU_SET(cpu, &new_mask);

    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &new_mask);
    if (rc != 0) throw std::runtime_error(std::string("pthread_setaffinity_np failed: ") + std::strerror(rc));
}

static inline void restore_affinity(const cpu_set_t& old_mask) {
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_mask);
    if (rc != 0) throw std::runtime_error(std::string("restore pthread_setaffinity_np failed: ") + std::strerror(rc));
}
#endif

static inline void pin_omp_thread_to_node1_core(int tid)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    const auto& cpus = epic::scatter_worker_cpus();
    int cpu = cpus[tid % cpus.size()];
    CPU_SET(cpu, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

namespace epic {

    static_assert(std::is_trivially_copyable<ycsb::YcsbRecords>::value, "TRec must be POD");
    static_assert(std::is_trivially_copyable<ycsb::YcsbVersions>::value, "TVer must be POD");
    static_assert(std::is_trivially_copyable<ycsb::YcsbFieldRecords>::value, "Field TRec must be POD");
    static_assert(std::is_trivially_copyable<ycsb::YcsbFieldVersions>::value, "Field TVer must be POD");

    // ------- Timer helpers --------
    struct CpuTimer {
        const char* tag;
        std::chrono::high_resolution_clock::time_point t0;

        explicit CpuTimer(const char* t)
            : tag(t),
              t0(std::chrono::high_resolution_clock::now())
        {}

        ~CpuTimer() {
            using namespace std::chrono;
            auto us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
            Logger::GetInstance().Info("[SG] {}: {} us", tag, us);
        }
    };

    // GPU event timer (µs) on a given stream (defaults to 0)
    struct GpuTimer {
        const char* tag;
        cudaEvent_t start{}, stop{};
        cudaStream_t stream{};

        explicit GpuTimer(const char* t, cudaStream_t s = 0)
            : tag(t),
              stream(s)
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, stream);
        }

        ~GpuTimer() {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, start, stop);
            Logger::GetInstance().Info("[SG] {}: {} us", tag, (long long)(ms * 1000.0f));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    };

    static inline cudaStream_t as_cuda_stream(StreamHandle h) {
        return reinterpret_cast<cudaStream_t>(h);
    }

    static inline cudaEvent_t as_cuda_event(EventHandle e) {
        return reinterpret_cast<cudaEvent_t>(e);
    }

    static inline EventHandle as_event_handle(cudaEvent_t e) {
        return reinterpret_cast<EventHandle>(e);
    }

    static inline uint8_t classify_prev_slot_host(uint32_t v1, uint32_t v2, uint32_t epoch) {
        if (v1 == epoch)      return 1; // curr=slot1 => prev=slot2
        else if (v2 == epoch) return 0; // curr=slot2 => prev=slot1
        else if (v1 < v2)     return 1; // prev=slot2
        else                  return 0; // prev=slot1
    }

    // Device helpers
    template <typename TRec>
    __global__ void k_scatter_versions(const uint8_t* __restrict__ packed_vers,
                                       const uint32_t* __restrict__ grids,
                                       const uint8_t* __restrict__ prev_slots,
                                       TRec* __restrict__ dst_slots,
                                       uint32_t n)
    {
        using L = RecordLayout<TRec>;

        // We are intentionally implementing a 32-bit path only.
        static_assert((L::kVersBytes % sizeof(uint32_t)) == 0,
                      "kVersBytes must be divisible by 4 for 32-bit copy");
        static_assert((L::kValBytes % sizeof(uint32_t)) == 0,
                      "kValBytes must be divisible by 4 for 32-bit copy");

        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        const uint32_t g    = grids[i];
        const uint8_t  slot = prev_slots[i];  // expected 0 or 1

        const uint8_t* src = packed_vers + size_t(i) * L::kPackedBytes;
        uint8_t* dst       = reinterpret_cast<uint8_t*>(dst_slots + g);

        // -----------------------------
        // 1) Copy versions (packed ver1|ver2 -> record ver1, ver2 separately
        //    since they may not be adjacent in the hybrid record layout)
        // -----------------------------
        *reinterpret_cast<uint32_t*>(dst + L::kVer1Off) =
            *reinterpret_cast<const uint32_t*>(src + L::kPackedVer1Off);
        *reinterpret_cast<uint32_t*>(dst + L::kVer2Off) =
            *reinterpret_cast<const uint32_t*>(src + L::kPackedVer2Off);

        // -----------------------------
        // 2) Copy selected value
        //    source is packed value region
        //    destination is value1 or value2 based on prev slot
        // -----------------------------
        const uint8_t* src_val = src + L::kPackedValOff;
        uint8_t* dst_val       = dst + ((slot == 0) ? L::kVal1Off : L::kVal2Off);

        const uint32_t* s_val = reinterpret_cast<const uint32_t*>(src_val);
        uint32_t*       d_val = reinterpret_cast<uint32_t*>(dst_val);

    #pragma unroll
        for (size_t w = 0; w < (L::kValBytes / sizeof(uint32_t)); ++w) {
            d_val[w] = s_val[w];
        }
    }

    // Warp-cooperative variant mirroring k_pack_versions_by_grids_warp.
    // For H2D scatter: reads are from a packed contiguous source (coalesces
    // naturally within one record), writes go to the scattered destination slot.
    // With 32 lanes per record, adjacent lanes copy adjacent u32 words of the
    // value payload -> 128-byte coalesced reads and writes per iteration.
    // Lane 0 handles the two scalar version writes at kVer1Off / kVer2Off.
    template <typename TRec>
    __global__ void k_scatter_versions_warp(const uint8_t* __restrict__ packed_vers,
                                            const uint32_t* __restrict__ grids,
                                            const uint8_t* __restrict__ prev_slots,
                                            TRec* __restrict__ dst_slots,
                                            uint32_t n)
    {
        using L = RecordLayout<TRec>;

        static_assert((L::kVersBytes % sizeof(uint32_t)) == 0, "kVersBytes must be multiple of 4");
        static_assert((L::kValBytes  % sizeof(uint32_t)) == 0, "kValBytes must be multiple of 4");

        constexpr uint32_t kValWords = static_cast<uint32_t>(L::kValBytes / sizeof(uint32_t));

        const uint32_t lane             = threadIdx.x & 31u;
        const uint32_t warps_per_block  = blockDim.x >> 5;
        const uint32_t warp_in_block    = threadIdx.x >> 5;
        const uint32_t warp_in_grid     = blockIdx.x * warps_per_block + warp_in_block;
        const uint32_t warps_per_grid   = gridDim.x * warps_per_block;

        for (uint32_t i = warp_in_grid; i < n; i += warps_per_grid)
        {
            const uint32_t g    = grids[i];
            const uint8_t  slot = prev_slots[i];

            const uint8_t* src = packed_vers + size_t(i) * L::kPackedBytes;
            uint8_t*       dst = reinterpret_cast<uint8_t*>(dst_slots + g);

            // 1) version writes: lane 0 writes both version slots (non-adjacent
            // in the record layout, so two separate u32 scalar stores).
            if (lane == 0) {
                *reinterpret_cast<uint32_t*>(dst + L::kVer1Off) =
                    *reinterpret_cast<const uint32_t*>(src + L::kPackedVer1Off);
                *reinterpret_cast<uint32_t*>(dst + L::kVer2Off) =
                    *reinterpret_cast<const uint32_t*>(src + L::kPackedVer2Off);
            }

            // 2) value payload: 32 lanes cooperatively copy u32 words.
            // Adjacent lanes hit adjacent words -> each iteration coalesces
            // into a single 128-byte HBM read + 128-byte HBM write.
            const uint32_t* s_val = reinterpret_cast<const uint32_t*>(src + L::kPackedValOff);
            uint32_t* d_val       = reinterpret_cast<uint32_t*>(
                dst + ((slot == 0) ? L::kVal1Off : L::kVal2Off));

            #pragma unroll 4
            for (uint32_t w = lane; w < kValWords; w += 32u) {
                d_val[w] = s_val[w];
            }
        }
    }


    template <typename TRec>
    __global__ void k_pack_versions_by_grids(const TRec* __restrict__ src_slots,
                                         const uint32_t* __restrict__ grids,
                                         const uint8_t* __restrict__ curr_slots,
                                         uint8_t* __restrict__ packed_vers,
                                         uint32_t n)
    {
        using L = RecordLayout<TRec>;

        static_assert((L::kVersBytes % sizeof(uint32_t)) == 0, "kVersBytes must be multiple of 4");
        static_assert((L::kValBytes  % sizeof(uint32_t)) == 0, "kValBytes must be multiple of 4");

        // Grid-stride loop: allows launching fewer blocks for low-occupancy mode
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += blockDim.x * gridDim.x)
        {
            const uint32_t g    = grids[i];
            const uint8_t  slot = curr_slots[i]; // 0=>value1, 1=>value2

            const uint8_t* src = reinterpret_cast<const uint8_t*>(src_slots + g);
            // NEW packed layout: [u32 version | value]
            uint8_t* dst = packed_vers + size_t(i) * (sizeof(uint32_t) + L::kValBytes);

            // 1) selected version (one u32)
            const uint32_t* vsrc = reinterpret_cast<const uint32_t*>(
                src + ((slot == 0) ? L::kVer1Off : L::kVer2Off)
            );
            *reinterpret_cast<uint32_t*>(dst) = *vsrc;

            // 2) selected value
            const uint8_t* val_src = src + ((slot == 0) ? L::kVal1Off : L::kVal2Off);
            const uint32_t* s_val  = reinterpret_cast<const uint32_t*>(val_src);
            uint32_t* d_val        = reinterpret_cast<uint32_t*>(dst + sizeof(uint32_t));

#pragma unroll
            for (size_t w = 0; w < (L::kValBytes / sizeof(uint32_t)); ++w) {
                d_val[w] = s_val[w];
            }
        }
    }

    // Warp-cooperative variant for large records. One warp per dirty GRID
    // entry cooperatively copies the kValBytes value payload in coalesced
    // 128-byte transactions (32 lanes x 4 bytes). For 1 KB records this
    // reduces per-record HBM transactions from ~32 uncoalesced per thread
    // to ~8 coalesced per warp. Use only when kValBytes is large enough
    // that the cooperative copy isn't just wasting lanes -- see dispatch
    // threshold in wb_async_queue_pack().
    template <typename TRec>
    __global__ void k_pack_versions_by_grids_warp(const TRec* __restrict__ src_slots,
                                                   const uint32_t* __restrict__ grids,
                                                   const uint8_t* __restrict__ curr_slots,
                                                   uint8_t* __restrict__ packed_vers,
                                                   uint32_t n)
    {
        using L = RecordLayout<TRec>;

        static_assert((L::kVersBytes % sizeof(uint32_t)) == 0, "kVersBytes must be multiple of 4");
        static_assert((L::kValBytes  % sizeof(uint32_t)) == 0, "kValBytes must be multiple of 4");

        constexpr uint32_t kPackedBytes = static_cast<uint32_t>(sizeof(uint32_t) + L::kValBytes);
        constexpr uint32_t kValWords    = static_cast<uint32_t>(L::kValBytes / sizeof(uint32_t));

        const uint32_t lane             = threadIdx.x & 31u;
        const uint32_t warps_per_block  = blockDim.x >> 5;
        const uint32_t warp_in_block    = threadIdx.x >> 5;
        const uint32_t warp_in_grid     = blockIdx.x * warps_per_block + warp_in_block;
        const uint32_t warps_per_grid   = gridDim.x * warps_per_block;

        // Grid-stride over warps: warp i handles record i, n warps total.
        for (uint32_t i = warp_in_grid; i < n; i += warps_per_grid)
        {
            const uint32_t g    = grids[i];
            const uint8_t  slot = curr_slots[i];

            const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_slots + g);
            uint8_t* dst = packed_vers + size_t(i) * kPackedBytes;

            // 1) version header: lane 0 only (single u32 scalar write).
            if (lane == 0) {
                const uint32_t* vsrc = reinterpret_cast<const uint32_t*>(
                    src_base + ((slot == 0) ? L::kVer1Off : L::kVer2Off)
                );
                *reinterpret_cast<uint32_t*>(dst) = *vsrc;
            }

            // 2) value payload: 32 lanes cooperatively copy u32 words.
            // Adjacent lanes hit adjacent words => each iteration coalesces
            // into a single 128-byte HBM transaction.
            const uint32_t* vs = reinterpret_cast<const uint32_t*>(
                src_base + ((slot == 0) ? L::kVal1Off : L::kVal2Off)
            );
            uint32_t* vd = reinterpret_cast<uint32_t*>(dst + sizeof(uint32_t));

            #pragma unroll 4
            for (uint32_t w = lane; w < kValWords; w += 32u) {
                vd[w] = vs[w];
            }
        }
    }



    template <class TRec>
    ScatterGather<TRec>::ScatterGather() {}

    template <class TRec>
    ScatterGather<TRec>::~ScatterGather()
    {
        reset();
    }

    template <class TRec>
    void ScatterGather<TRec>::init(const Options& opt, StreamHandle stream)
    {
        options_    = opt;
        own_stream_ = opt.own_stream || (stream == nullptr);  // own the stream if caller asked or did not provide one
        chunk_mode_ = opt.chunk_mode;
        async_writeback_ = opt.async_writeback;
        chunk_bytes = opt.chunk_bytes ? opt.chunk_bytes : (32ull << 20);  // default 32 MiB


        // Stream 0 or provided
        if (!own_stream_) {
            // Use provided stream
            stream_ = stream;
        } else {
            // Create our own if needed
            if (stream_ == nullptr) {
                cudaStream_t s{};
                gpu_err_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
                stream_ = reinterpret_cast<StreamHandle>(s);
            }
        }

        // Stream 1: used for chunked mode (double-buffering)
        if (chunk_mode_) {

            if (copy_stream_ == nullptr) {
                cudaStream_t cs{};
                gpu_err_check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
                copy_stream_ = reinterpret_cast<StreamHandle>(cs);
            }
            if (compute_stream_ == nullptr) {
                cudaStream_t ks{};
                gpu_err_check(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
                compute_stream_ = reinterpret_cast<StreamHandle>(ks);
            }

            // create events
            for (int b = 0; b < 2; ++b) {
                if (!h2d_done_[b]) {
                    cudaEvent_t ev{};
                    gpu_err_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    h2d_done_[b] = as_event_handle(ev);
                }
                if (!d2h_done_[b]) {
                    cudaEvent_t ev{};
                    gpu_err_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    d2h_done_[b] = as_event_handle(ev);
                }
                if (!compute_done_[b]) {
                    cudaEvent_t ev{};
                    gpu_err_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    compute_done_[b] = as_event_handle(ev);
                }
            }

            tx_in_flight_[0] = tx_in_flight_[1] = false;
            wb_in_flight_[0] = wb_in_flight_[1] = false;
        }
    }

    template <class TRec>
    void ScatterGather<TRec>::reset()
    {
        if (d_grids_persistent_) {
            cudaFree(d_grids_persistent_);
            d_grids_persistent_ = nullptr;
        }

        cap_grids_elems_ = 0;

        reset_versions_buffers_();
        if (chunk_mode_) {
            reset_chunk_buffers_();
        }

        if (own_stream_ && stream_) {
            cudaStreamDestroy(as_cuda_stream(stream_));
            stream_ = nullptr;
        }
    }

    template <class TRec>
    void ScatterGather<TRec>::reset_chunk_buffers_()
    {
        for (int b = 0; b < 2; ++b) {
            if (d_grids_buf_[b]) {
                cudaFree(d_grids_buf_[b]);
                d_grids_buf_[b] = nullptr;
            }

            if (h_grids_buf_[b]) {
                cudaFreeHost(h_grids_buf_[b]);
                h_grids_buf_[b] = nullptr;
            }

            if (h2d_done_[b]) {
                cudaEventDestroy(as_cuda_event(h2d_done_[b]));
                h2d_done_[b] = nullptr;
            }

            if (d2h_done_[b]) {
                cudaEventDestroy(as_cuda_event(d2h_done_[b]));
                d2h_done_[b] = nullptr;
            }
            if (compute_done_[b]) {
                cudaEventDestroy(as_cuda_event(compute_done_[b]));
                compute_done_[b] = nullptr;
            }

        }

        tx_in_flight_[0] = tx_in_flight_[1] = false;
        wb_in_flight_[0] = wb_in_flight_[1] = false;

        chunk_cap_elems_ = 0;


        if (copy_stream_) {
            cudaStreamDestroy(as_cuda_stream(copy_stream_));
            copy_stream_ = nullptr;
        }
        if (compute_stream_) {
            cudaStreamDestroy(as_cuda_stream(compute_stream_));
            compute_stream_ = nullptr;
        }
    }

    template <class TRec>
    void ScatterGather<TRec>::reset_versions_buffers_()
    {
        // Non-chunk versions staging buffers (if present)
        if (h_packed_vers_) {
            cudaFreeHost(h_packed_vers_);
            h_packed_vers_ = nullptr;
        }

        if (d_packed_vers_) {
            cudaFree(d_packed_vers_);
            d_packed_vers_ = nullptr;
        }

        if (h_prev_slots_) {
            cudaFreeHost(h_prev_slots_);
            h_prev_slots_ = nullptr;
        }

        cap_vers_elems_ = 0;
        cap_prev_slots_elems_ = 0;

        // Chunked versions double-buffers
        for (int b = 0; b < 2; ++b) {
            if (h_packed_vers_buf_[b]) {
                cudaFreeHost(h_packed_vers_buf_[b]);
                h_packed_vers_buf_[b] = nullptr;
            }

            if (d_packed_vers_buf_[b]) {
                cudaFree(d_packed_vers_buf_[b]);
                d_packed_vers_buf_[b] = nullptr;
            }

            if (h_prev_slots_buf_[b]) {
                cudaFreeHost(h_prev_slots_buf_[b]);
                h_prev_slots_buf_[b] = nullptr;
            }

            if (d_prev_slots_buf_[b]) {
                cudaFree(d_prev_slots_buf_[b]);
                d_prev_slots_buf_[b] = nullptr;
            }

            if (h_curr_slots_buf_[b]) {
                cudaFreeHost(h_curr_slots_buf_[b]);
                h_curr_slots_buf_[b] = nullptr;
             }

             if (d_curr_slots_buf_[b]) {
                cudaFree(d_curr_slots_buf_[b]);
                d_curr_slots_buf_[b] = nullptr;
             }
        }

        chunk_ver_cap_elems_ = 0;
        prev_slots_cap_elems_ = 0;
        curr_slots_cap_elems_ = 0;
    }





    template <class TRec>
    void ScatterGather<TRec>::ensure_versions_capacity(size_t n)
    {
        ensure_versions_capacity_bytes_(n);
    }

    template <class TRec>
    void ScatterGather<TRec>::ensure_versions_capacity_bytes_(size_t n)
    {
        if (n <= cap_vers_elems_) return;

        size_t new_cap = cap_vers_elems_ ? (cap_vers_elems_ + cap_vers_elems_ / 2) : n;
        if (new_cap < n) new_cap = n;

        const size_t total_bytes = new_cap * kPackedVersionsBytes_;

        Logger::GetInstance().Info(
            "[SG] ensure_versions_capacity: total allocated bytes = {}",
            formatSizeBytes(total_bytes));

        // Recreate pinned host packed versions buffer
        if (h_packed_vers_) {
            cudaFreeHost(h_packed_vers_);
            h_packed_vers_ = nullptr;
        }
        gpu_err_check(cudaHostAlloc(&h_packed_vers_, total_bytes, cudaHostAllocDefault));

        // Recreate device packed versions buffer
        if (d_packed_vers_) {
            cudaFree(d_packed_vers_);
            d_packed_vers_ = nullptr;
        }
        gpu_err_check(cudaMalloc(&d_packed_vers_, total_bytes));

        cap_vers_elems_ = new_cap;
        Logger::GetInstance().Info(
            "[SG] ensure_versions_capacity: cap_vers_elems_ -> {} (bytes/elem={})",
            cap_vers_elems_, kPackedVersionsBytes_);

        if (n > cap_prev_slots_elems_) {
            if (h_prev_slots_) cudaFreeHost(h_prev_slots_);
            gpu_err_check(cudaHostAlloc(&h_prev_slots_, n * sizeof(uint8_t), cudaHostAllocDefault));
            cap_prev_slots_elems_ = n;
        }
    }


    template <class TRec>
    void ScatterGather<TRec>::ensure_grids_capacity_(size_t n)
    {
        if (n <= cap_grids_elems_) return;

        // grow-only policy (similar to packed buffers, but independent)
        size_t new_cap = cap_grids_elems_ ? (cap_grids_elems_ + cap_grids_elems_ / 2) : n;
        if (new_cap < n) new_cap = n;

        if (d_grids_persistent_) {
            cudaFree(d_grids_persistent_);
            d_grids_persistent_ = nullptr;
        }

        gpu_err_check(cudaMalloc(&d_grids_persistent_, new_cap * sizeof(uint32_t)));

        cap_grids_elems_ = new_cap;
        Logger::GetInstance().Info("[SG] ensure_grids_capacity: cap_grids_elems_ -> {}", cap_grids_elems_);
    }

    template <class TRec>
    void ScatterGather<TRec>::ensure_chunk_buffers_(size_t chunk_elems)
    {
        if (chunk_elems <= chunk_cap_elems_) return;

        // free old
        for (int b = 0; b < 2; ++b) {
            if (d_grids_buf_[b]) {
                cudaFree(d_grids_buf_[b]);
                d_grids_buf_[b] = nullptr;
            }
            if (h_grids_buf_[b])
            {
                cudaFreeHost(h_grids_buf_[b]);
                h_grids_buf_[b] = nullptr;
            }
        }

        // allocate A/B pinned host + device buffers for grids
        size_t grid_bytes = chunk_elems * sizeof(uint32_t);

        for (int b = 0; b < 2; ++b) {
            gpu_err_check(cudaMalloc(&d_grids_buf_[b],         grid_bytes));
            gpu_err_check(cudaHostAlloc(&h_grids_buf_[b],      grid_bytes, cudaHostAllocDefault));
        }

        chunk_cap_elems_ = chunk_elems;

        tx_in_flight_[0] = tx_in_flight_[1] = false;
        wb_in_flight_[0] = wb_in_flight_[1] = false;

        Logger::GetInstance().Info("[SG] ensure_chunk_buffers: chunk_cap_elems_ -> {}", chunk_cap_elems_);
    }

    template <class TRec>
    void ScatterGather<TRec>::ensure_versions_chunk_buffers_(size_t chunk_elems)
    {
        if (chunk_elems <= chunk_ver_cap_elems_) return;

        // Free old version chunk buffers
        for (int b = 0; b < 2; ++b) {
            if (h_packed_vers_buf_[b]) {
                cudaFreeHost(h_packed_vers_buf_[b]);
                h_packed_vers_buf_[b] = nullptr;
            }
            if (d_packed_vers_buf_[b]) {
                cudaFree(d_packed_vers_buf_[b]);
                d_packed_vers_buf_[b] = nullptr;
            }
            if (h_prev_slots_buf_[b]) {
                cudaFreeHost(h_prev_slots_buf_[b]);
                h_prev_slots_buf_[b] = nullptr;
            }
            if (d_prev_slots_buf_[b]) {
                cudaFree(d_prev_slots_buf_[b]);
                d_prev_slots_buf_[b] = nullptr;
            }
            if (h_curr_slots_buf_[b]) {
                cudaFreeHost(h_curr_slots_buf_[b]);
                h_curr_slots_buf_[b] = nullptr;
            }
             if (d_curr_slots_buf_[b]) {
                cudaFree(d_curr_slots_buf_[b]);
                d_curr_slots_buf_[b] = nullptr;
            }
        }

        // Allocate new A/B buffers sized for packed versions payload
        const size_t vers_bytes = chunk_elems * kPackedVersionsBytes_;

        for (int b = 0; b < 2; ++b) {
            gpu_err_check(cudaHostAlloc(&h_packed_vers_buf_[b], vers_bytes, cudaHostAllocDefault));
            gpu_err_check(cudaMalloc(&d_packed_vers_buf_[b], vers_bytes));
            gpu_err_check(cudaHostAlloc(&h_prev_slots_buf_[b], chunk_elems * sizeof(uint8_t), cudaHostAllocDefault));
            gpu_err_check(cudaMalloc(&d_prev_slots_buf_[b], chunk_elems * sizeof(uint8_t)));
            gpu_err_check(cudaHostAlloc(&h_curr_slots_buf_[b], chunk_elems * sizeof(uint8_t), cudaHostAllocDefault));
            gpu_err_check(cudaMalloc(&d_curr_slots_buf_[b], chunk_elems * sizeof(uint8_t)));
        }

        chunk_ver_cap_elems_ = chunk_elems;

        Logger::GetInstance().Info(
            "[SG] ensure_versions_chunk_buffers: chunk_ver_cap_elems_ -> {} (bytes/elem={}, total_per_buf={})",
            chunk_ver_cap_elems_, kPackedVersionsBytes_, formatSizeBytes(vers_bytes));
    }


    template <class TRec>
    void ScatterGather<TRec>::gather_host_versions_(const TRec* CPU_recs,
                                                const uint32_t* h_crids,
                                                void* out_vers,
                                                uint8_t* out_prev_slots,
                                                size_t n,
                                                uint32_t epoch)
    {
        auto* out = static_cast<uint8_t*>(out_vers);
        if (n == 0) return;

        constexpr size_t target_chunk_bytes = 2ull << 20; // ~2 MiB target work chunk
        constexpr size_t bytes_per_elem = kPackedVersionsBytes_;
        const size_t chunk_elems = std::max<size_t>(1, target_chunk_bytes / bytes_per_elem);

        // Count bad CRIDs leaking into the gather. The host-side
        // gather has no num_units context to bounds-check against, but the
        // sentinel value 0xffffffffu is a universal "invalid CRID" marker
        // that should never reach this point. If it does, skipping is safe
        // (the slot's pre-existing GPU content is then untouched by the
        // gather; subsequent scatter still ships zeros, which is harmless
        // because the matching sentinel guard in gpuWriteToTableCoop skips
        // the slot's executor write too).
        std::atomic<size_t> n_sentinel{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t base = 0; base < n; base += chunk_elems) {
            const size_t end = std::min(n, base + chunk_elems);

            for (size_t i = base; i < end; ++i) {
                const uint32_t crid = h_crids[i];
                if (crid == 0xffffffffu) {
                    n_sentinel.fetch_add(1, std::memory_order_relaxed);
                    std::memset(out + i * kPackedVersionsBytes_, 0, kPackedVersionsBytes_);
                    out_prev_slots[i] = 0;
                    continue;
                }

                const uint8_t* rec = reinterpret_cast<const uint8_t*>(CPU_recs)
                                   + size_t(crid) * sizeof(TRec);

                const uint32_t v1 = *reinterpret_cast<const uint32_t*>(rec + L::kVer1Off);
                const uint32_t v2 = *reinterpret_cast<const uint32_t*>(rec + L::kVer2Off);
                const uint8_t slot = classify_prev_slot_host(v1, v2, epoch);
                out_prev_slots[i] = slot;

                uint8_t* dst = out + i * kPackedVersionsBytes_;

                // Copy both versions into packed buffer (separately, since they
                // may not be adjacent in the hybrid record layout)
                std::memcpy(dst + L::kPackedVer1Off,
                            rec + L::kVer1Off,
                            sizeof(uint32_t));
                std::memcpy(dst + L::kPackedVer2Off,
                            rec + L::kVer2Off,
                            sizeof(uint32_t));

                // Copy the selected value
                if (slot == 0) {
                    std::memcpy(dst + L::kPackedValOff,
                                rec + L::kVal1Off,
                                L::kValBytes);
                } else {
                    std::memcpy(dst + L::kPackedValOff,
                                rec + L::kVal2Off,
                                L::kValBytes);
                }
            }
        }
        if (n_sentinel.load() > 0) {
            Logger::GetInstance().Info("gather_host_versions_: skipped {} sentinel CRID(s) of {} (TRec sz={})",
                                       n_sentinel.load(), n, sizeof(TRec));
        }
    }


    template <class TRec>
    void ScatterGather<TRec>::gather_host_versions_range_(const TRec* CPU_recs,
                                                      const uint32_t* h_crids,
                                                      size_t base,
                                                      size_t n,
                                                      uint32_t epoch,
                                                      void* out_vers,
                                                      uint8_t* out_prev_slots)
    {
        auto* out = static_cast<uint8_t*>(out_vers);
        if (n == 0) return;

        // Same sentinel defense as gather_host_versions_; see comment there.
        std::atomic<size_t> n_sentinel{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            const size_t idx = base + i;
            const uint32_t crid = h_crids[idx];
            if (crid == 0xffffffffu) {
                n_sentinel.fetch_add(1, std::memory_order_relaxed);
                std::memset(out + i * kPackedVersionsBytes_, 0, kPackedVersionsBytes_);
                out_prev_slots[i] = 0;
                continue;
            }

            const uint8_t* rec = reinterpret_cast<const uint8_t*>(CPU_recs)
                               + size_t(crid) * sizeof(TRec);

            const uint32_t v1 = *reinterpret_cast<const uint32_t*>(rec + L::kVer1Off);
            const uint32_t v2 = *reinterpret_cast<const uint32_t*>(rec + L::kVer2Off);
            const uint8_t slot = classify_prev_slot_host(v1, v2, epoch);
            out_prev_slots[i] = slot;

            uint8_t* dst = out + i * kPackedVersionsBytes_;


            // Copy both versions separately (safe for both original and hybrid layout)
            std::memcpy(dst + L::kPackedVer1Off,
                        rec + L::kVer1Off,
                        sizeof(uint32_t));
            std::memcpy(dst + L::kPackedVer2Off,
                        rec + L::kVer2Off,
                        sizeof(uint32_t));

            // Copy the selected value
            if (slot == 0) {
                std::memcpy(dst + L::kPackedValOff,
                            rec + L::kVal1Off,
                            L::kValBytes);
            } else {
                std::memcpy(dst + L::kPackedValOff,
                            rec + L::kVal2Off,
                            L::kValBytes);
            }
        }
        if (n_sentinel.load() > 0) {
            Logger::GetInstance().Info("gather_host_versions_range_: skipped {} sentinel CRID(s) of {} at base={}",
                                       n_sentinel.load(), n, base);
        }
    }


    template <class TRec>
void ScatterGather<TRec>::scatter_device_versions_(TRec* GPU_recs,
                                                   const uint32_t* h_grids,
                                                   const uint8_t* h_prev_slots,
                                                   const void* d_packed_vers,
                                                   size_t n)
    {
        if (n == 0) return;

        ensure_grids_capacity_(n);
        auto s = as_cuda_stream(stream_);

        // Reuse persistent grids device buffer
        gpu_err_check(cudaMemcpyAsync(d_grids_persistent_,
                                      h_grids,
                                      n * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice,
                                      s));

        // Non-chunk path reuses d_prev_slots_buf_[0] as its persistent device
        // slots buffer; grow it to n elements when needed.
        if (d_prev_slots_buf_[0] == nullptr || prev_slots_cap_elems_ < n) {
            if (d_prev_slots_buf_[0]) {
                cudaFree(d_prev_slots_buf_[0]);
                d_prev_slots_buf_[0] = nullptr;
            }
            gpu_err_check(cudaMalloc(&d_prev_slots_buf_[0], n * sizeof(uint8_t)));
            prev_slots_cap_elems_ = n;
        }

        gpu_err_check(cudaMemcpyAsync(d_prev_slots_buf_[0],
                                      h_prev_slots,
                                      n * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice,
                                      s));

        const int BLK = 256;
        using L = RecordLayout<TRec>;
        // Same crossover threshold as the pack-warp kernel: below 32 B of value
        // payload (8 u32 words), per-thread beats warp-coop because too many
        // lanes idle on every iteration. At/above 32 B, warp-coop wins by turning
        // the scattered destination writes into coalesced 128-B HBM transactions.
        constexpr bool kUseWarpCoop = (L::kValBytes >= 32);

        if (kUseWarpCoop) {
            const int warps_per_block = BLK / 32;
            const int GRD = static_cast<int>((n + warps_per_block - 1) / warps_per_block);
            k_scatter_versions_warp<TRec><<<GRD, BLK, 0, s>>>(
                reinterpret_cast<const uint8_t*>(d_packed_vers),
                d_grids_persistent_,
                d_prev_slots_buf_[0],
                GPU_recs,
                static_cast<uint32_t>(n));
        } else {
            const int GRD = static_cast<int>((n + BLK - 1) / BLK);
            k_scatter_versions<TRec><<<GRD, BLK, 0, s>>>(
                reinterpret_cast<const uint8_t*>(d_packed_vers),
                d_grids_persistent_,
                d_prev_slots_buf_[0],
                GPU_recs,
                static_cast<uint32_t>(n));
        }

        gpu_err_check(cudaPeekAtLastError());
    }


template <class TRec>
void ScatterGather<TRec>::transfer_versions(const uint32_t* h_crids,
                                            const uint32_t* h_grids,
                                            uint32_t epoch,
                                            size_t n,
                                            TRec* CPU_recs,
                                            TRec* GPU_recs)
{
    if (n == 0) {
        return;
    }

    admit_h2d_payload_bytes_total_ += static_cast<uint64_t>(n) * kPackedVersionsBytes_;

    CpuTimer _t_total("SG.transfer_versions.total");
        // We will start by printing the cpu's that OMP will be running with to make sure its on node 0
            {
                std::string cpus = sched_getcpu() >= 0 ? std::to_string(sched_getcpu()) : "unknown";
                Logger::GetInstance().Info("[SG] transfer_versions: running on CPU {}", cpus);
            }

    // ------------------ non-chunk ------------------
    if (!chunk_mode_) {
        ensure_versions_capacity(n);

        // 1) gather packed [ver1|ver2|selected_value] on host pinned buffer
        gather_host_versions_(CPU_recs, h_crids, h_packed_vers_, h_prev_slots_, n, epoch);

        // 2) H2D packed versions
        auto s = as_cuda_stream(stream_);
        gpu_err_check(cudaMemcpyAsync(d_packed_vers_,
                                      h_packed_vers_,
                                      n * kPackedVersionsBytes_,
                                      cudaMemcpyHostToDevice,
                                      s));
        // sync
        gpu_err_check(cudaStreamSynchronize(s));
        gpu_err_check(cudaPeekAtLastError());

        // 3) scatter into GPU slots using grids + prev_slots
        scatter_device_versions_(GPU_recs, h_grids, h_prev_slots_, d_packed_vers_, n);
        gpu_err_check(cudaStreamSynchronize(s));
        gpu_err_check(cudaPeekAtLastError());

        return;
    }

    // ------------------ chunk mode ------------------
    const size_t bytes_per_elem = kPackedVersionsBytes_;
    // Adaptive chunk_bytes per call: aim for 1 chunk up to MAX_CHUNK_CAP. Per-chunk
    // overhead under 8-way parallel-sections + nested OMP measured at ~414 us/chunk
    // (vs ~17 us/chunk with no contention), so n_chunks=1 minimizes wall-clock.
    // YCSB stays at 4 MiB floor since its gathers are small. TPCC OL grows the
    // chunk to fit its 3-25 MB gather across W=16..128 in one chunk. Cap at 32 MiB
    // because 64 MiB pinned-buffer alloc segfaulted in tests.
    constexpr size_t MAX_CHUNK_CAP   = 32ull << 20;  // 32 MiB
    constexpr size_t MIN_CHUNK_FLOOR = 4ull  << 20;  // 4 MiB (current default)
    const size_t total_payload_bytes = n * bytes_per_elem;
    const size_t adaptive_chunk_bytes =
        std::min(MAX_CHUNK_CAP, std::max(MIN_CHUNK_FLOOR, total_payload_bytes));
    // chunk_bytes member acts as a hard floor (env override or caller-provided opt).
    // If user explicitly set chunk_bytes > MIN_CHUNK_FLOOR (e.g. via env), respect it
    // as the floor instead. This preserves the env-var sweep tool.
    const size_t effective_chunk_bytes =
        std::min(MAX_CHUNK_CAP, std::max(chunk_bytes, adaptive_chunk_bytes));
    const size_t chunk_elems = std::max<size_t>(1, effective_chunk_bytes / bytes_per_elem);

    ensure_versions_chunk_buffers_(chunk_elems);
    ensure_chunk_buffers_(chunk_elems);
    size_t base = 0;
    int buf_idx = 0;
    int chunk_idx = 0;
    auto sg_t0 = std::chrono::high_resolution_clock::now();

    cudaStream_t cs = as_cuda_stream(copy_stream_);
    cudaStream_t ks = as_cuda_stream(compute_stream_);

    while (base < n) {
        const size_t m = std::min(chunk_elems, n - base);

        // Safe reuse of this A/B slot
        if (tx_in_flight_[buf_idx]) {
            gpu_err_check(cudaEventSynchronize(as_cuda_event(compute_done_[buf_idx])));
        }

        auto t_gather_start = std::chrono::high_resolution_clock::now();
        // Host gather for this chunk (packed versions)
        gather_host_versions_range_(CPU_recs,
                                    h_crids,
                                    base,
                                    m,
                                    epoch,
                                    h_packed_vers_buf_[buf_idx],
                                    h_prev_slots_buf_[buf_idx]);

        auto t_gather_end = std::chrono::high_resolution_clock::now();
        auto gather_us = std::chrono::duration_cast<std::chrono::microseconds>(t_gather_end - t_gather_start).count();
        auto gather_offset = std::chrono::duration_cast<std::chrono::microseconds>(t_gather_start - sg_t0).count();
        Logger::GetInstance().Info("[SG][chunk] c{} gather: {}us (at +{}us) n={}", chunk_idx, gather_us, gather_offset, m);

        // Copy packed versions H2D (copy stream)
        gpu_err_check(cudaMemcpyAsync(d_packed_vers_buf_[buf_idx],
                                      h_packed_vers_buf_[buf_idx],
                                      m * kPackedVersionsBytes_,
                                      cudaMemcpyHostToDevice,
                                      cs));
        gpu_err_check(cudaPeekAtLastError());

        // Copy grids chunk into pinned host A/B, then H2D
        std::memcpy(h_grids_buf_[buf_idx], h_grids + base, m * sizeof(uint32_t));
        gpu_err_check(cudaMemcpyAsync(d_grids_buf_[buf_idx],
                                      h_grids_buf_[buf_idx],
                                      m * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice,
                                      cs));
        gpu_err_check(cudaPeekAtLastError());

        // Copy prev_slots chunk into pinned host A/B, then H2D
        gpu_err_check(cudaMemcpyAsync(d_prev_slots_buf_[buf_idx],
                                      h_prev_slots_buf_[buf_idx],
                                      m * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice,
                                      cs));
        gpu_err_check(cudaPeekAtLastError());

        // Mark copy phase done for this buffer
        gpu_err_check(cudaEventRecord(as_cuda_event(h2d_done_[buf_idx]), cs));
        gpu_err_check(cudaPeekAtLastError());

        // Compute waits for copy
        gpu_err_check(cudaStreamWaitEvent(ks, as_cuda_event(h2d_done_[buf_idx]), 0));
        gpu_err_check(cudaPeekAtLastError());

        const int BLK = 256;
        using L = RecordLayout<TRec>;
        constexpr bool kUseWarpCoop = (L::kValBytes >= 32);

        if (kUseWarpCoop) {
            const int warps_per_block = BLK / 32;
            const int GRD = static_cast<int>((m + warps_per_block - 1) / warps_per_block);
            k_scatter_versions_warp<TRec><<<GRD, BLK, 0, ks>>>(
                reinterpret_cast<const uint8_t*>(d_packed_vers_buf_[buf_idx]),
                d_grids_buf_[buf_idx],
                d_prev_slots_buf_[buf_idx],
                GPU_recs,
                static_cast<uint32_t>(m));
        } else {
            const int GRD = static_cast<int>((m + BLK - 1) / BLK);
            k_scatter_versions<TRec><<<GRD, BLK, 0, ks>>>(
                reinterpret_cast<const uint8_t*>(d_packed_vers_buf_[buf_idx]),
                d_grids_buf_[buf_idx],
                d_prev_slots_buf_[buf_idx],
                GPU_recs,
                static_cast<uint32_t>(m));
        }
        gpu_err_check(cudaPeekAtLastError());

        // Mark compute done for this buffer
        gpu_err_check(cudaEventRecord(as_cuda_event(compute_done_[buf_idx]), ks));
        tx_in_flight_[buf_idx] = true;

        base += m;
        buf_idx ^= 1;
        chunk_idx++;
    }

    // Final sync
    auto t_before_cs = std::chrono::high_resolution_clock::now();
    gpu_err_check(cudaStreamSynchronize(cs));
    auto t_after_cs = std::chrono::high_resolution_clock::now();
    gpu_err_check(cudaStreamSynchronize(ks));
    auto t_after_ks = std::chrono::high_resolution_clock::now();
    auto cs_wait = std::chrono::duration_cast<std::chrono::microseconds>(t_after_cs - t_before_cs).count();
    auto ks_wait = std::chrono::duration_cast<std::chrono::microseconds>(t_after_ks - t_after_cs).count();
    auto total = std::chrono::duration_cast<std::chrono::microseconds>(t_after_ks - sg_t0).count();
    Logger::GetInstance().Info("[SG][chunk] final: h2d_wait={}us gpu_scatter_wait={}us (at +{}us) total={}us chunks={}",
        cs_wait, ks_wait,
        std::chrono::duration_cast<std::chrono::microseconds>(t_before_cs - sg_t0).count(),
        total, chunk_idx);
}




    template <class TRec>
    void ScatterGather<TRec>::sync()
    {
        if (!chunk_mode_) {
            if (stream_) gpu_err_check(cudaStreamSynchronize(as_cuda_stream(stream_)));
            return;
        }

        if (copy_stream_)   gpu_err_check(cudaStreamSynchronize(as_cuda_stream(copy_stream_)));
        if (compute_stream_) gpu_err_check(cudaStreamSynchronize(as_cuda_stream(compute_stream_)));
    }


    template <class TRec>
    void ScatterGather<TRec>::scatter_host_versions_range_from_packed_(TRec* CPU_recs,
        const uint32_t* h_crids, const uint8_t* h_curr_slots, size_t base, size_t n, const void* packed_vers)
    {
        using L = RecordLayout<TRec>;
        if (n == 0) return;
        const uint8_t* src = static_cast<const uint8_t*>(packed_vers);
        std::atomic<size_t> n_sentinel_wb{0};
#ifdef _OPENMP
        constexpr int kScatterThreads = 12;
        #pragma omp parallel for schedule(static) num_threads(kScatterThreads)
        #endif
        for (size_t i = 0; i < n; ++i)
        {
            const size_t   idx  = base + i;
            const uint32_t crid = h_crids[idx];
            const uint8_t  slot = h_curr_slots[idx]; // 0=>slot1, 1=>slot2

            if (crid == 0xffffffffu) {
                n_sentinel_wb.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint8_t* dst_rec = reinterpret_cast<uint8_t*>(CPU_recs) + size_t(crid) * sizeof(TRec);
            const uint8_t* src_i = src + i * kPackedWritebackBytes_;

            // 2) write selected value to selected slot
            std::memcpy(dst_rec + ((slot == 0) ? L::kVal1Off : L::kVal2Off),
                        src_i + sizeof(uint32_t),
                        L::kValBytes);

            // Ensure value is fully written before publishing the version,
            // so concurrent readers never see a new version with stale value data.
            std::atomic_thread_fence(std::memory_order_release);

            // 1) write selected version to selected slot
            *reinterpret_cast<uint32_t*>(dst_rec + ((slot == 0) ? L::kVer1Off : L::kVer2Off))
                = *reinterpret_cast<const uint32_t*>(src_i);




        }
        if (n_sentinel_wb.load() > 0) {
            Logger::GetInstance().Info("scatter (sync wb): skipped {} sentinel CRID(s) of {}",
                                       n_sentinel_wb.load(), n);
        }

    }

    // All async does here is pin to another core to avoid interfering with host gather/scatter threads
    template <class TRec>
    void ScatterGather<TRec>::scatter_host_versions_range_from_packed_async_(TRec* CPU_recs,
                                                                       const uint32_t* h_crids,
                                                                       const uint8_t*  h_curr_slots,
                                                                       size_t base,
                                                                       size_t n,
                                                                       const void* packed_vers)
    {
        using L = RecordLayout<TRec>;
        if (n == 0) return;

        const uint8_t* src = static_cast<const uint8_t*>(packed_vers);

#ifdef _OPENMP
        // Node-1 physical cores via numa_affinity.h.
        //
        // Pinning policy (default): mask covers only the 12 node-1 physical
        // cores (kCpusPhys, e.g. CPUs 12..23 on the reference box). Total CPU
        // footprint of the process is 24 node-0 (main + index OMP) + 12 node-1
        // (scatter OMP) = 36 logical CPUs.
        //
        // We restrict the scatter master/workers to physical cores, not the
        // HT siblings. Including the siblings gives +2-3% on an idle box but is
        // fragile under co-tenancy (CFS time-slices the scatter threads against
        // a co-tenant sharing the logical CPU, ballooning flush_sync_prev
        // ~524 us -> 6.9 ms, -33% end-to-end), so physical cores are the stable
        // ship default.
        const auto& kCpusPhys = epic::scatter_worker_cpus();
        const int kScatterThreads = static_cast<int>(kCpusPhys.size());
        omp_set_dynamic(0);

        const auto& kCpusMaster = kCpusPhys;
        cpu_set_t master_mask;
        CPU_ZERO(&master_mask);
        for (int c : kCpusMaster) CPU_SET(c, &master_mask);
        int mrc = pthread_setaffinity_np(pthread_self(), sizeof(master_mask), &master_mask);
        if (mrc != 0) {
            throw std::runtime_error(
                std::string("scatter master pthread_setaffinity_np failed: ")
                + std::strerror(mrc));
        }

        std::atomic<size_t> n_sentinel_wb{0};
#pragma omp parallel num_threads(kScatterThreads)
    {

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const size_t   idx  = base + i;
            const uint32_t crid = h_crids[idx];
            const uint8_t  slot = h_curr_slots[idx]; // 0=>slot1, 1=>slot2

            if (crid == 0xffffffffu) {
                n_sentinel_wb.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint8_t* dst_rec = reinterpret_cast<uint8_t*>(CPU_recs) + size_t(crid) * sizeof(TRec);
            const uint8_t* src_i = src + i * kPackedWritebackBytes_;

            // 1) write selected value to selected slot (must happen before version)
            std::memcpy(dst_rec + ((slot == 0) ? L::kVal1Off : L::kVal2Off),
                        src_i + sizeof(uint32_t),
                        L::kValBytes);

            std::atomic_thread_fence(std::memory_order_release);

            // 2) publish version to selected slot
            *reinterpret_cast<uint32_t*>(dst_rec + ((slot == 0) ? L::kVer1Off : L::kVer2Off))
                = *reinterpret_cast<const uint32_t*>(src_i);
        }
    }
        if (n_sentinel_wb.load() > 0) {
            Logger::GetInstance().Info("scatter (async wb): skipped {} sentinel CRID(s) of {}",
                                       n_sentinel_wb.load(), n);
        }
#else
    for (size_t i = 0; i < n; ++i) {
        const size_t   idx  = base + i;
        const uint32_t crid = h_crids[idx];
        const uint8_t  slot = h_curr_slots[idx];

        if (crid == 0xffffffffu) continue;
        uint8_t* dst_rec = reinterpret_cast<uint8_t*>(CPU_recs) + size_t(crid) * sizeof(TRec);
        const uint8_t* src_i = src + i * kPackedWritebackBytes_;

        // 1) write selected value first
        std::memcpy(dst_rec + ((slot == 0) ? L::kVal1Off : L::kVal2Off),
                    src_i + sizeof(uint32_t),
                    L::kValBytes);

        std::atomic_thread_fence(std::memory_order_release);

        // 2) publish version
        *reinterpret_cast<uint32_t*>(dst_rec + ((slot == 0) ? L::kVer1Off : L::kVer2Off))
            = *reinterpret_cast<const uint32_t*>(src_i);
    }
#endif
}
    template <class TRec>
    void ScatterGather<TRec>::writeback_versions(const uint32_t* h_grids,      // size n
                                                 const uint32_t* h_crids,      // size n
                                                 const uint8_t*  h_curr_slots, // size n (0=>val1, 1=>val2)
                                                 size_t n,
                                                 const TRec* GPU_recs,
                                                 TRec* CPU_recs,
                                                 uint32_t staging_capacity)
    {
        using L = RecordLayout<TRec>;
        (void)sizeof(L);

        if (n == 0) return;

        writeback_d2h_payload_bytes_total_ += static_cast<uint64_t>(n) * kPackedWritebackBytes_;

        // ------------------ non-chunk ------------------
        if (!chunk_mode_) {
            ensure_versions_capacity(n);
            ensure_grids_capacity_(n);

            cudaStream_t s = as_cuda_stream(stream_);

            // H2D grids
            gpu_err_check(cudaMemcpyAsync(d_grids_persistent_,
                                          h_grids,
                                          n * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice,
                                          s));

            // Ensure a persistent d_curr_slots buffer
            if (d_curr_slots_buf_[0] == nullptr || curr_slots_cap_elems_ < n) {
                if (d_curr_slots_buf_[0]) {
                    cudaFree(d_curr_slots_buf_[0]);
                    d_curr_slots_buf_[0] = nullptr;
                }
                gpu_err_check(cudaMalloc(&d_curr_slots_buf_[0], n * sizeof(uint8_t)));
                curr_slots_cap_elems_ = n;
            }

            // H2D curr slots
            gpu_err_check(cudaMemcpyAsync(d_curr_slots_buf_[0],
                                          h_curr_slots,
                                          n * sizeof(uint8_t),
                                          cudaMemcpyHostToDevice,
                                          s));

            // Timeline events (no intermediate syncs — just markers on the stream)
            cudaEvent_t ev_h2d_done, ev_pack_done, ev_d2h_done;
            gpu_err_check(cudaEventCreate(&ev_h2d_done));
            gpu_err_check(cudaEventCreate(&ev_pack_done));
            gpu_err_check(cudaEventCreate(&ev_d2h_done));

            gpu_err_check(cudaEventRecord(ev_h2d_done, s));

            // Pack versions+selected value by GRID
            {
                const int B = 256;
                int G = int((n + B - 1) / B);

                k_pack_versions_by_grids<TRec><<<G, B, 0, s>>>(
                    GPU_recs,
                    d_grids_persistent_,
                    d_curr_slots_buf_[0],
                    reinterpret_cast<uint8_t*>(d_packed_vers_),
                    static_cast<uint32_t>(n));
                gpu_err_check(cudaPeekAtLastError());
            }

            gpu_err_check(cudaEventRecord(ev_pack_done, s));

            // D2H packed vers payload
            gpu_err_check(cudaMemcpyAsync(h_packed_vers_,
                                          d_packed_vers_,
                                          n * kPackedWritebackBytes_,
                                          cudaMemcpyDeviceToHost,
                                          s));

            gpu_err_check(cudaEventRecord(ev_d2h_done, s));

            // For async writeback: use blocking event sync instead of cudaStreamSynchronize.
            // Stream sync busy-waits (spins), polluting node 1 caches for ~7.5ms.
            // Blocking event sync sleeps until GPU signals, keeping caches clean
            // for the scatter that follows.
            if (async_writeback_) {
                cudaEvent_t blocking_ev;
                gpu_err_check(cudaEventCreateWithFlags(&blocking_ev, cudaEventBlockingSync));
                gpu_err_check(cudaEventRecord(blocking_ev, s));
                gpu_err_check(cudaEventSynchronize(blocking_ev)); // sleeps, not spins
                gpu_err_check(cudaEventDestroy(blocking_ev));
            } else {
                gpu_err_check(cudaStreamSynchronize(s));
            }

            // Log GPU phase durations (events already completed, no extra sync needed)
            {
                float pack_ms = 0, d2h_ms = 0;
                cudaEventElapsedTime(&pack_ms, ev_h2d_done, ev_pack_done);
                cudaEventElapsedTime(&d2h_ms, ev_pack_done, ev_d2h_done);
                Logger::GetInstance().Info(
                    "[SG][timeline] gpu_done pack={:.0f}us d2h={:.0f}us",
                    pack_ms * 1000.0f, d2h_ms * 1000.0f);
                cudaEventDestroy(ev_h2d_done);
                cudaEventDestroy(ev_pack_done);
                cudaEventDestroy(ev_d2h_done);
            }

            Logger::GetInstance().Info("[SG][timeline] scatter_start");

            // Scatter into CPU records by CRID using curr slot
            if (async_writeback_)
            {
                scatter_host_versions_range_from_packed_async_(CPU_recs,
                                                                h_crids,
                                                                h_curr_slots,
                                                                /*base=*/0,
                                                                n,
                                                                h_packed_vers_);
            } else
            {
                CpuTimer _t_scatter("SG.writeback.scatter_host_versions_range");
                scatter_host_versions_range_from_packed_(CPU_recs,
                                                    h_crids,
                                                    h_curr_slots,
                                                    /*base=*/0,
                                                    n,
                                                    h_packed_vers_);
            }

            Logger::GetInstance().Info("[SG][timeline] scatter_end");

            return;
        }

        // ------------------ chunk mode ------------------
        const size_t bytes_per_elem = kPackedWritebackBytes_;
        const size_t chunk_elems    = std::max<size_t>(1, chunk_bytes / bytes_per_elem);

        // Versions buffers + grid buffers (reuse your existing chunk buffers)
        ensure_versions_chunk_buffers_(chunk_elems);
        ensure_chunk_buffers_(chunk_elems);

        cudaStream_t cs = as_cuda_stream(copy_stream_);
        cudaStream_t ks = as_cuda_stream(compute_stream_);

        size_t base = 0;
        int buf = 0;

        // Track which chunk lives in each buffer so we know what to scatter on CPU
        size_t buf_base[2] = {0, 0};
        size_t buf_len[2]  = {0, 0};

        while (base < n) {
            const size_t m = std::min(chunk_elems, n - base);

            // Ensure it's safe to reuse this buffer (previous D2H into this host buf completed)
            if (wb_in_flight_[buf]) {
                gpu_err_check(cudaEventSynchronize(as_cuda_event(d2h_done_[buf])));
            }

            // 1) COPY STREAM: H2D grids + curr_slots
            std::memcpy(h_grids_buf_[buf], h_grids + base, m * sizeof(uint32_t));
            gpu_err_check(cudaMemcpyAsync(d_grids_buf_[buf],
                                          h_grids_buf_[buf],
                                          m * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice,
                                          cs));

            std::memcpy(h_curr_slots_buf_[buf], h_curr_slots + base, m * sizeof(uint8_t));
            gpu_err_check(cudaMemcpyAsync(d_curr_slots_buf_[buf],
                                          h_curr_slots_buf_[buf],
                                          m * sizeof(uint8_t),
                                          cudaMemcpyHostToDevice,
                                          cs));

            gpu_err_check(cudaEventRecord(as_cuda_event(h2d_done_[buf]), cs));

            // 2) COMPUTE STREAM: wait + pack versions kernel
            gpu_err_check(cudaStreamWaitEvent(ks, as_cuda_event(h2d_done_[buf]), 0));


            const int B = 256;
            const int G = int((m + B - 1) / B);

            k_pack_versions_by_grids<TRec><<<G, B, 0, ks>>>(
                GPU_recs,
                d_grids_buf_[buf],
                d_curr_slots_buf_[buf],
                reinterpret_cast<uint8_t*>(d_packed_vers_buf_[buf]),
                static_cast<uint32_t>(m));
            gpu_err_check(cudaPeekAtLastError());


            gpu_err_check(cudaEventRecord(as_cuda_event(compute_done_[buf]), ks));

            // 3) COPY STREAM: wait compute then D2H packed versions
            gpu_err_check(cudaStreamWaitEvent(cs, as_cuda_event(compute_done_[buf]), 0));

            gpu_err_check(cudaMemcpyAsync(h_packed_vers_buf_[buf],
                          d_packed_vers_buf_[buf],
                          m * kPackedWritebackBytes_,
                          cudaMemcpyDeviceToHost,
                          cs));

            gpu_err_check(cudaEventRecord(as_cuda_event(d2h_done_[buf]), cs));


            wb_in_flight_[buf] = true;

            // Remember what lives in this buffer
            buf_base[buf] = base;
            buf_len[buf]  = m;

            // While GPU works on newly enqueued chunk, CPU scatters the *other* buffer if ready
            int prev = buf ^ 1;
            if (buf_len[prev] != 0) {
                gpu_err_check(cudaEventSynchronize(as_cuda_event(d2h_done_[prev])));
                if (async_writeback_)
                {
                    scatter_host_versions_range_from_packed_async_(CPU_recs,
                                                                h_crids,
                                                                h_curr_slots,
                                                                buf_base[prev],
                                                                buf_len[prev],
                                                                h_packed_vers_buf_[prev]);
                } else
                {
                    scatter_host_versions_range_from_packed_(CPU_recs,
                                                        h_crids,
                                                        h_curr_slots,
                                                        buf_base[prev],
                                                        buf_len[prev],
                                                        h_packed_vers_buf_[prev]);
                }



                buf_len[prev] = 0;
            }

            base += m;
            buf ^= 1;
        }

        // Drain remaining buffer(s)
        for (int b = 0; b < 2; ++b) {
            if (buf_len[b] != 0) {
                gpu_err_check(cudaEventSynchronize(as_cuda_event(d2h_done_[b])));
                if (async_writeback_)
                {
                    scatter_host_versions_range_from_packed_async_(CPU_recs,
                                    h_crids,
                                    h_curr_slots,
                                    buf_base[b],
                                    buf_len[b],
                                    h_packed_vers_buf_[b]);
                } else
                {
                    scatter_host_versions_range_from_packed_(CPU_recs,
                                        h_crids,
                                        h_curr_slots,
                                        buf_base[b],
                                        buf_len[b],
                                        h_packed_vers_buf_[b]);
                }



                buf_len[b] = 0;
            }
        }

        // Stream hygiene
        gpu_err_check(cudaStreamSynchronize(cs));
        gpu_err_check(cudaStreamSynchronize(ks));
    }

    // ---- Async flush: granular phase control ----

    template <class TRec>
    void ScatterGather<TRec>::wb_async_ensure(size_t n)
    {
        ensure_versions_capacity(n);
        ensure_grids_capacity_(n);

        // Ensure d_curr_slots buffer
        if (d_curr_slots_buf_[0] == nullptr || curr_slots_cap_elems_ < n) {
            if (d_curr_slots_buf_[0]) {
                cudaFree(d_curr_slots_buf_[0]);
                d_curr_slots_buf_[0] = nullptr;
            }
            gpu_err_check(cudaMalloc(&d_curr_slots_buf_[0], n * sizeof(uint8_t)));
            curr_slots_cap_elems_ = n;
        }
    }

    template <class TRec>
    void ScatterGather<TRec>::wb_async_queue_d2d(const uint32_t* d_grids,
                                                  const uint8_t* d_curr_slots,
                                                  size_t n)
    {
        cudaStream_t s = as_cuda_stream(stream_);
        gpu_err_check(cudaMemcpyAsync(d_grids_persistent_, d_grids,
                                      n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
        gpu_err_check(cudaMemcpyAsync(d_curr_slots_buf_[0], d_curr_slots,
                                      n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, s));
    }

    template <class TRec>
    void ScatterGather<TRec>::wb_async_queue_pack(size_t n, const TRec* GPU_recs)
    {
        cudaStream_t s = as_cuda_stream(stream_);
        const int B = 256;

        using L = RecordLayout<TRec>;
        // Warp-coop wins when enough lanes stay busy; per-thread wins for tiny
        // records where launching a full warp to copy a handful of u32s wastes
        // 29 lanes. Empirically the crossover is between 12 B (3 words,
        // warp-coop loses by ~10%) and 100 B (25 words, warp-coop wins by 2x).
        // 32 B (8 words) keeps the 12 B case on per-thread and sends every
        // larger record through the cooperative path.
        constexpr bool kUseWarpCoop = (L::kValBytes >= 32);

        if (kUseWarpCoop) {
            // Warp-cooperative: 1 warp per record. B=256 => 8 warps per block.
            const int warps_per_block = B / 32;
            int G = int((n + warps_per_block - 1) / warps_per_block);

            k_pack_versions_by_grids_warp<TRec><<<G, B, 0, s>>>(
                GPU_recs, d_grids_persistent_, d_curr_slots_buf_[0],
                reinterpret_cast<uint8_t*>(d_packed_vers_),
                static_cast<uint32_t>(n));
        } else {
            // Thread-per-record fallback for small records.
            int G = int((n + B - 1) / B);

            k_pack_versions_by_grids<TRec><<<G, B, 0, s>>>(
                GPU_recs, d_grids_persistent_, d_curr_slots_buf_[0],
                reinterpret_cast<uint8_t*>(d_packed_vers_),
                static_cast<uint32_t>(n));
        }
        gpu_err_check(cudaPeekAtLastError());
    }

    template <class TRec>
    void ScatterGather<TRec>::wb_async_sync()
    {
        cudaStream_t s = as_cuda_stream(stream_);
        gpu_err_check(cudaStreamSynchronize(s));
    }

    template <class TRec>
    void ScatterGather<TRec>::wb_chunked_d2h_scatter(const uint32_t* h_crids,
                                                      const uint8_t* h_curr_slots,
                                                      size_t n, TRec* CPU_recs,
                                                      size_t start_record, size_t num_records)
    {
        // Default: process all records
        if (num_records == 0) num_records = n - start_record;
        if (num_records == 0) return;

        writeback_d2h_payload_bytes_total_ += static_cast<uint64_t>(num_records) * kPackedWritebackBytes_;

        const size_t bytes_per_elem = kPackedWritebackBytes_;
        // CHUNKS_PER_HALF chunks per call. Smaller chunks reduce the time main-
        // thread high-priority D2Hs (e.g. pre_sg_d2h evict-metadata) spend
        // queued behind an in-flight worker chunk on the single shared D2H copy
        // engine: the engine is non-preempting (FIFO once a DMA starts), so the
        // worst-case wait equals one chunk's transfer time. At 1 KB non-split,
        // 2 chunks/half meant ~119 MB / 4.6 ms per chunk; 8 chunks/half drops
        // that to ~30 MB / 1.2 ms per chunk, cutting pre_sg_d2h from ~7.8 ms
        // to ~0.8 ms without slowing worker_total. Trade-off: 4x
        // more cuda API calls per writeback (still tiny in absolute terms).
        // (independent of SG transfer chunk_bytes which stays at 4MB)
        // Auto-tune chunks-per-half from writeback byte size: tiny writebacks
        // (warehouse, district) do 1 chunk while large ones (order_line) do
        // ~4. 6 MiB/chunk was the empirical optimum on tpccdeck W=128,
        // matching a fixed per-table chunk count without per-table machinery.
        constexpr size_t kAutotuneBytesPerChunk = 6 * 1024 * 1024;
        size_t effective_chunks_per_half;
        {
            const size_t total_bytes = num_records * bytes_per_elem;
            size_t chunks = (total_bytes + kAutotuneBytesPerChunk - 1) / kAutotuneBytesPerChunk;
            if (chunks < 1) chunks = 1;
            if (chunks > 64) chunks = 64;
            effective_chunks_per_half = chunks;
        }

        const size_t chunk_elems = std::max<size_t>(1,
            (num_records + effective_chunks_per_half - 1) / effective_chunks_per_half);

        // One-time init: create the D2H stream + persistent blocking-sync events.
        if (!wb_d2h_initialized_) {
            cudaStream_t d2h_s;
            // Non-blocking flag so worker's D2H does not implicitly synchronize
            // with legacy stream-0 ops (main thread's index_transfer etc.).
            gpu_err_check(cudaStreamCreateWithFlags(&d2h_s, cudaStreamNonBlocking));
            wb_d2h_stream_ = d2h_s;

            cudaEvent_t ev0, ev1;
            gpu_err_check(cudaEventCreateWithFlags(&ev0, cudaEventBlockingSync));
            gpu_err_check(cudaEventCreateWithFlags(&ev1, cudaEventBlockingSync));
            wb_d2h_events_[0] = ev0;
            wb_d2h_events_[1] = ev1;

            Logger::GetInstance().Info("[SG] wb_chunked_d2h_scatter: created D2H stream + events");

            wb_d2h_initialized_ = true;
        }

        // Ensure double-buffered host pinned buffers. Use vector-style amortized
        // growth (new_cap = max(needed, 1.5x current)) so each cudaHostAlloc
        // covers many epochs of slow chunk_elems growth instead of re-allocating
        // every time dirty-grid count ticks up by a few hundred. At deep E this
        // matters: under the spec mix, OL dirty grids grow with the undelivered
        // queue, and naive grow-on-exact-overflow triggered ~6 allocs/sec which
        // both added latency and fragmented pinned memory enough to hang
        // cudaHostAlloc around e155 in a prior W=128 E=200 run.
        if (wb_d2h_chunk_cap_ < chunk_elems) {
            const size_t new_cap = std::max(chunk_elems,
                wb_d2h_chunk_cap_ + wb_d2h_chunk_cap_ / 2);
            for (int i = 0; i < 2; ++i) {
                if (h_wb_d2h_buf_[i]) { cudaFreeHost(h_wb_d2h_buf_[i]); h_wb_d2h_buf_[i] = nullptr; }
                gpu_err_check(cudaHostAlloc(&h_wb_d2h_buf_[i], new_cap * bytes_per_elem, cudaHostAllocDefault));
            }
            wb_d2h_chunk_cap_ = new_cap;
            Logger::GetInstance().Info("[SG] wb_chunked_d2h_scatter: allocated 2x{:.1f}MB pinned buffers (chunk_elems={}, new_cap={})",
                new_cap * bytes_per_elem / (1024.0*1024.0), chunk_elems, new_cap);
        }

        cudaStream_t s = static_cast<cudaStream_t>(wb_d2h_stream_);
        const uint8_t* d_src = static_cast<const uint8_t*>(d_packed_vers_) + start_record * bytes_per_elem;
        cudaEvent_t* d2h_done = reinterpret_cast<cudaEvent_t*>(wb_d2h_events_);

        size_t offset = 0;
        int buf = 0;
        bool first_chunk = true;
        size_t prev_abs_base = 0, prev_m = 0;
        int prev_buf = 0;

        while (offset < num_records) {
            const size_t m = std::min(chunk_elems, num_records - offset);

            // Queue D2H on the flush-context stream (separate driver lock from main thread)
            cudaMemcpyAsync(h_wb_d2h_buf_[buf],
                            d_src + offset * bytes_per_elem,
                            m * bytes_per_elem,
                            cudaMemcpyDeviceToHost, s);
            cudaEventRecord(d2h_done[buf], s);

            const size_t abs_base = start_record + offset;

            // While D2H runs, scatter the PREVIOUS chunk
            if (!first_chunk) {
                cudaEventSynchronize(d2h_done[prev_buf]);
                scatter_host_versions_range_from_packed_async_(
                    CPU_recs, h_crids, h_curr_slots,
                    prev_abs_base, prev_m, h_wb_d2h_buf_[prev_buf]);
            }

            prev_abs_base = abs_base;
            prev_m = m;
            prev_buf = buf;
            first_chunk = false;
            offset += m;
            buf ^= 1;
        }

        // Scatter the last chunk
        {
            cudaEventSynchronize(d2h_done[prev_buf]);
            scatter_host_versions_range_from_packed_async_(
                CPU_recs, h_crids, h_curr_slots,
                prev_abs_base, prev_m, h_wb_d2h_buf_[prev_buf]);
        }
    }

    template class ScatterGather<ycsb::YcsbRecords>;
    template class ScatterGather<ycsb::YcsbFieldRecords>;

    // TPC-C explicit instantiations — one per writeable TPC-C table. Same
    // pattern as the YCSB entries above; required because TpccHybridStager
    // calls ScatterGather methods whose definitions live in this TU. History
    // gets no entry (no stager, no SG path).
    template class ScatterGather<Record<tpcc::WarehouseValue>>;
    template class ScatterGather<Record<tpcc::DistrictValue>>;
    template class ScatterGather<Record<tpcc::CustomerValue>>;
    template class ScatterGather<Record<tpcc::ItemValue>>;
    template class ScatterGather<Record<tpcc::StockValue>>;
    template class ScatterGather<Record<tpcc::NewOrderValue>>;
    template class ScatterGather<Record<tpcc::OrderValue>>;
    template class ScatterGather<Record<tpcc::OrderLineValue>>;

}  // namespace epic



#endif  // EPIC_CUDA_AVAILABLE

