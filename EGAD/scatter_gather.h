//
//Created by Christian Tabbah on 2025-10-31
//

#ifndef EPIC_SCATTER_GATHER_H_
#define EPIC_SCATTER_GATHER_H_

#include <benchmarks/ycsb_storage.h>

#include <cstdint>
#include <cstddef>
#include <atomic>

using StreamHandle = void*;
using EventHandle  = void*;

namespace epic {

    template <typename R>
    struct RecordLayout;

    template <typename V>
    struct RecordLayout<Record<V>> {
        using Rec = Record<V>;
        using Val = V;

        static constexpr size_t kRecBytes      = sizeof(Rec);
        static constexpr size_t kVer1Off       = offsetof(Rec, version1);
        static constexpr size_t kVer2Off       = offsetof(Rec, version2);
        static constexpr size_t kVersBytes     = sizeof(uint32_t) * 2;

        static constexpr size_t kVal1Off       = offsetof(Rec, value1);
        static constexpr size_t kVal2Off       = offsetof(Rec, value2);
        static constexpr size_t kValBytes      = sizeof(Val);

        // packed layout for transfer_versions
        static constexpr size_t kPackedVer1Off = 0;
        static constexpr size_t kPackedVer2Off = sizeof(uint32_t);
        static constexpr size_t kPackedValOff  = sizeof(uint32_t) * 2;
        static constexpr size_t kPackedBytes    = sizeof(uint32_t) * 2 + sizeof(Val);

    };


    /*
     * Generic gather -> bulk H2D -> device scatter pipeline for fixed-size POD records
     *
     * CPU arrays are dense by CRID (index = CRID). GPU arrays are dense by GRID.
     * Callers pass CRIDs and their target GRIDs; the pipeline packs from CPU,
     * copies in bulk to temporary device buffers, then scatters in parallel
     * into GPU slots.
     */

    template <class TRec>
    class ScatterGather {
    public:
        struct Options {
            bool own_stream  = true;  // create internal stream if none provided
            bool chunk_mode  = true;  // use chunked gather/scatter to limit host memory usage
            bool async_writeback = false;
            // chunk size in bytes if chunk_mode is enabled
            size_t chunk_bytes = (4ull << 20);  // 4 MB optimal for pipeline overlap
        };

        ScatterGather();
        ~ScatterGather();

        // initialize with optional external stream
        void init(const Options& opt, StreamHandle stream = nullptr);

        // release all resources; safe to call multiple times
        void reset();


        // ensure internal buffers can hold at least n elements
        void ensure_versions_capacity(size_t n);

        void writeback_versions(const uint32_t* h_grids,
                const uint32_t* h_crids,
                const uint8_t*  h_curr_slots,
                size_t n,
                const TRec* GPU_recs,
                TRec* CPU_recs,
                uint32_t staging_capacity);

        // --- Async flush: granular phase control ---
        // Call ensure + queue, then sync at the right time from the epoch loop.

        // Phase 0: ensure buffers are allocated (call once before phases 1-3)
        void wb_async_ensure(size_t n);

        // Phase 1: queue D2D copy from device grids+slots (skips the host round-trip)
        void wb_async_queue_d2d(const uint32_t* d_grids,
                const uint8_t* d_curr_slots, size_t n);

        // Phase 2: queue pack kernel on flush stream (non-blocking)
        void wb_async_queue_pack(size_t n, const TRec* GPU_recs);

        // Phase 3: sync flush stream (waits for all queued GPU work to complete)
        void wb_async_sync();

        // Cumulative payload bytes pushed across the PCIe bus by this instance,
        // counted at the call boundary of transfer_versions / writeback_versions
        // (i.e., n * kPacked{Versions,Writeback}Bytes_ — does not include the
        // small grids+slots H2D copies, which round to <1% of the payload).
        // Intended for end-of-run reporting in the prototype's instrumented runs;
        // not reset between epochs.
        uint64_t admit_h2d_payload_bytes() const { return admit_h2d_payload_bytes_total_; }
        uint64_t writeback_d2h_payload_bytes() const { return writeback_d2h_payload_bytes_total_; }

        // Chunked D2H+scatter pipeline: after pack completes (d_packed_vers_ populated),
        // D2H the packed data in chunks, scattering each chunk as it arrives.
        // Double-buffered: D2H chunk N+1 runs on GPU while CPU scatters chunk N.
        // Call from the worker thread (node 1 cores).
        // start_record/num_records select a subset of the packed buffer (default: all).
        // Chunks-per-half is auto-tuned from total writeback bytes (see .cu).
        void wb_chunked_d2h_scatter(const uint32_t* h_crids,
                const uint8_t* h_curr_slots,
                size_t n, TRec* CPU_recs,
                size_t start_record = 0, size_t num_records = 0);

        /*
         * Single-shot pipeline:
         * - h_crids/h_grids are host arrays of size n
         * - CPU_{recs} are CPU bases
         * - GPU_{recs} are GPU bases
         * Non-blocking; synchronize on the returned stream or call sync()
         */
        void transfer_versions(const uint32_t* h_crids,
                                       const uint32_t* h_grids,
                                       uint32_t epoch,
                                       size_t n,
                                       TRec* CPU_recs,
                                       TRec* GPU_recs);

        // wait for all in-flight work on stream
        void sync();

        // Access the CUDA stream used by this object
        StreamHandle get_stream() const { return stream_; }

    private:
        // internal buffers
        uint32_t*  d_grids_persistent_ = nullptr;  // persistent device buffer for scatter_device_versions and writeback_versions

        size_t cap_grids_elems_ = 0;

        bool own_stream_ = true;                  // whether we own the stream
        StreamHandle stream_ = nullptr;

        Options options_{};

        void ensure_grids_capacity_(size_t n);

        void scatter_host_versions_range_from_packed_(TRec* CPU_recs,
                                                                   const uint32_t* h_crids,
                                                                   const uint8_t*  h_curr_slots,
                                                                   size_t base,
                                                                   size_t n,
                                                                   const void* packed_vers);
        void scatter_host_versions_range_from_packed_async_(TRec* CPU_recs,
                                                                  const uint32_t* h_crids,
                                                                  const uint8_t*  h_curr_slots,
                                                                  size_t base,
                                                                  size_t n,
                                                                  const void* packed_vers);

        // --- chunk mode -------
        bool   chunk_mode_ = false;
        size_t chunk_bytes = (4ull << 20);
        bool async_writeback_ = false;

        size_t chunk_cap_elems_ = 0;

        uint32_t* d_grids_buf_[2]       = {nullptr, nullptr};  // device grids double buffer
        // New: pinned host grids buffers to make grids H2D truly async
        uint32_t* h_grids_buf_[2]       = {nullptr, nullptr};
        EventHandle h2d_done_[2] = {nullptr, nullptr};         // signals H2D finished for that buffer
        EventHandle d2h_done_[2] = {nullptr, nullptr};         // signals D2H finished for that buffer

        StreamHandle copy_stream_   = nullptr;
        StreamHandle compute_stream_ = nullptr;

        EventHandle compute_done_[2] = {nullptr, nullptr};  // signals compute (scatter/pack kernel) done for that buffer

        // Needed to track whether a given double-buffer slot has ever been used.
        bool tx_in_flight_[2] = {false, false};
        bool wb_in_flight_[2] = {false, false};

        using L = RecordLayout<TRec>;
        static constexpr size_t kPackedVersionsBytes_ = L::kPackedBytes;

        // NEW: for writeback_versions only: [one version (4B) | one value]
        static constexpr size_t kPackedWritebackBytes_ = sizeof(uint32_t) + L::kValBytes;

        // PCIe payload accounting (cumulative). Incremented on each
        // transfer_versions / writeback_versions call by n*packed_size.
        uint64_t admit_h2d_payload_bytes_total_ = 0;
        uint64_t writeback_d2h_payload_bytes_total_ = 0;

        // New Buffers used for the Versions path(chunk only, because non-chunk bad)
        // non-chunk versions staging
        void*  h_packed_vers_ = nullptr;
        void*  d_packed_vers_ = nullptr;
        size_t cap_vers_elems_ = 0;

        // chunk versions staging
        size_t chunk_ver_cap_elems_ = 0;
        void*  h_packed_vers_buf_[2] = {nullptr, nullptr};
        void*  d_packed_vers_buf_[2] = {nullptr, nullptr};
        uint8_t* d_prev_slots_buf_[2] = {nullptr, nullptr};   // for chunk transfer_versions
        uint8_t* h_prev_slots_buf_[2] = {nullptr, nullptr};   // pinned, optional but recommended
        size_t prev_slots_cap_elems_ = 0;
        void*    h_curr_slots_buf_[2] = {nullptr, nullptr}; // pinned host (uint8_t bytes)
        uint8_t* d_curr_slots_buf_[2] = {nullptr, nullptr}; // device
        size_t   curr_slots_cap_elems_ = 0;

        // NEW: writeback_versions single-version commit staging (non-chunk)
        void*  h_packed_commit_ = nullptr;
        void*  d_packed_commit_ = nullptr;
        size_t cap_commit_elems_ = 0;

        // Double-buffered host pinned buffers for chunked D2H writeback scatter
        void*  h_wb_d2h_buf_[2] = {nullptr, nullptr};
        size_t wb_d2h_chunk_cap_ = 0;  // capacity in elements per buffer
        void*  wb_d2h_stream_ = nullptr;    // dedicated non-blocking D2H stream (isolated from main-thread streams)
        void*  wb_d2h_events_[2] = {nullptr, nullptr};  // persistent blocking sync events
        bool   wb_d2h_initialized_ = false;

    private:

        // non-chunk prev slots staging
        uint8_t* h_prev_slots_ = nullptr;   // pinned
        size_t   cap_prev_slots_elems_ = 0;

        void ensure_versions_capacity_bytes_(size_t n);
        void ensure_versions_chunk_buffers_(size_t chunk_elems);
        void reset_versions_buffers_();

        void gather_host_versions_(const TRec* CPU_recs,
                                   const uint32_t* h_crids,
                                   void* out_vers,
                                   uint8_t* out_prev_slots,
                                   size_t n,
                                   uint32_t epoch);

        void gather_host_versions_range_(const TRec* CPU_recs,
                                         const uint32_t* h_crids,
                                         size_t base,
                                         size_t n,
                                         uint32_t epoch,
                                         void* out_vers,
                                         uint8_t* out_prev_slots);

        void scatter_device_versions_(TRec* GPU_recs,
                                      const uint32_t* h_grids,
                                      const uint8_t* h_prev_slots,
                                      const void* d_packed_vers,
                                      size_t n);






        //helpers
        void ensure_chunk_buffers_(size_t chunk_elems);
        void reset_chunk_buffers_();

    };

}  // namespace epic

#endif  // EPIC_SCATTER_GATHER_H_

