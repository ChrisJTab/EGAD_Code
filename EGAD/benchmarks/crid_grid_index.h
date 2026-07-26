// crid_grid_index.h
// Created by Christian Tabbah on 2025-10-01.

#pragma once

// Use a flat array (num_units * 4 bytes GPU RAM) instead of cuco hash map
// for CRID->GRID lookups. O(1) with no tombstone degradation.
#define USE_FLAT_ARRAY_INDEX

#include <any>
#include <benchmarks/ycsb_index.h>
#include <gpu_allocator.h>
#include <benchmarks/ycsb_config.h>

// Forward-declare cudaStream_t so this header is includable from .cpp TUs that
// don't have cuda_runtime.h on their include path. Matches the typedef in
// cuda_runtime_api.h (pointer to opaque CUstream_st).
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace epic {
template<typename TxnType> class TxnArray;  // forward decl (from txn.h)
}

namespace epic::ycsb {
struct YcsbTxnParam;  // forward decl (from ycsb_txn.h)
}

namespace epic::ycsb {

struct CridGridIndexView {
    uint32_t* grid_epoch_last;
    uint8_t* grid_dirty_v1;
    uint8_t* grid_dirty_v2;
    uint32_t  capacity;
    size_t bucket_count;
    // Direct read-only access to the flat lookup table for callers that need
    // to do per-CRID lookups inside their own custom kernels (TPC-C's
    // per-txn-type Remap reads many heterogeneous fields from each
    // TxnParams struct, so it can't go through a generic bulk-lookup
    // wrapper). Same value as Impl::d_crid_to_grid; same num_units bound
    // semantics as the flat-lookup kernels in crid_grid_index.cu.
    const uint32_t* d_crid_to_grid;
    uint32_t        num_units;
};

class CridGridIndex {
public:
    CridGridIndex(Allocator& alloc, uint32_t capacity, uint32_t num_units = 0, double load_factor = 0.5);
    ~CridGridIndex();

    CridGridIndexView device_view() const;

    // Bulk ops – launch CUDA kernels inside .cu.
    // Optional `stream` arg: caller's CUDA stream for the kernel launch
    // (the per-stager prepare streams); default 0 = legacy default stream.
    void bulk_insert(const uint32_t* d_crids, const uint32_t* d_grids, uint32_t n, cudaStream_t stream = 0);
    void bulk_update_crid(const uint32_t* d_old_crids, const uint32_t* d_new_crids, const uint32_t* d_grids, uint32_t n, cudaStream_t stream = 0);
    // Remap in-place: for each op in each txn, rewrite record_ids[i] from CRID → GRID.
    // When split_field=true, INSERT/FULL_READ/FULL_READ_MODIFY_WRITE are skipped (their
    // per-field GRIDs are filled separately via fill_{insert,full_read}_field_grids).
    // When split_field=false, every op is remapped because the non-split executor reads
    // record_ids[i] as a single slot index.
    void remap_txn_record_ids(TxnArray<YcsbTxnParam>& txns,
                              uint32_t num_txns,
                              uint32_t staging_capacity,
                              bool split_field = true);

    void fill_insert_field_grids(TxnArray<YcsbTxnParam>& txns,
                             TxnArray<YcsbExecPlan>& plans,
                             uint32_t num_txns);
    void fill_full_read_field_grids(TxnArray<YcsbTxnParam>& txns,
                             TxnArray<YcsbExecPlan>& plans,
                             uint32_t num_txns);
    uint32_t collect_dirty_grids_v1(uint32_t* d_out_grids, cudaStream_t stream = 0) const;
    uint32_t collect_dirty_grids_v2(uint32_t* d_out_grids, cudaStream_t stream = 0) const;
    void clear_dirty_by_grids(const uint32_t* d_grids, const uint8_t* d_slots, uint32_t n, cudaStream_t stream = 0);
    void touch_by_grids(const uint32_t* d_grids, uint32_t n, uint32_t epoch, cudaStream_t stream = 0);
    void bulk_lookup_crid_to_grid_and_mark_needed(const uint32_t* d_crids,
                                                            uint32_t* d_out_grids,
                                                            uint32_t n,
                                                            uint8_t* d_needed_flag,
                                                            uint32_t cap,
                                                            cudaStream_t stream = 0);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace epic::ycsb
