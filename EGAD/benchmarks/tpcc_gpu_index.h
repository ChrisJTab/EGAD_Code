//
// Created by Shujian Qian on 2023-11-20.
//

#ifndef EPIC_BENCHMARKS_TPCC_GPU_INDEX_H
#define EPIC_BENCHMARKS_TPCC_GPU_INDEX_H

#include <any>

#include <benchmarks/tpcc_cpu_shadow_index.h>  // TpccFreeStarts + TpccCpuShadowIndex
#include <benchmarks/tpcc_index.h>
#include <benchmarks/tpcc_table.h>

namespace epic::tpcc {

template <typename TxnArrayType, typename TxnParamArrayType>
class TpccGpuIndex : public TpccIndex<TxnArrayType, TxnParamArrayType>
{
public:
    TpccConfig tpcc_config;
    std::any gpu_index_impl;
    // The CPU shadow is owned externally (by TpccDb): host-side ground
    // truth the recovery path rebuilds the GPU index from. Bound at
    // construction.
    explicit TpccGpuIndex(TpccConfig tpcc_config, TpccCpuShadowIndex& shadow);

    void loadInitialData() override;
    void indexTxns(TxnArrayType &txn_array, TxnParamArrayType &index_array, uint32_t epoch_id) override;

    // === CPU-shadow recovery API ===
    //
    // Rebuild the 8 active GPU indices (W/D/C/I/S/NO/O/OL; History is
    // excluded) from the CPU shadow at the given free_starts. Recreates
    // the cuco maps and the OL flat array, reseeds the 3 d_*_free_rows
    // arrays, batch-uploads every non-sentinel shadow entry, and
    // refreshes index_device_view. current_no_delete_count resyncs the
    // NewOrder delete-log cursor so replay re-appends at the same
    // positions. Assumes the CUDA context is alive. Must NOT be called
    // concurrently with any other GPU activity.
    void rebuildIndexesFromShadow(TpccFreeStarts current_free_starts,
                                  uint32_t current_no_delete_count = 0);

    // Returns the current free_starts for the 3 growing tables. Equal
    // to the per-table free-list cursors maintained by indexTxns.
    TpccFreeStarts getInsertCounts() const;

    // Total NewOrder deletes applied since startup (the NO delete-log
    // cursor). Delivery consumes each NewOrder row exactly once, so this
    // advances with delivered orders on mixes that include Delivery.
    uint32_t getNoDeleteCount() const;

    // This epoch's deleted NewOrder CRIDs (device pointer + count), valid
    // until the next indexTxns call; the NO stager marks these slots
    // reclaim-first. Count is 0 on mixes without Delivery.
    const uint32_t* noDeleteCridsDevice() const;
    uint32_t numNoDeletesThisEpoch() const;
};

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_GPU_INDEX_H
