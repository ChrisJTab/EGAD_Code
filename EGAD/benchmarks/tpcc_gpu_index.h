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
    // refreshes index_device_view. Assumes the CUDA context is alive.
    // Must NOT be called concurrently with any other GPU activity.
    void rebuildIndexesFromShadow(TpccFreeStarts current_free_starts);

    // Returns the current free_starts for the 3 growing tables. Equal
    // to the per-table free-list cursors maintained by indexTxns.
    TpccFreeStarts getInsertCounts() const;
};

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_GPU_INDEX_H
