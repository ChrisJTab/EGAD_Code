//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
#define EPIC_BENCHMARKS_YCSB_GPU_INDEX_H

#include <any>

#include <benchmarks/ycsb_config.h>
#include <benchmarks/ycsb_cpu_shadow_index.h>
#include <benchmarks/ycsb_index.h>

namespace epic::ycsb {

class YcsbGpuIndex : public YcsbIndex
{
public:
    YcsbConfig ycsb_config;
    std::any gpu_index_impl;
    // The CPU shadow is owned externally (by YcsbBenchmark): host-side
    // ground truth the recovery path rebuilds the GPU index from.
    // Bound at construction.
    explicit YcsbGpuIndex(YcsbConfig ycsb_config, YcsbCpuShadowIndex& shadow);

    void loadInitialData() override;
    void indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id) override;

    // Rebuild the GPU cuco::static_map from the CPU shadow_shards_.
    // Intended for GPU-crash recovery: call after reinitializing the GPU
    // context, passing the last-known committed free_start. Only entries
    // whose assigned CRID is < starting_num_records + current_free_start are
    // uploaded; any future CRIDs (speculatively pre-computed and never used)
    // are skipped. Must NOT be called concurrently with any other GPU activity.
    void rebuildCucoFromShadow(uint32_t current_free_start);

    // Total number of insert CRIDs allocated since startup. Equal to the
    // free-list cursor (free_start). Inserted CRIDs occupy the dense range
    // [starting_num_records, starting_num_records + getInsertCount()).
    uint32_t getInsertCount() const;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
