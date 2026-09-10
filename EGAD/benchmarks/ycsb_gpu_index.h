//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
#define EPIC_BENCHMARKS_YCSB_GPU_INDEX_H

#include <any>
#include <vector>

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
    // are skipped. current_delete_count resyncs the delete cursor so replay
    // re-appends the delete log at the same positions. Must NOT be called
    // concurrently with any other GPU activity.
    void rebuildCucoFromShadow(uint32_t current_free_start, uint32_t current_delete_count = 0);

    // Total number of insert CRIDs allocated since startup. Equal to the
    // free-list cursor (free_start). Inserted CRIDs occupy the dense range
    // [starting_num_records, starting_num_records + getInsertCount()).
    uint32_t getInsertCount() const;

    // Total number of deletes applied since startup (the delete-log cursor).
    uint32_t getDeleteCount() const;

    // This epoch's deleted CRIDs (device pointer + count), valid until the
    // next indexTxns call. The stager marks these cache slots
    // reclaim-first before its eviction pass. Count is 0 for mixes
    // without deletes.
    const uint32_t* deleteCridsDevice() const;
    uint32_t numDeletesThisEpoch() const;

    // The records minted this epoch occupy the CRID range
    // [mintedBegin(), mintedBegin() + numMintedThisEpoch()). An INSERT op
    // whose record id lies in it creates the record (the stager allocates
    // its cache slot without a Primary Store fetch); an INSERT op resolving
    // outside it writes an existing record.
    uint32_t mintedBegin() const;
    uint32_t numMintedThisEpoch() const;

#ifdef EGAD_VALIDATION
    // Map check: every (key, expected CRID) pair must be found in the GPU
    // index with that CRID and none of the dead keys may be found. Logs one
    // [MAP-CHECK] line; returns the number of violations.
    uint32_t verifyLiveMapping(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& expected,
                               const std::vector<uint32_t>& dead) const;
#endif
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
