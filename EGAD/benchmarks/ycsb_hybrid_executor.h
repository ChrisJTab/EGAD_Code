//
// Created by Christian Tabbah on 2025-09-17.
//

#ifndef EPIC__YCSB_HYBRID_EXECUTOR_H
#define EPIC__YCSB_HYBRID_EXECUTOR_H

#include "ycsb_executor.h"

namespace epic::ycsb {

class HybridExecutor : public Executor
{
public:
    HybridExecutor(YcsbRecordArrType GPU_records, YcsbVersionArrType GPU_versions, uint32_t gpu_capacity, TxnArray<YcsbTxnParam> txn, TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : Executor(GPU_records, GPU_versions, txn, plan, config) // base class holds the GPU records and versions pointers
        , gpu_capacity_(gpu_capacity)
    {
    };
    ~HybridExecutor() override = default;

    void execute(uint32_t epoch) override;

    // Per-epoch dirty-flag arrays (one byte per cache slot), set once at
    // construction by HybridStager from CridGridIndex::device_view(). The
    // exec kernels propagate these into gpuWriteToTableCoop, which marks
    // the slot it picks. Replaces the staging-time mark_dirty pipeline.
    void set_dirty_arrays(uint8_t *dirty_v1, uint8_t *dirty_v2)
    {
        dirty_v1_ = dirty_v1;
        dirty_v2_ = dirty_v2;
    }

private:
    uint64_t gpu_capacity_;
    uint8_t *dirty_v1_ = nullptr;
    uint8_t *dirty_v2_ = nullptr;
};
}
#endif // EPIC__YCSB_HYBRID_EXECUTOR_H
