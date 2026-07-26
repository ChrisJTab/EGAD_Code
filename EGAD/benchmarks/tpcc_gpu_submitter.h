//
// Created by Shujian Qian on 2023-10-11.
//

#ifndef TPCC_GPU_SUBMITTER_H
#define TPCC_GPU_SUBMITTER_H

#include "benchmarks/tpcc_submitter.h"

#include <any>
#include <vector>

#ifdef EPIC_CUDA_AVAILABLE

namespace epic::tpcc {

template <typename TxnParamArrayType>
class TpccGpuSubmitter : public TpccSubmitter<TxnParamArrayType>
{
    std::vector<std::any> cuda_streams;

    // Per-table derive-pass scratch — 9 tables × 3 counters (read/write/insert).
    // d_counters_[table_idx * 3 + 0..2] holds the device-side atomic targets
    // for countReadWriteInsertKeysTpcc; h_mapped_counters_[same offset] is the
    // mapped-host readback (bypasses the D2H copy engine).
    static constexpr int kCountersPerTable = 3;
    static constexpr int kNumTablesForCounters = 9;
    uint32_t* d_counters_       = nullptr;       // device, 27 scalars
    uint32_t* d_counters_mapped = nullptr;       // device pointer to mapped host
    uint32_t* h_counters_mapped = nullptr;       // host-side mapped block (27)

public:
    using TableSubmitDest = typename TpccSubmitter<TxnParamArrayType>::TableSubmitDest;
    using TpccSubmitter<TxnParamArrayType>::warehouse_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::district_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::customer_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::history_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::order_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::new_order_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::order_line_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::item_submit_dest;
    using TpccSubmitter<TxnParamArrayType>::stock_submit_dest;

    TpccGpuSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
        TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest,
        TableSubmitDest new_order_submit_dest, TableSubmitDest order_submit_dest,
        TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest, TableSubmitDest stock_submit_dest);

    ~TpccGpuSubmitter() override;

    void submit(TxnParamArrayType &txn_array) override;
};

} // namespace epic::tpcc

#endif // EPIC_CUDA_AVAILABLE

#endif // TPCC_GPU_SUBMITTER_H
