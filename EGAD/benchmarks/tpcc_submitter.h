//
// Created by Shujian Qian on 2023-10-11.
//

#ifndef TPCC_SUBMITTER_H
#define TPCC_SUBMITTER_H

#include <cstdint>
#include <tuple>

#include "util_log.h"
#include "tpcc_txn.h"

namespace epic::tpcc {

template <typename TxnParamArrayType>
class TpccSubmitter
{
public:
    struct TableSubmitDest
    {
        uint32_t *d_num_ops = nullptr;
        uint32_t *d_op_offsets = nullptr;
        void *d_submitted_ops = nullptr;
        void *temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        uint32_t &curr_num_ops;

        // Per-epoch counts populated by the submitter's derive pass; the
        // hybrid_staging path consumes them as the n_all / n_read / n_write /
        // n_insert arguments to TpccHybridStager::prepareEpoch. References
        // into the corresponding TableExecutionPlanner.
        uint32_t &curr_num_read_ops;
        uint32_t &curr_num_write_ops;
        uint32_t &curr_num_insert_ops;

        // Per-epoch dense arrays of CRIDs touched by this table this epoch.
        // d_all_keys is the union (with duplicates kept); d_read_keys,
        // d_write_keys, d_insert_keys are sentinel-padded subsets.
        // d_op_is_insert is parallel to d_submitted_ops and marks which ops
        // are insert-emitting per the submitter's per-txn-type knowledge
        // (NewOrder writes for Order/NewOrder/OrderLine; nothing else).
        uint32_t *d_all_keys      = nullptr;
        uint32_t *d_read_keys     = nullptr;
        uint32_t *d_write_keys    = nullptr;
        uint32_t *d_insert_keys   = nullptr;
        uint8_t  *d_op_is_insert  = nullptr;

        // Per-table planner allocation size for d_op_is_insert (and the
        // related per-op arrays). The submitter uses this as the memset
        // bound when re-zeroing d_op_is_insert each epoch; without it the
        // submitter would have to walk the full op range every epoch.
        size_t   max_num_ops      = 0;
    };

    TableSubmitDest warehouse_submit_dest;
    TableSubmitDest district_submit_dest;
    TableSubmitDest customer_submit_dest;
    TableSubmitDest history_submit_dest;
    TableSubmitDest new_order_submit_dest;
    TableSubmitDest order_submit_dest;
    TableSubmitDest order_line_submit_dest;
    TableSubmitDest item_submit_dest;
    TableSubmitDest stock_submit_dest;

    virtual ~TpccSubmitter() = default;
    TpccSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
        TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest,
        TableSubmitDest new_order_submit_dest, TableSubmitDest order_submit_dest,
        TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest, TableSubmitDest stock_submit_dest)
        : warehouse_submit_dest(warehouse_submit_dest)
        , district_submit_dest(district_submit_dest)
        , customer_submit_dest(customer_submit_dest)
        , history_submit_dest(history_submit_dest)
        , new_order_submit_dest(new_order_submit_dest)
        , order_submit_dest(order_submit_dest)
        , order_line_submit_dest(order_line_submit_dest)
        , item_submit_dest(item_submit_dest)
        , stock_submit_dest(stock_submit_dest)
    {}

    virtual void submit(TxnParamArrayType &txn_array)
    {
        auto &logger = Logger::GetInstance();
        logger.Error("TpccSubmittor::submit not implemented");
    };
};
} // namespace epic::tpcc

#endif // TPCC_SUBMITTER_H
