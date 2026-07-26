//
// Created by Shujian Qian on 2024-04-12.
//

#ifndef TPCC_GPU_AUX_INDEX_H
#define TPCC_GPU_AUX_INDEX_H

#include <any>

#include <txn.h>
#include <storage.h>                              // Record<>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/tpcc_table.h>                // OrderValue, OrderLineValue
#include <benchmarks/tpcc_config.h>
#include <benchmarks/tpcc_cpu_shadow_index.h>     // TpccCpuShadowIndex

namespace epic::tpcc {

template <typename TxnArrayType, typename TxnParamArrayType>
class TpccGpuAuxIndex
{
    std::any impl;

public:
    explicit TpccGpuAuxIndex(TpccConfig &config);
    void loadInitialData();
    void insertTxnUpdates(TxnArrayType &txns, size_t epoch);
    void performRangeQueries(TxnArrayType &txns, TxnParamArrayType &index, size_t epoch);

    // Recovery-side rebuild of the aux index from durable host
    // state. Calls loadInitialData() for the initial population, then layers on
    // every post-initial order/orderline insert by walking the CPU shadow's
    // (w,d,o)->CRID maps and reading the actual o_c_id / o_ol_cnt / ol_i_id
    // values from cpu_records.{order,order_line}_record[CRID]. Required during
    // recovery so Delivery's lookup of customer_id for orders inserted in
    // pre-crash epochs returns the correct CRID (not garbage from an
    // uninitialized GPU slot). Only supports the flat-aux path
    // (EPIC_FLAT_AUX_INDEX=1); throws otherwise.
    void rebuildFromShadow(const TpccCpuShadowIndex& shadow,
                           const Record<OrderValue>* cpu_orders,
                           const Record<OrderLineValue>* cpu_ols);

    // Explicitly free the raw device buffers this aux index
    // owns (the flat insert log and the flat OrderStatus index), so the recover
    // block can discard a loaded instance before building a fresh one without
    // leaking its device memory. Only frees the raw cuda_allocator-owned
    // buffers; the RAII CustomerOrderBTree member self-frees. Idempotent
    // (nulls each pointer); safe to call once on an instance that is about to
    // be destroyed. NOT wired into a destructor on purpose: the aux is built on
    // the normal path too, and freeing in a dtor at program teardown would risk
    // CUDA shutdown-order errors.
    void freeDeviceBuffers();
};

} // namespace epic::tpcc

#endif // TPCC_GPU_AUX_INDEX_H
