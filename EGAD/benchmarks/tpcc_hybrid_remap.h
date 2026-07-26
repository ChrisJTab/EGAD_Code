//
// tpcc_hybrid_remap.h — per-txn-type Remap that rewrites record_id fields in
// TpccTxnParam from CRID to GRID using each per-table CridGridIndex flat
// lookup. Lives separately from TpccHybridStager because the stager owns one
// table's CRID→GRID mapping and Remap fans out across all tables for each
// txn type. Apply before the hybrid executor so params->X_id resolves
// directly into the appropriate cache pool.
//

#ifndef EPIC__TPCC_HYBRID_REMAP_H
#define EPIC__TPCC_HYBRID_REMAP_H

#include "benchmarks/crid_grid_index.h"
#include "benchmarks/tpcc_txn.h"

namespace epic::tpcc {

// One CridGridIndexView per writeable TPC-C table. History is excluded
// because the executor doesn't read or write it.
struct TpccCridGridViews {
    epic::ycsb::CridGridIndexView warehouse;
    epic::ycsb::CridGridIndexView district;
    epic::ycsb::CridGridIndexView customer;
    epic::ycsb::CridGridIndexView item;
    epic::ycsb::CridGridIndexView stock;
    epic::ycsb::CridGridIndexView new_order;
    epic::ycsb::CridGridIndexView order;
    epic::ycsb::CridGridIndexView order_line;
};

// Dispatch over the txn array: each thread reads its txn's type and runs the
// matching per-txn-type Remap. CRID→GRID lookups happen in place against the
// caller-supplied views. Mirrors tpcc_gpu_submitter.cu's submitTpccTxn
// dispatch shape.
void launchRemapTpccTxns(TpccTxnParamArrayT& params, uint32_t num_txns,
                         const TpccCridGridViews& views);

} // namespace epic::tpcc

#endif // EPIC__TPCC_HYBRID_REMAP_H
