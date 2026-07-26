// tpcc_staging_capacities.h
//
// Per-record byte sizes for the 8 TPC-C tables. tpcc.cpp fills this with
// sizeof(Record<X>) values and passes it to the workload-aware autosizer
// (tpcc_workload_aware_capacities.h), which derives the per-table GPU cache
// capacities from it. Kept in its own header so both stay pure host code.

#ifndef TPCC_STAGING_CAPACITIES_H
#define TPCC_STAGING_CAPACITIES_H

#include <cstddef>

namespace epic::tpcc {

// Per-record byte sizes for the 8 tables. Caller fills with sizeof(Record<X>)
// from tpcc_storage.h.
struct TpccRecordBytes {
    size_t warehouse;
    size_t district;
    size_t customer;
    size_t item;
    size_t stock;
    size_t new_order;
    size_t order;
    size_t order_line;
};

} // namespace epic::tpcc

#endif // TPCC_STAGING_CAPACITIES_H
