//
// Created by Shujian Qian on 2023-10-31.
//

#ifndef TPCC_STORAGE_H
#define TPCC_STORAGE_H

#include <benchmarks/tpcc_table.h>
#include <storage.h>

namespace epic::tpcc {

struct TpccVersions
{
    Version<WarehouseValue> *warehouse_version = nullptr;
    Version<DistrictValue> *district_version = nullptr;
    Version<CustomerValue> *customer_version = nullptr;
    Version<HistoryValue> *history_version = nullptr;
    Version<NewOrderValue> *new_order_version = nullptr;
    Version<OrderValue> *order_version = nullptr;
    Version<OrderLineValue> *order_line_version = nullptr;
    Version<ItemValue> *item_version = nullptr;
    Version<StockValue> *stock_version = nullptr;
};

struct TpccRecords
{
    Record<WarehouseValue> *warehouse_record = nullptr;
    Record<DistrictValue> *district_record = nullptr;
    Record<CustomerValue> *customer_record = nullptr;
    Record<HistoryValue> *history_record = nullptr;
    Record<NewOrderValue> *new_order_record = nullptr;
    Record<OrderValue> *order_record = nullptr;
    Record<OrderLineValue> *order_line_record = nullptr;
    Record<ItemValue> *item_record = nullptr;
    Record<StockValue> *stock_record = nullptr;
    // Per-cache-slot "delivered" flag for the OL stager only. Populated by
    // tpcc.cpp after stager construction. Null in non-hybrid modes and on
    // stagers that did not enable delivery-aware eviction. The Delivery
    // executor writes 1 to this flag (per OL grid) after setting OL_DELIVERY_D,
    // so the stager's eviction kernel can prefer delivered slots as victims.
    uint8_t *order_line_delivered_flag = nullptr;

    // Per-table per-slot dirty-flag arrays (one byte per cache slot). Populated
    // by tpcc.cpp after stager construction from each TpccHybridStager's
    // CridGridIndex::device_view(). Null in GPU_ONLY mode (no stagers, no cache).
    // The executor's gpuWriteToTableCoop / gpuWriteToTableThread sets
    // dirty_v1[record_id] or dirty_v2[record_id] inline whenever it commits
    // a record_b write, replacing the staging-time mark_dirty pipeline that
    // ran four GPU kernels per stager per epoch.
    uint8_t *warehouse_dirty_v1  = nullptr; uint8_t *warehouse_dirty_v2  = nullptr;
    uint8_t *district_dirty_v1   = nullptr; uint8_t *district_dirty_v2   = nullptr;
    uint8_t *customer_dirty_v1   = nullptr; uint8_t *customer_dirty_v2   = nullptr;
    uint8_t *new_order_dirty_v1  = nullptr; uint8_t *new_order_dirty_v2  = nullptr;
    uint8_t *order_dirty_v1      = nullptr; uint8_t *order_dirty_v2      = nullptr;
    uint8_t *order_line_dirty_v1 = nullptr; uint8_t *order_line_dirty_v2 = nullptr;
    uint8_t *item_dirty_v1       = nullptr; uint8_t *item_dirty_v2       = nullptr;
    uint8_t *stock_dirty_v1      = nullptr; uint8_t *stock_dirty_v2      = nullptr;
};

} // namespace epic

#endif // TPCC_STORAGE_H
