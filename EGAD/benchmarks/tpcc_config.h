//
// Created by Shujian Qian on 2023-09-20.
//

#ifndef TPCC_CONFIG_H
#define TPCC_CONFIG_H

#include <cstdint>
#include <cstdlib>

#include "util_device_type.h"
#include "util_exec_mode.h"

namespace epic::tpcc {

// These index/store toggles are part of EGAD's hybrid design, so they
// default ON for hybrid_staging and OFF otherwise (cpu_only stays stock Epic;
// gpu_only opts in explicitly for fairness via its env vars). An explicit env
// value always wins: VAR=1 forces on, VAR=0 forces off, unset uses the mode
// default.
inline bool envBoolOrHybridDefault(const char *var, ExecMode mode)
{
    const char *e = std::getenv(var);
    return e ? (std::atoi(e) != 0) : (mode == ExecMode::HYBRID_STAGING);
}

struct TpccTxnMix
{
    uint32_t new_order = 50;
    uint32_t payment = 50;
    uint32_t order_status = 0;
    uint32_t delivery = 0;
    uint32_t stock_level = 0;

    TpccTxnMix() = default;
    TpccTxnMix(uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level);

    // Weights are interpreted as ratios of any positive sum. Helpers below
    // give the actual share of each txn type, which is what callers that
    // need a probability or a per-epoch count multiplier should use instead
    // of dividing by a hardcoded 100.
    uint32_t totalWeight() const {
        return new_order + payment + order_status + delivery + stock_level;
    }
    double newOrderShare() const {
        return double(new_order) / double(totalWeight());
    }
};

struct TpccConfig
{
    TpccTxnMix txn_mix;
    size_t num_txns = 100'000;
    size_t epochs = 20;
    size_t num_warehouses = 8;
    size_t order_table_size = 1'000'000;
    size_t orderline_table_size = 15'000'000;
    DeviceType index_device = DeviceType::GPU;
    DeviceType initialize_device = DeviceType::GPU;
    DeviceType execution_device = DeviceType::GPU;
    ExecMode execution_mode = ExecMode::GPU_ONLY;
    bool gacco_separate_txn_queue = true;
    bool gacco_use_atomic = false;
    bool gacco_tpcc_stock_use_atomic = true;
    uint32_t cpu_exec_num_threads = 1;
    bool overlap_flush = false;

    // Insert-CRID verifier. Off by default; --verify_tpcc enables.
    // Mirrors the YCSB ycsb.cpp::verifyInsertedRecords pattern. Catches the
    // failure class where insert writes don't reach the CPU primary store.
    bool verify_tpcc = false;

    // Per-table GPU cache capacity for hybrid_staging mode. 0 (the default)
    // means "fall back to <table>TableSize()" — same value as Epic's existing
    // GPU allocation, so admission populates everything once and FIFO eviction
    // never triggers. Eviction-heavy runs override these via CLI flag to
    // undersize a specific table's cache. cacheCapacity<X>() below
    // resolves the 0-default to the table-size fallback.
    size_t cache_capacity_warehouse  = 0;
    size_t cache_capacity_district   = 0;
    size_t cache_capacity_customer   = 0;
    size_t cache_capacity_item       = 0;
    size_t cache_capacity_stock      = 0;
    size_t cache_capacity_new_order  = 0;
    size_t cache_capacity_order      = 0;
    size_t cache_capacity_order_line = 0;

    // Autosizer options for hybrid_staging mode. The autosizer runs at startup
    // (between planner init and record alloc) and sets cache_capacity_<X> for
    // any table whose value is still 0, sized to fit in HBM.
    //   hybrid_hbm_reserve_bytes -- HBM left unallocated when computing the
    //     cache budget. Covers SG pinned device buffers, CUDA driver overhead,
    //     cudaMalloc fragmentation, and runtime allocations during the run.
    //     Per-slot stager scratch is accounted for separately inside the
    //     autosizer (kPerSlotScratchBytes). Default 4 GiB.
    //   hybrid_hbm_budget_bytes  -- explicit cache budget. 0 = use cudaMemGetInfo.
    size_t hybrid_hbm_reserve_bytes = 4ULL * 1024 * 1024 * 1024;
    size_t hybrid_hbm_budget_bytes  = 0;

    size_t warehouseTableSize() const
    {
        return num_warehouses * 2;
    }
    size_t districtTableSize() const
    {
        return num_warehouses * 2 * 10;
    }
    size_t customerTableSize() const
    {
        return num_warehouses * 2 * 10 * 3'000;
    }
    size_t historyTableSize() const
    {
        return num_warehouses * 2 * 20;
        /* TODO: fix history table size */
        return num_warehouses * 2 * 20 * 96'000 * num_warehouses * 2 * 20;
    }
    // Spec-mix-realistic insert-pool sizing for the three growing tables.
    // The worst-case formula assumes every txn could be a NewOrder × 15 OL,
    // which over-allocates by 7.5x for the spec-compliant 45/43/4/4/4 mix.
    // Defaults to mix-aware sizing for hybrid_staging (capped at the
    // worst-case formula so high-NewOrder mixes like tpccn do not break),
    // and to the worst-case formula otherwise. EPIC_MIX_REALISTIC_SIZING
    // overrides either way (=1 forces mix-aware, =0 forces worst-case).
    //
    // formula:  realistic_per_epoch = num_txns * NewOrderShare * safety
    //           realistic_ol_per_epoch = realistic_per_epoch * avg_OL_per_NewOrder
    //                                  = realistic_per_epoch * 10
    //           pool = min( realistic, worst_case )
    bool mixRealisticSizingEnabled() const
    {
        return envBoolOrHybridDefault("EPIC_MIX_REALISTIC_SIZING", execution_mode);
    }
    size_t newOrderInsertPoolSize() const
    {
        const size_t worst = num_txns * (epochs + 1);
        if (!mixRealisticSizingEnabled()) return worst;
        constexpr double safety = 1.5;
        const double new_order_share = txn_mix.newOrderShare();
        const size_t realistic = static_cast<size_t>(
            double(num_txns) * double(epochs + 1) * new_order_share * safety);
        return realistic < worst ? realistic : worst;
    }
    size_t orderLineInsertPoolSize() const
    {
        const size_t worst = num_txns * (epochs + 1) * 15;
        if (!mixRealisticSizingEnabled()) return worst;
        constexpr double safety = 1.5;
        constexpr double avg_ol = 10.0;
        const double new_order_share = txn_mix.newOrderShare();
        const size_t realistic = static_cast<size_t>(
            double(num_txns) * double(epochs + 1) * new_order_share * avg_ol * safety);
        return realistic < worst ? realistic : worst;
    }
    size_t newOrderTableSize() const
    {
        return num_warehouses * 2 * 10 * 900 + newOrderInsertPoolSize();
    }
    size_t orderTableSize() const
    {
        return num_warehouses * 2 * 10 * 3000 + newOrderInsertPoolSize();
    }
    size_t orderLineTableSize() const
    {
        return num_warehouses * 10 * 15 * 3000 + orderLineInsertPoolSize();
    }
    size_t itemTableSize() const
    {
        return 200'000;
    }
    size_t stockTableSize() const
    {
        return 200'000 * num_warehouses;
    }

    // Per-district slot count for the GPU aux index. Each district gets this
    // many slots in order_num_items / order_customers / order_items, and an
    // order is parked at `district_id * num_slots_per_district + o_id`, so this
    // must exceed the largest runtime o_id or NewOrder inserts overflow the
    // array (corruption / illegal access).
    //
    // SINGLE SOURCE OF TRUTH: the gpu_aux_index ctor sizes its arrays from this,
    // and runBenchmark's startup overflow guard warns against this same value,
    // so the array size and the guard threshold can never disagree.
    //
    // The formula ceil-divides (truncation would undercount by up to ~1
    // NewOrder per district per epoch), scales by the mix's NewOrder share
    // under EPIC_MIX_REALISTIC_SIZING (assuming every txn is a NewOrder
    // over-provisions milder mixes), and multiplies by 3/2 for the binomial
    // spread of NewOrder counts across districts (a 3-4 sigma busy district
    // can exceed the mean by a few hundred orders); +3000 covers the initial
    // per-district load (o_id 1..3000).
    uint32_t auxNumSlotsPerDistrict() const
    {
        const size_t denom = num_warehouses * 10;
        const double share = mixRealisticSizingEnabled() ? txn_mix.newOrderShare() : 1.0;
        const size_t total_new_orders = static_cast<size_t>(
            double(num_txns) * double(epochs) * share);
        const size_t per_district = (total_new_orders + denom - 1) / denom;
        return static_cast<uint32_t>(per_district * 3 / 2 + 3000);
    }

    // Resolve cache_capacity_<table> to the configured value, falling back to
    // <table>TableSize() when zero (the cache then stays quiescent: the
    // resident set fills it exactly and eviction never triggers).
    // Eviction-heavy runs set these explicitly to undersize a table.
    size_t cacheCapacityWarehouse() const  { return cache_capacity_warehouse  ? cache_capacity_warehouse  : warehouseTableSize();  }
    size_t cacheCapacityDistrict() const   { return cache_capacity_district   ? cache_capacity_district   : districtTableSize();   }
    size_t cacheCapacityCustomer() const   { return cache_capacity_customer   ? cache_capacity_customer   : customerTableSize();   }
    size_t cacheCapacityItem() const       { return cache_capacity_item       ? cache_capacity_item       : itemTableSize();       }
    size_t cacheCapacityStock() const      { return cache_capacity_stock      ? cache_capacity_stock      : stockTableSize();      }
    size_t cacheCapacityNewOrder() const   { return cache_capacity_new_order  ? cache_capacity_new_order  : newOrderTableSize();   }
    size_t cacheCapacityOrder() const      { return cache_capacity_order      ? cache_capacity_order      : orderTableSize();      }
    size_t cacheCapacityOrderLine() const  { return cache_capacity_order_line ? cache_capacity_order_line : orderLineTableSize();  }
};

} // namespace epic::tpcc

#endif // TPCC_CONFIG_H
