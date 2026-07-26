// tpcc_workload_aware_capacities.h
//
// Workload-aware autosizer for the TPC-C staging caches. Derives the three
// growing tables' (NewOrder, Order, OrderLine) cache sizes from the
// per-table workload demand:
//
//   working_set(table) = per_epoch_reads + per_epoch_writes + per_epoch_inserts
//                       + reach_back_epochs(table) × per_epoch_inserts
//   cap(table) = max( working_set × kSafetyFactor,
//                     per_epoch_inserts × kInsertHeadroomFactor )
//
// If the sum of caps exceeds the HBM budget, all caps are scaled down
// proportionally and a warning is emitted. If it fits, the unused HBM
// stays free.
//
// Per-(transaction type, table) access patterns are hardcoded from the
// TPC-C specification — they are part of the workload definition, not
// empirical tuning constants. The reach-back depth per (txn_type, table)
// is the only parameter that is workload-property-derived rather than
// directly specified; see comments at kTpccAccess for the rationale per
// pair.
//
// The sizing policy for hybrid staging: input is TpccConfig + the measured
// HBM budget; output is cfg.cache_capacity_<X> for any table still at its
// 0 (auto) default. Explicit CLI overrides are preserved.

#ifndef TPCC_WORKLOAD_AWARE_CAPACITIES_H
#define TPCC_WORKLOAD_AWARE_CAPACITIES_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "benchmarks/tpcc_config.h"
#include "benchmarks/tpcc_staging_capacities.h"
#include "util_log.h"

namespace epic::tpcc {

// Per-(txn_type, table) access pattern, in counts per single transaction
// invocation, plus a reach_back_epochs value indicating how far back the
// transaction's reads reach into past inserts (used for working-set sizing
// of the growing tables NO / O / OL).
struct TxnTableAccess {
    double per_txn_reads;          // average reads of this table per txn
    double per_txn_writes;         // average writes (updates) of this table per txn
    double per_txn_inserts;        // average inserts into this table per txn
    int    reach_back_epochs;      // for reads of this table: how far back into past inserts
};

// Table indices into the per-mix access arrays below.
enum TpccTableIdx : int {
    kT_W = 0, kT_D, kT_I, kT_C, kT_S, kT_NO, kT_O, kT_OL,
    kT_count
};

// Per-(transaction type, table) access pattern. Numbers come from the
// TPC-C 5.11 specification. reach_back_epochs is approximated from the
// transaction semantics (see comments).
//
// Indexed [txn_type] where txn_type matches TpccTxnMix's field order:
//   0=NewOrder, 1=Payment, 2=OrderStatus, 3=Delivery, 4=StockLevel
//
// Access counts are per-call averages (some are over per-warehouse vs
// per-district scopes; the per-call interpretation matches how TpccTxnMix
// percentages combine with num_txns in compute_demand below).
constexpr std::array<std::array<TxnTableAccess, kT_count>, 5> kTpccAccess = {{
    // NewOrder
    {{
        /*W*/  {1, 0,  0, 0},     // read warehouse for tax info
        /*D*/  {1, 1,  0, 0},     // read + write district (next_o_id++)
        /*I*/  {10, 0, 0, 0},     // 10 line items
        /*C*/  {1, 0,  0, 0},     // customer for discount
        /*S*/  {10, 10, 0, 0},    // stock for each line item: read + update
        /*NO*/ {0, 0,  1, 0},     // 1 NewOrder insert
        /*O*/  {0, 0,  1, 0},     // 1 Order insert
        /*OL*/ {0, 0,  10, 0},    // 10 OrderLine inserts
    }},
    // Payment
    {{
        /*W*/  {1, 1, 0, 0},      // ytd update
        /*D*/  {1, 1, 0, 0},      // ytd update
        /*I*/  {0, 0, 0, 0},
        /*C*/  {1, 1, 0, 0},      // balance update
        /*S*/  {0, 0, 0, 0},
        /*NO*/ {0, 0, 0, 0},
        /*O*/  {0, 0, 0, 0},
        /*OL*/ {0, 0, 0, 0},
    }},
    // OrderStatus
    {{
        /*W*/  {0, 0, 0, 0},
        /*D*/  {0, 0, 0, 0},
        /*I*/  {0, 0, 0, 0},
        /*C*/  {1, 0, 0, 0},      // customer lookup
        /*S*/  {0, 0, 0, 0},
        /*NO*/ {0, 0, 0, 0},
        /*O*/  {1, 0, 0, 50},     // most recent order for customer; reach back unbounded in
                                  //   theory, capped here at 50 epochs (matches measured WS).
        /*OL*/ {10, 0, 0, 50},    // 10 OL rows of that order; same reach-back as O.
    }},
    // Delivery
    {{
        /*W*/  {0, 0, 0, 0},
        /*D*/  {0, 0, 0, 0},
        /*I*/  {0, 0, 0, 0},
        /*C*/  {10, 10, 0, 0},    // one per district
        /*S*/  {0, 0, 0, 0},
        /*NO*/ {10, 10, 0, 10},   // oldest pending per district; pending bounded by NO_rate-Del_rate
        /*O*/  {10, 10, 0, 10},   // same orders
        /*OL*/ {100, 100, 0, 10}, // 10 OL × 10 orders
    }},
    // StockLevel
    {{
        /*W*/  {0, 0, 0, 0},
        /*D*/  {1, 0, 0, 0},
        /*I*/  {0, 0, 0, 0},
        /*C*/  {0, 0, 0, 0},
        /*S*/  {20, 0, 0, 0},     // ~20 unique stocks across last 20 orders
        /*NO*/ {0, 0, 0, 0},
        /*O*/  {0, 0, 0, 0},
        /*OL*/ {200, 0, 0, 2},    // last 20 orders × 10 OL each; recent only
    }},
}};

// Aggregated per-table demand for one epoch given the mix.
struct PerTableDemand {
    double reads        = 0;
    double writes       = 0;
    double inserts      = 0;
    int    reach_back   = 0;   // max over txn types in mix (only those with mix > 0)
};

// Compute per-epoch per-table demand from the mix, num_txns/epoch.
inline PerTableDemand compute_per_table_demand(const TpccConfig& cfg, TpccTableIdx t)
{
    PerTableDemand d;
    const uint32_t mix_w[5] = {
        cfg.txn_mix.new_order, cfg.txn_mix.payment, cfg.txn_mix.order_status,
        cfg.txn_mix.delivery,  cfg.txn_mix.stock_level
    };
    // Weights are interpreted as ratios summing to anything > 0. Standard
    // mixes sum to 100 (canonical percentages); the deck mix sums to 23.
    // Divide by the total to get the actual share, not a hardcoded 100.
    const double total_w = static_cast<double>(cfg.txn_mix.totalWeight());
    for (int tt = 0; tt < 5; ++tt) {
        if (mix_w[tt] == 0) continue;
        const double frac = static_cast<double>(mix_w[tt]) / total_w;
        const auto& a = kTpccAccess[tt][t];
        d.reads   += frac * cfg.num_txns * a.per_txn_reads;
        d.writes  += frac * cfg.num_txns * a.per_txn_writes;
        d.inserts += frac * cfg.num_txns * a.per_txn_inserts;
        if (a.reach_back_epochs > d.reach_back) d.reach_back = a.reach_back_epochs;
    }
    return d;
}

// Cap formula. kSafetyFactor scales the per-epoch component; the insert
// headroom must cover the current epoch's admits PLUS enough evictable older
// rows for the FIFO eviction to free room, so a pure-insert table (reach_back=0,
// e.g. OrderLine under the NP mix) needs roughly 2x its per-epoch inserts just
// to keep up. The per-epoch insert count is random (NewOrder's ol_cnt is uniform
// 5..15, so the realized count fluctuates ~1% around the mean-10 estimate); a
// factor of exactly 2.0 left zero margin and overflowed when the realized count
// ran above the estimate (NP crashed at cap=1M with ~500.9K actual inserts while
// the estimate was 500K). 3.0 gives safe slack, and the cache budget easily
// absorbs it. kMinCap is the smallest non-zero cap we allow — TPC-C
// mixes that don't touch a growing table at all (e.g., tpccp = 100% Payment
// has zero NO/O/OL touches) still need a non-zero cap because cap=0
// triggers cacheCapacity<X>()'s "fall back to full table size" behavior in
// tpcc_config.h, which would over-allocate and OOM. A small minimum is
// safe because the stager allocations scale linearly with cap.
inline size_t derive_cap(const PerTableDemand& d)
{
    constexpr double kSafetyFactor       = 1.5;
    constexpr double kInsertHeadroomFactor = 3.0;  // eviction-churn + realized-insert-variance margin (see above)
    constexpr size_t kMinCap             = 1024;
    const double per_epoch_demand = d.reads + d.writes + d.inserts;
    const double working_set      = per_epoch_demand + d.reach_back * d.inserts;
    const double from_workload    = working_set * kSafetyFactor;
    const double from_inserts     = d.inserts * kInsertHeadroomFactor;
    const size_t computed         = static_cast<size_t>(std::max(from_workload, from_inserts));
    return std::max(computed, kMinCap);
}

// Top-level entry point; tpcc.cpp calls this at startup in hybrid_staging
// mode, after planner+index allocations and before record-array allocations.
inline void computeAndApplyTpccStagingCapacitiesWorkloadAware(
    TpccConfig& cfg,
    const TpccRecordBytes& bytes,
    size_t free_bytes,
    size_t total_bytes)
{
    auto& logger = Logger::GetInstance();
    logger.Info("[STAGING-AUTOSIZE-WL] workload-aware autosizer engaged");

    // 1. HBM budget.
    size_t total_budget;
    if (cfg.hybrid_hbm_budget_bytes > 0) {
        total_budget = cfg.hybrid_hbm_budget_bytes;
    } else {
        if (free_bytes <= cfg.hybrid_hbm_reserve_bytes) {
            throw std::runtime_error(
                "[STAGING-AUTOSIZE-WL] HBM exhausted: free=" +
                std::to_string(free_bytes / (1024 * 1024)) + " MB <= reserve=" +
                std::to_string(cfg.hybrid_hbm_reserve_bytes / (1024 * 1024)) + " MB");
        }
        total_budget = free_bytes - cfg.hybrid_hbm_reserve_bytes;
    }
    logger.Info("[STAGING-AUTOSIZE-WL] HBM cache budget = {} MB", total_budget / (1024 * 1024));

    // 2. Always-resident tables stay at full size. (Workload-aware sizing for
    //    these would also produce values close to full-size for typical
    //    mixes, so save the implementation cost.)
    constexpr size_t kPerSlotScratchBytes = 64;
    auto bytes_for = [&](size_t records, size_t per_record) {
        return records * (per_record + kPerSlotScratchBytes);
    };

    const size_t w_bytes = bytes_for(cfg.cacheCapacityWarehouse(), bytes.warehouse);
    const size_t d_bytes = bytes_for(cfg.cacheCapacityDistrict(),  bytes.district);
    const size_t i_bytes = bytes_for(cfg.cacheCapacityItem(),      bytes.item);
    const size_t c_bytes = bytes_for(cfg.cacheCapacityCustomer(),  bytes.customer);
    const size_t s_bytes = bytes_for(cfg.cacheCapacityStock(),     bytes.stock);
    const size_t static_cost = w_bytes + d_bytes + i_bytes + c_bytes + s_bytes;

    if (static_cost >= total_budget) {
        throw std::runtime_error(
            "[STAGING-AUTOSIZE-WL] Static tables alone (" +
            std::to_string(static_cost / (1024 * 1024)) + " MB) exceed HBM budget (" +
            std::to_string(total_budget / (1024 * 1024)) + " MB).");
    }
    const size_t growing_budget = total_budget - static_cost;
    logger.Info("[STAGING-AUTOSIZE-WL] static (W+D+I+C+S) cost {} MB, growing budget {} MB",
                static_cost / (1024 * 1024), growing_budget / (1024 * 1024));

    // 3. Workload-derived caps for the 3 growing tables.
    auto d_no = compute_per_table_demand(cfg, kT_NO);
    auto d_o  = compute_per_table_demand(cfg, kT_O);
    auto d_ol = compute_per_table_demand(cfg, kT_OL);

    size_t no_cap = derive_cap(d_no);
    size_t o_cap  = derive_cap(d_o);
    size_t ol_cap = derive_cap(d_ol);

    // Cap each at its full table size (no benefit caching beyond the table).
    no_cap = std::min(no_cap, cfg.newOrderTableSize());
    o_cap  = std::min(o_cap,  cfg.orderTableSize());
    ol_cap = std::min(ol_cap, cfg.orderLineTableSize());

    auto fmt_demand = [](const PerTableDemand& d) {
        return std::string("reads=")    + std::to_string(static_cast<size_t>(d.reads)) +
               " writes="    + std::to_string(static_cast<size_t>(d.writes)) +
               " inserts="   + std::to_string(static_cast<size_t>(d.inserts)) +
               " reach_back=" + std::to_string(d.reach_back);
    };
    logger.Info("[STAGING-AUTOSIZE-WL] NO demand: {} -> cap {}",  fmt_demand(d_no), no_cap);
    logger.Info("[STAGING-AUTOSIZE-WL] O  demand: {} -> cap {}",  fmt_demand(d_o),  o_cap);
    logger.Info("[STAGING-AUTOSIZE-WL] OL demand: {} -> cap {}",  fmt_demand(d_ol), ol_cap);

    // 4. Check vs HBM budget. If we exceed, scale all proportionally.
    const size_t no_bytes_req = bytes_for(no_cap, bytes.new_order);
    const size_t o_bytes_req  = bytes_for(o_cap,  bytes.order);
    const size_t ol_bytes_req = bytes_for(ol_cap, bytes.order_line);
    const size_t total_req    = no_bytes_req + o_bytes_req + ol_bytes_req;

    if (total_req > growing_budget) {
        const double scale = static_cast<double>(growing_budget) / static_cast<double>(total_req);
        no_cap = static_cast<size_t>(no_cap * scale);
        o_cap  = static_cast<size_t>(o_cap  * scale);
        ol_cap = static_cast<size_t>(ol_cap * scale);
        logger.Warn("[STAGING-AUTOSIZE-WL] workload demand ({} MB) exceeds HBM budget ({} MB); "
                    "scaling all caps by {:.3f}",
                    total_req / (1024 * 1024), growing_budget / (1024 * 1024), scale);
    } else {
        const size_t free_left = growing_budget - total_req;
        logger.Info("[STAGING-AUTOSIZE-WL] workload fits: req={} MB / budget={} MB ({} MB unused)",
                    total_req / (1024 * 1024), growing_budget / (1024 * 1024),
                    free_left / (1024 * 1024));
    }

    // 5. Apply only when CLI override hasn't already set the cap.
    if (cfg.cache_capacity_new_order  == 0) cfg.cache_capacity_new_order  = no_cap;
    if (cfg.cache_capacity_order      == 0) cfg.cache_capacity_order      = o_cap;
    if (cfg.cache_capacity_order_line == 0) cfg.cache_capacity_order_line = ol_cap;

    auto pct = [](size_t cache, size_t full) -> double {
        return full > 0 ? 100.0 * static_cast<double>(cache) / static_cast<double>(full) : 0.0;
    };
    logger.Info(
        "[STAGING-AUTOSIZE-WL] Final caps: NO={} ({:.2f}% of {}); O={} ({:.2f}% of {}); OL={} ({:.2f}% of {})",
        cfg.cache_capacity_new_order,  pct(cfg.cache_capacity_new_order,  cfg.newOrderTableSize()),  cfg.newOrderTableSize(),
        cfg.cache_capacity_order,      pct(cfg.cache_capacity_order,      cfg.orderTableSize()),      cfg.orderTableSize(),
        cfg.cache_capacity_order_line, pct(cfg.cache_capacity_order_line, cfg.orderLineTableSize()), cfg.orderLineTableSize());
}

} // namespace epic::tpcc

#endif // TPCC_WORKLOAD_AWARE_CAPACITIES_H
