//
// Created by Shujian Qian on 2023-11-05.
//

#ifndef EPIC_BENCHMARKS_TPCC_TXN_GEN_H
#define EPIC_BENCHMARKS_TPCC_TXN_GEN_H

#include <cstdint>
#include <cstdlib>
#include <random>
#include <cassert>

#include "tpcc_config.h"
#include <benchmarks/tpcc_txn.h>

namespace epic::tpcc {

// Deterministic seeding for replay-correctness paired runs. When
// EPIC_TPCC_SEED is set, returns a seed derived from it mixed with `salt`
// (splitmix64) so distinct RNGs stay independent yet reproducible; otherwise
// returns fresh entropy, preserving the original non-deterministic behavior.
// Mirrors the YCSB EPIC_YCSB_SEED mechanism.
inline uint64_t tpccSeedOrEntropy(uint64_t salt)
{
#ifdef EGAD_VALIDATION
    // Deterministic seed for replay-correctness validation (paired same-seed runs).
    const char *s = std::getenv("EPIC_TPCC_SEED");
    if (!s) return static_cast<uint64_t>(std::random_device{}());
    uint64_t z = std::strtoull(s, nullptr, 10) + salt * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
#else
    (void)salt;
    return static_cast<uint64_t>(std::random_device{}());
#endif
}

class TpccNuRand
{
public:
    uint32_t C, x, y;
    std::uniform_int_distribution<uint32_t> dist1, dist2;
    TpccNuRand(uint32_t A, uint32_t x, uint32_t y)
        : C(static_cast<uint32_t>(tpccSeedOrEntropy(
              (static_cast<uint64_t>(A) << 40) ^ (static_cast<uint64_t>(x) << 20) ^ static_cast<uint64_t>(y))))
        , x(x)
        , y(y)
        , dist1(0, A)
        , dist2(x, y)
    {
        assert((x == 0 && y == 999 && A == 255) || (x == 1 && y == 3000 && A == 1023) ||
               (x == 1 && y == 100000 && A == 8191));
    }

    template<typename RandomEngine>
    inline uint32_t operator()(RandomEngine &gen)
    {
        uint32_t rand = (dist1(gen) | dist2(gen)) + C;
        return (rand % (y - x + 1)) + x;
    }
};

class TpccOIdGenerator
{
    uint32_t next_oid_counter[256][20]{};
    uint32_t next_deliver_oid_counter[256][20]{};
public:
    explicit TpccOIdGenerator(uint32_t initial_orders_per_district = 3000, uint32_t initial_delivered_orders_per_district = 2100)
    {
        for (size_t i = 0; i < 256; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                next_oid_counter[i][j] = initial_orders_per_district;
                next_deliver_oid_counter[i][j] = initial_delivered_orders_per_district;
            }
        }
    }
    inline uint32_t getOId(uint32_t timestamp)
    {
        constexpr uint32_t loader_max_order_id = 1'000'000; /* max number of orders used by loader */
        return timestamp + loader_max_order_id;
    }

    inline uint32_t nextOId(uint32_t w_id, uint32_t d_id) {
        return ++next_oid_counter[w_id][d_id];
    }

    inline uint32_t currOId(uint32_t w_id, uint32_t d_id) {
        return next_oid_counter[w_id][d_id];
    }

    // Real TPC-C Delivery (Clause 2.7) delivers the oldest undelivered order of
    // EACH district independently. Districts diverge in order count because
    // NewOrder picks a random district, so a single per-warehouse counter would
    // deliver a phantom (non-existent) order in the slowest district. This
    // advances and returns the district's own next deliver o_id only if an
    // undelivered order actually exists (next_deliver < next_oid). Otherwise it
    // does NOT advance and returns 0, the "nothing to deliver" sentinel (o_id is
    // always >= 1 for real orders, so 0 is unambiguous).
    inline uint32_t nextDeliverOId(uint32_t w_id, uint32_t d_id)
    {
        if (next_deliver_oid_counter[w_id][d_id] + 1 < next_oid_counter[w_id][d_id])
        {
            return ++next_deliver_oid_counter[w_id][d_id];
        }
        return 0;
    }
};

struct TpccTxnGenerator
{
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint32_t> txn_type_dist, w_id_dist, d_id_dist, num_item_dist, percentage_gen,
        order_quantity_dist, payment_amount_dist, carrier_id_dist, threshold_dist;
    TpccNuRand c_id_dist, i_id_dist;
    TpccOIdGenerator o_id_gen;

    const TpccConfig config;

    TpccTxnGenerator(const TpccConfig &config)
        : gen(tpccSeedOrEntropy(0xACE1ull))
        // Range is [0, sum_of_weights). Standard mixes sum to 100 (canonical
        // percentages). The deck mix (Clause 5.2.4.2) sums to 23. getTxnType
        // walks the cumulative weight sum to pick.
        , txn_type_dist(0, config.txn_mix.new_order + config.txn_mix.payment
                         + config.txn_mix.order_status + config.txn_mix.delivery
                         + config.txn_mix.stock_level - 1)
        , w_id_dist(1, config.num_warehouses)
        , d_id_dist(1, 10)
        , c_id_dist(1023, 1, 3000)
        , i_id_dist(8191, 1, 100'000)
        , num_item_dist(5, 15)
        , percentage_gen(1, 100)
        , order_quantity_dist(1, 10)
        , payment_amount_dist(1, 5000)
        , carrier_id_dist(1, 10)
        , threshold_dist(10, 20)
        , config(config)
    {}

    TpccTxnType getTxnType();
    void generateTxn(NewOrderTxnInput<FixedSizeTxn> *txn, uint32_t timestamp);
    void generateTxn(PaymentTxnInput *txn, uint32_t timestamp);
    void generateTxn(OrderStatusTxnInput *txn, uint32_t timestamp);
    void generateTxn(DeliveryTxnInput *txn, uint32_t timestamp);
    void generateTxn(StockLevelTxnInput *txn, uint32_t timestamp);
    void generateTxn(BaseTxn *txn, uint32_t timestamp);
    void generateTxn(TpccTxnType txn_type, BaseTxn *txn, uint32_t timestamp);
};

}

#endif // EPIC_BENCHMARKS_TPCC_TXN_GEN_H
