//
// Created by Shujian Qian on 2023-08-08.
//

#include <iostream>
#include <memory>
#include <getopt.h>

#include "gpu_execution_planner.h"
#include "gpu_allocator.h"
#include <benchmarks/benchmark.h>
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"

#include "benchmarks/tpcc.h"
#include "gacco/benchmarks/tpcc.h"
#include "gacco/benchmarks/ycsb.h"

static constexpr struct option long_options[] = {{"benchmark", required_argument, nullptr, 'b'},
    {"database", required_argument, nullptr, 'd'}, {"num_warehouses", required_argument, nullptr, 'w'},
    {"skew_factor", required_argument, nullptr, 'a'}, {"fullread", required_argument, nullptr, 'r'},
    {"cpu_exec_num_threads", required_argument, nullptr, 'c'}, {"num_epochs", required_argument, nullptr, 'e'},
    {"num_txns", required_argument, nullptr, 's'}, {"split_fields", required_argument, nullptr, 'f'},
    {"commutative_ops", required_argument, nullptr, 'm'}, {"num_records", required_argument, nullptr, 'n'},
    {"starting_num_records", required_argument, nullptr, 'N'},
    {"exec_device", required_argument, nullptr, 'x'}, {"exec_mode", required_argument, nullptr, 'y'},
    {"overlap_flush", required_argument, nullptr, 'z'},
    // Long-only options (val >= 256 keeps them out of the short-option space).
    {"verify_tpcc",            no_argument,       nullptr, 300},
    {"cache_capacity_orderline", required_argument, nullptr, 301},
    // Autosizer (TPC-C hybrid_staging): see tpcc_staging_capacities.h.
    {"hybrid_hbm_reserve_gb",  required_argument, nullptr, 302},
    {"hybrid_hbm_budget_gb",   required_argument, nullptr, 303},
    // Per-table cache_capacity overrides for the 4 tables not exposed as 301.
    // OL is 301 above. W/D/I are tiny (<50 MB total at any -w) and not exposed.
    {"cache_capacity_customer",  required_argument, nullptr, 304},
    {"cache_capacity_stock",     required_argument, nullptr, 305},
    {"cache_capacity_new_order", required_argument, nullptr, 306},
    {"cache_capacity_order",     required_argument, nullptr, 307},
    // Recovery (GPU-crash fault tolerance), opt-in. --durable_store <dir> creates
    // the durable Primary Store; --recover re-maps it and replays after a crash.
    {"durable_store", required_argument, nullptr, 308},
    {"recover",       no_argument,       nullptr, 309},
    {nullptr, 0, nullptr, 0}};

static char optstring[] = "b:d:w:a:r:c:e:s:f:m:n:N:x:y:z:";

int main(int argc, char **argv)
{

    epic::tpcc::TpccConfig tpcc_config;
    epic::ycsb::YcsbConfig ycsb_config;

    int retval = 0;
    char *end_char = nullptr;
    std::string bench = "tpcc";
    std::string db = "epic";
    bool commutative_ops = false;
    // Tracks whether the user passed -N. When set, -n must not overwrite
    // starting_num_records with the default 80% rule.
    bool starting_num_records_set = false;
    std::string durable_dir;    // --durable_store <dir>: enable recovery (opt-in, default off)
    bool recover_mode = false;  // --recover: re-map the durable store and replay after a crash
    while ((retval = getopt_long(argc, argv, optstring, long_options, nullptr)) != -1)
    {
        switch (retval)
        {
        case 'b':
            bench = std::string(optarg);
            if (bench == "ycsba")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 50, 0, 0}; // the transaction mix for the Ycsb benchmark
            }
            else if (bench == "ycsbb")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {95, 5, 0, 0};
            }
            else if (bench == "ycsbc")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {100, 0, 0, 0};
            }
            else if (bench == "ycsbf")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 0, 50, 0};// is supposed to be 50, 0, 50, 0
            }
            else if (bench == "ycsbi")
            {
                // Insert-heavy mix for the EGAD insert-path prototype:
                // 20% reads, 0% writes, 30% rmw, 50% inserts. Mirrors the burst
                // behaviour of TPC-C NewOrder + Delivery (every NewOrder
                // produces 7-17 inserts; Delivery reads many recently-inserted
                // OrderLines).
                //
                // The recent_read_bias mirrors Delivery's read pattern: 30% of
                // non-insert ops draw from the last 100k inserted CRIDs, the
                // rest follow the Zipfian. This exercises the hybrid cache's
                // cross-epoch retention behaviour.
                bench = "ycsb";
                ycsb_config.txn_mix = {20, 0, 30, 50};
                ycsb_config.recent_read_bias = 0.3;
                ycsb_config.recent_window_size = 100'000;
            }
            else if (bench == "tpccn")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {100, 0, 0, 0, 0};
            }
            else if (bench == "tpccp")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {0, 100, 0, 0, 0};
            }
            else if (bench == "tpcc")
            {
                tpcc_config.txn_mix = {50, 50, 0, 0, 0};
            }
            else if (bench == "tpccfull")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {45, 43, 4, 4, 4};
            }
            else if (bench == "tpccdeck")
            {
                // TPC-C v5.11 Clause 5.2.4.2: deck-based selection with the
                // spec's prescribed 10/10/1/1/1 composition (10 NewOrder,
                // 10 Payment, 1 each of OrderStatus, Delivery, StockLevel).
                // The 10:1 NewOrder:Delivery ratio exactly balances order
                // creation (10 NewOrders create 10 orders) against delivery
                // (1 Delivery delivers 10 orders, one per district), so the
                // undelivered queue stays at 9000W indefinitely. Implemented
                // via ratio-weighted random rather than a true 23-card
                // shuffle; the difference is small Bernoulli noise that
                // averages out over an epoch.
                bench = "tpcc";
                tpcc_config.txn_mix = {10, 10, 1, 1, 1};
            }
            else
            {

                throw std::runtime_error("Invalid benchmark name");
            }
            break;
        case 'd':
            db = std::string(optarg);
            if (db != "epic" && db != "gacco")
            {
                throw std::runtime_error("Invalid database name");
            }
            break;
        case 'w':
            errno = 0;
            tpcc_config.num_warehouses = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of warehouses");
            }
            break;
        case 'a':
            errno = 0;
            ycsb_config.skew_factor = strtod(optarg, &end_char);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid skew factor");
            }
            break;
        case 'r':
            if (std::string(optarg) == "true")
            {
                ycsb_config.full_record_read = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.full_record_read = false;
            }
            else
            {
                throw std::runtime_error("Invalid full record read");
            }
            break;
        case 'c':
            errno = 0;
            ycsb_config.cpu_exec_num_threads = strtoul(optarg, &end_char, 0);
            tpcc_config.cpu_exec_num_threads = ycsb_config.cpu_exec_num_threads;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of CPU execution threads");
            }
            break;
        case 'e':
            errno = 0;
            tpcc_config.epochs = strtoul(optarg, &end_char, 0);
            ycsb_config.epochs = tpcc_config.epochs;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of epochs");
            }
            break;
        case 's':
            errno = 0;
            tpcc_config.num_txns = strtoul(optarg, &end_char, 0);
            ycsb_config.num_txns = tpcc_config.num_txns;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of transactions");
            }
            break;
        case 'f':
            if (std::string(optarg) == "true")
            {
                ycsb_config.split_field = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.split_field = false;
            }
            else
            {
                throw std::runtime_error("Invalid split fields");
            }
            break;
        case 'm':
            if (std::string(optarg) == "true")
            {
                commutative_ops = true;
            }
            else if (std::string(optarg) == "false")
            {
                commutative_ops = false;
            }
            else
            {
                throw std::runtime_error("Invalid commutative ops");
            }
            break;
        case 'n':
            errno = 0;
            ycsb_config.num_records = strtoul(optarg, &end_char, 0);
            if (!starting_num_records_set) {
                // Default split: 80% pre-existing, 20% headroom for inserts.
                // Overridden by -N (regardless of flag order) below.
                ycsb_config.starting_num_records = ycsb_config.num_records * 80 / 100;
            }
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of records");
            }
            break;
        case 'N':
            // Explicit override of starting_num_records (the count of CRIDs
            // pre-loaded by loadInitialData; the remaining num_records -
            // starting_num_records CRIDs are insert headroom on the GPU
            // free-list). Defaults to 80 % of -n; set to a smaller value for
            // long ycsbi runs where 20 % headroom isn't enough.
            errno = 0;
            ycsb_config.starting_num_records = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid starting number of records");
            }
            starting_num_records_set = true;
            break;
        case 'x':
            if (std::string(optarg) == "cpu")
            {
                tpcc_config.execution_device = epic::DeviceType::CPU;
                ycsb_config.execution_device = epic::DeviceType::CPU;
            }
            else if (std::string(optarg) == "gpu")
            {
                tpcc_config.execution_device = epic::DeviceType::GPU;
                ycsb_config.execution_device = epic::DeviceType::GPU;
            }
            else
            {
                throw std::runtime_error("Invalid execution device");
            }
            break;
        case 'y':
            if (std::string(optarg) == "cpu_only")
            {
                tpcc_config.execution_mode = epic::ExecMode::CPU_ONLY;
                ycsb_config.execution_mode = epic::ExecMode::CPU_ONLY;
            }
            else if (std::string(optarg) == "gpu_only")
            {
                tpcc_config.execution_mode = epic::ExecMode::GPU_ONLY;
                ycsb_config.execution_mode = epic::ExecMode::GPU_ONLY;
            }
            else if (std::string(optarg) == "hybrid_staging")
            {
                tpcc_config.execution_mode = epic::ExecMode::HYBRID_STAGING;
                ycsb_config.execution_mode = epic::ExecMode::HYBRID_STAGING;
            }
            else
            {
                throw std::runtime_error("Invalid execution mode");
            }
            break;
        case 'z':
            if (std::string(optarg) == "true")
            {
                tpcc_config.overlap_flush = true;
                ycsb_config.overlap_flush = true;
            }
            else if (std::string(optarg) == "false")
            {
                tpcc_config.overlap_flush = false;
                ycsb_config.overlap_flush = false;
            }
            else
            {
                throw std::runtime_error("Invalid overlap flush");
            }
            break;
        case 300: // --verify_tpcc — insert-CRID verifier
            // No-argument flag; toggles tpcc_config.verify_tpcc on. Default
            // off so benchmark runs are not affected by the verifier scan
            // cost (~tens of ms across the 3 growing-table CPU primary
            // stores). Mirror of the YCSB ycsb.cpp::verifyInsertedRecords
            // pattern, flag-gated for the same reason.
            tpcc_config.verify_tpcc = true;
            break;
        case 301: // --cache_capacity_orderline N — undersize OrderLine cache
            // OrderLine is the dominant memory consumer; undersizing its
            // cache forces FIFO eviction without touching the other 7
            // tables' (always-resident or comfortably-cached) layouts,
            // which makes it the forcing function for eviction-heavy runs.
            tpcc_config.cache_capacity_order_line = std::strtoull(optarg, &end_char, 10);
            break;
        case 302: // --hybrid_hbm_reserve_gb N — autosizer reserve, in GiB.
            tpcc_config.hybrid_hbm_reserve_bytes =
                std::strtoull(optarg, &end_char, 10) * 1024ULL * 1024ULL * 1024ULL;
            break;
        case 303: // --hybrid_hbm_budget_gb N — autosizer hard cap, in GiB.
            // 0 = use cudaMemGetInfo to determine. Set explicitly to limit
            // the cache budget for sweep experiments.
            tpcc_config.hybrid_hbm_budget_bytes =
                std::strtoull(optarg, &end_char, 10) * 1024ULL * 1024ULL * 1024ULL;
            break;
        case 304: // --cache_capacity_customer N
            tpcc_config.cache_capacity_customer = std::strtoull(optarg, &end_char, 10);
            break;
        case 305: // --cache_capacity_stock N
            tpcc_config.cache_capacity_stock = std::strtoull(optarg, &end_char, 10);
            break;
        case 306: // --cache_capacity_new_order N
            tpcc_config.cache_capacity_new_order = std::strtoull(optarg, &end_char, 10);
            break;
        case 307: // --cache_capacity_order N
            tpcc_config.cache_capacity_order = std::strtoull(optarg, &end_char, 10);
            break;
        case 308: // --durable_store <dir>
            durable_dir = optarg;
            break;
        case 309: // --recover
            recover_mode = true;
            break;
        default:
            throw std::runtime_error("Invalid option");
        }
    }

    // Recovery is opt-in via --durable_store (+ --recover to reload after a crash),
    // off by default so the standard build reproduces the non-recovery headline.
    // Layered over the durable-store mechanism: --durable_store creates the store,
    // --recover re-maps it (recover takes precedence in MallocDurable). Set before
    // the benchmark is constructed, since its allocators read these at startup.
    if (!durable_dir.empty()) {
        if (recover_mode) {
            setenv("EPIC_RECOVER_FROM", durable_dir.c_str(), 1);
        } else {
            setenv("EPIC_DURABLE_STORE", durable_dir.c_str(), 1);
        }
    } else if (recover_mode) {
        throw std::runtime_error("--recover requires --durable_store <dir>");
    }

    /* this is a hack to run gacco NewOrder without holding locks on warehouse... */
    if (commutative_ops)
    {
        tpcc_config.gacco_use_atomic = true;
        tpcc_config.gacco_tpcc_stock_use_atomic = true;
    }
    else if (tpcc_config.txn_mix.new_order > 0)
    {
        tpcc_config.gacco_use_atomic = true;
        tpcc_config.gacco_tpcc_stock_use_atomic = false;
    } else {
        tpcc_config.gacco_use_atomic = false;
        tpcc_config.gacco_tpcc_stock_use_atomic = false;
    }

    std::unique_ptr<epic::Benchmark> benchmark;
    if (bench == "tpcc")
    {
        if (db == "epic")
        {
            benchmark = std::make_unique<epic::tpcc::TpccDb>(tpcc_config);
        }
        else if (db == "gacco")
        {
            benchmark = std::make_unique<gacco::tpcc::TpccDb>(tpcc_config);
        }
        else
        {
            throw std::runtime_error("Invalid database name");
        }
    }
    else if (bench == "ycsb")
    {
        if (db == "epic")
        {
            benchmark = std::make_unique<epic::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else if (db == "gacco")
        {
            benchmark = std::make_unique<gacco::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else
        {
            throw std::runtime_error("Invalid database name");
        }
    }
    benchmark->loadInitialData();
    benchmark->generateTxns();
    benchmark->runBenchmark();

    return 0;
}