//
// ycsb_timeline_oracle.h -- host reference model of the serial-order
// semantics of inserts and deletes (validation build only).
//
// Replays every epoch's transactions in serial order over a key -> CRID map
// with the index's minting rule (the life-creating inserts of an epoch get
// base + free_start + j in slot order) and compares the record each
// operation must resolve to with the record the GPU indexing resolved.
// Independent of the GPU code by construction: a plain sequential loop.
//

#ifndef EPIC_BENCHMARKS_YCSB_TIMELINE_ORACLE_H
#define EPIC_BENCHMARKS_YCSB_TIMELINE_ORACLE_H

#ifdef EGAD_VALIDATION

#include <cstdint>
#include <vector>
#include <utility>

#include <ankerl/unordered_dense.h>

#include <benchmarks/ycsb_config.h>
#include <benchmarks/ycsb_txn.h>
#include <txn.h>

namespace epic::ycsb {

class YcsbTimelineOracle
{
public:
    explicit YcsbTimelineOracle(const YcsbConfig& config);

    // Replay one epoch. params_host points at the host copy of the GPU's
    // resolved params (BaseTxnSize<YcsbTxnParam>::value bytes per
    // transaction); pass nullptr to replay without comparing (the epochs a
    // recovering process does not run). Logs one [TIMELINE-ORACLE] line
    // and returns the number of mismatching operations.
    uint32_t replayEpoch(uint32_t epoch_id, TxnArray<YcsbTxn>& inputs, const uint8_t* params_host);

    // Restart the model from the initial population (a recovering process
    // replays the epochs before its resume point silently).
    void reset();

    const ankerl::unordered_dense::map<uint32_t, uint32_t>& live() const { return live_; }
    // Keys that were deleted and are not live now (bounded sample).
    std::vector<uint32_t> deadSample(size_t max_n) const;
    // Every record whose life ended, with the epoch of the delete that
    // ended it: no write may reach such a record in a later epoch.
    const std::vector<std::pair<uint32_t, uint32_t>>& endedLives() const { return ended_; }

    uint64_t totalAbsentOps() const { return absent_ops_; }
    uint64_t totalMinted() const { return minted_; }
    uint64_t totalEffectiveDeletes() const { return effective_deletes_; }
    uint64_t totalReinserts() const { return reinserts_; }
    uint64_t totalWriteInserts() const { return write_inserts_; }
    uint64_t totalMismatches() const { return mismatches_; }

private:
    YcsbConfig config_;
    uint32_t free_start_ = 0;
    ankerl::unordered_dense::map<uint32_t, uint32_t> live_;
    ankerl::unordered_dense::set<uint32_t> dead_;
    std::vector<std::pair<uint32_t, uint32_t>> ended_;   // (crid, epoch of its delete)
    uint64_t absent_ops_ = 0, minted_ = 0, effective_deletes_ = 0, reinserts_ = 0, write_inserts_ = 0, mismatches_ = 0;
};

} // namespace epic::ycsb

#endif // EGAD_VALIDATION

#endif // EPIC_BENCHMARKS_YCSB_TIMELINE_ORACLE_H
