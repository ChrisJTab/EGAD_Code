//
// ycsb_timeline_oracle.cpp -- see the header. Validation build only.
//

#ifdef EGAD_VALIDATION

#include <benchmarks/ycsb_timeline_oracle.h>

#include <util_log.h>

namespace epic::ycsb {

namespace {
constexpr uint32_t kAbsent = 0xffffffffu;
}

YcsbTimelineOracle::YcsbTimelineOracle(const YcsbConfig& config)
    : config_(config)
{
    reset();
}

void YcsbTimelineOracle::reset()
{
    free_start_ = 0;
    live_.clear();
    dead_.clear();
    live_.reserve(config_.num_records);
    for (uint32_t k = 0; k < config_.starting_num_records; ++k) live_.emplace(k, k);
}

uint32_t YcsbTimelineOracle::replayEpoch(uint32_t epoch_id, TxnArray<YcsbTxn>& inputs, const uint8_t* params_host)
{
    auto& logger = Logger::GetInstance();
    const uint32_t base = static_cast<uint32_t>(config_.starting_num_records);
    const size_t param_stride = BaseTxnSize<YcsbTxnParam>::value;
    uint32_t epoch_mismatches = 0, epoch_absent = 0, epoch_minted = 0, epoch_edel = 0, epoch_reins = 0;
    uint32_t reported = 0;
    for (uint32_t t = 0; t < config_.num_txns; ++t) {
        const YcsbTxn* txn = reinterpret_cast<const YcsbTxn*>(inputs.getTxn(t)->data);
        const YcsbTxnParam* got = params_host
            ? reinterpret_cast<const YcsbTxnParam*>(reinterpret_cast<const BaseTxn*>(params_host + t * param_stride)->data)
            : nullptr;
        for (uint32_t i = 0; i < config_.num_ops_per_txn; ++i) {
            const uint32_t key = txn->keys[i];
            const YcsbOpType op = txn->ops[i];
            auto it = live_.find(key);
            uint32_t expected = kAbsent;
            switch (op) {
            case YcsbOpType::INSERT:
                if (it == live_.end()) {
                    expected = base + free_start_ + epoch_minted;   // slot-ordered minting
                    ++epoch_minted;
                    if (dead_.erase(key)) ++epoch_reins;
                    live_.emplace(key, expected);
                } else {
                    expected = it->second;                        // insert of a live key: a write
                    ++write_inserts_;
                }
                break;
            case YcsbOpType::DELETE:
                if (it != live_.end()) {
                    expected = it->second;                        // the record the delete ends
                    live_.erase(it);
                    dead_.insert(key);
                    ++epoch_edel;
                } else {
                    expected = kAbsent;
                    ++epoch_absent;
                }
                break;
            default:
                expected = (it != live_.end()) ? it->second : kAbsent;
                if (expected == kAbsent) ++epoch_absent;
                break;
            }
            if (got && got->record_ids[i] != expected) {
                ++epoch_mismatches;
                if (reported < 20) {
                    ++reported;
                    logger.Error("[TIMELINE-ORACLE] mismatch epoch={} txn={} op={} type={} key={} expected={} got={}",
                                 epoch_id, t, i, static_cast<int>(op), key,
                                 static_cast<int64_t>(expected == kAbsent ? -1 : static_cast<int64_t>(expected)),
                                 static_cast<int64_t>(got->record_ids[i] == kAbsent ? -1 : static_cast<int64_t>(got->record_ids[i])));
                }
            }
        }
    }
    free_start_ += epoch_minted;
    absent_ops_ += epoch_absent; minted_ += epoch_minted; effective_deletes_ += epoch_edel; reinserts_ += epoch_reins;
    if (params_host) {
        mismatches_ += epoch_mismatches;
        logger.Info("[TIMELINE-ORACLE] epoch={} ops={} mismatches={} absent={} minted={} edel={} reinserts={} live={}",
                    epoch_id, static_cast<uint64_t>(config_.num_txns) * config_.num_ops_per_txn, epoch_mismatches,
                    epoch_absent, epoch_minted, epoch_edel, epoch_reins, live_.size());
    }
    return epoch_mismatches;
}

std::vector<uint32_t> YcsbTimelineOracle::deadSample(size_t max_n) const
{
    std::vector<uint32_t> out;
    out.reserve(std::min(max_n, dead_.size()));
    for (uint32_t k : dead_) {
        if (out.size() >= max_n) break;
        out.push_back(k);
    }
    return out;
}

} // namespace epic::ycsb

#endif // EGAD_VALIDATION
