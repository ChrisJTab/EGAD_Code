//
// recovery_meta.h -- crash-atomic durable recovery marker, shared by the
// TPC-C and YCSB recovery paths.
//
// The marker is a set of cursor snapshots plus the crash epoch, living in
// an mmap'd durable file (MallocDurable). A recover process trusts it to
// name the rollback target, so a half-written marker is a corrupt one:
// the epoch and the cursors must always agree. The update runs on the
// main thread, but a real GPU fault terminates the process from whichever
// thread touches CUDA next (typically a flush worker mid-copy), so the
// update sequence can be cut at any store boundary.
//
// A two-bank publish makes every cut coherent. The writer fills the
// inactive cursor bank with plain stores, then publishes with one aligned
// 64-bit release store carrying a validity tag, a generation (whose
// parity selects the live bank), and the crash epoch. A reader sees
// either the previous complete snapshot or the new complete snapshot,
// never a mix. The all-zero word of a freshly provisioned durable file
// reads as "never published".
//

#ifndef EPIC_BENCHMARKS_RECOVERY_META_H
#define EPIC_BENCHMARKS_RECOVERY_META_H

#include <cstddef>
#include <cstdint>

namespace epic {

// Validity tag in the publish word's top 16 bits. A publish word is
// trusted only when the tag matches; a zeroed fresh file never does.
inline constexpr uint16_t kRecoveryMetaTag = 0xE6ADu;

template <typename CursorBankT>
struct BankedRecoveryMeta
{
    CursorBankT banks[2];
    // Publish word: [tag:16 | generation:16 | crash_epoch:32]. Generation
    // parity selects the live bank; 8-byte alignment makes the release
    // store below a single machine store.
    alignas(8) uint64_t publish;

    // Reader (recover process). Returns false when nothing was ever
    // published; crash_epoch and bank are left untouched in that case.
    bool read(uint32_t& crash_epoch, const CursorBankT*& bank) const
    {
        const uint64_t w = __atomic_load_n(&publish, __ATOMIC_ACQUIRE);
        if (static_cast<uint16_t>(w >> 48) != kRecoveryMetaTag) return false;
        crash_epoch = static_cast<uint32_t>(w);
        bank = &banks[(w >> 32) & 1u];
        return true;
    }

    // Writer, step 1: fill the inactive bank and return its publish word.
    // fill(new_bank, old_bank) writes the shifted cursor history; old_bank
    // is null on the first bump, matching a zeroed fresh file.
    template <typename FillFn>
    uint64_t prepare(uint32_t crash_epoch, FillFn fill)
    {
        const uint64_t w = publish;
        const bool valid = static_cast<uint16_t>(w >> 48) == kRecoveryMetaTag;
        const uint16_t gen =
            valid ? static_cast<uint16_t>(static_cast<uint16_t>(w >> 32) + 1u)
                  : uint16_t{1};
        fill(banks[gen & 1u], valid ? &banks[(gen & 1u) ^ 1u] : nullptr);
        return (static_cast<uint64_t>(kRecoveryMetaTag) << 48)
             | (static_cast<uint64_t>(gen) << 32)
             | static_cast<uint64_t>(crash_epoch);
    }

    // Writer, step 2: the single atomic publish. Split from prepare so the
    // validation build can inject a process death between the two (the
    // torn-marker probe); production callers invoke them back to back.
    void publishPrepared(uint64_t word)
    {
        __atomic_store_n(&publish, word, __ATOMIC_RELEASE);
    }
};

} // namespace epic

#endif // EPIC_BENCHMARKS_RECOVERY_META_H
