//
// recovery_verify.h -- shared, byte-exact recovery/determinism VERIFICATION
// primitives used by tpcc_recovery.cpp and ycsb_recovery.cpp.
//
// This header holds ONLY the verification building blocks that are genuinely
// identical (byte-for-byte) across the TPC-C and YCSB recovery paths, factored
// out to remove the duplicated inline bodies. Everything in here must reproduce
// the EXACT same FNV-1a constants, slot-sort direction, byte ranges, and combine
// operations as the previous inline code, because the canonical recovery GO/NO-GO
// gate hashes ([STATE-HASH], [STATE-HASH-VALMS], [STATE-HASH-VALONLY]) are
// validated by re-running and must not move.
//
// What is shared here (byte-exact):
//   - fnv1a / fnv1aUpdate : the FNV-1a 64-bit primitive (offset-basis + prime),
//                           with an update form so a caller can fold several byte
//                           ranges into one running hash.
//   - foldRecordValuesOnly<T> : the value-only, slot-sorted per-record fold used
//                               by TPC-C's per-table val_ms and YCSB's VALONLY.
//   - negControlPerturb<T>    : the verification-only negative-control hook
//                               (EPIC_RECOVERY_CORRUPT_ONE / EPIC_RECOVERY_SWAP_TWO).
//
// What is intentionally NOT shared (left inline in each .cpp): the positional
// full-record folds. TPC-C's [STATE-HASH] xor-reduces per-record hashes within
// each table and folds the 8 table hashes with xor-then-multiply, and its
// per-table [TBL-HASH] positional additionally salts each record hash by
// (r * 0x9E3779B97F4A7C15); YCSB's hashCpuRecords folds a single table with a
// plain xor and no index salt. They are not byte-identical across the two
// benchmarks, so sharing them would risk the canonical hashes.
//
#ifndef EPIC_BENCHMARKS_RECOVERY_VERIFY_H
#define EPIC_BENCHMARKS_RECOVERY_VERIFY_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>   // std::getenv
#include <cstring>   // std::memcmp / std::memcpy

#include <storage.h> // epic::Record<T>
#include "util_log.h"

namespace epic {

// Validation-only: recovery state-hash + negative-control primitives. Excluded
// from the default product build; -DEGAD_VALIDATION compiles the harness.
#ifdef EGAD_VALIDATION

// FNV-1a 64-bit constants. Shared by the TPC-C and YCSB hash sites
// in tpcc_recovery.cpp and ycsb_recovery.cpp.
inline constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

// FNV-1a update: fold `len` bytes of `data` into the running hash `h`
// (h ^= byte; h *= prime), byte by byte, in ascending address order. Seed a
// fresh hash with kFnvOffsetBasis. This is the exact inner loop the inline
// folds used; passing the same seed and the same byte range reproduces the
// same hash bit-for-bit.
EPIC_FORCE_INLINE uint64_t fnv1aUpdate(uint64_t h, const void *data, size_t len)
{
    const uint8_t *p = reinterpret_cast<const uint8_t *>(data);
    for (size_t b = 0; b < len; ++b) { h ^= p[b]; h *= kFnvPrime; }
    return h;
}

// FNV-1a over a single contiguous byte range, seeded with the offset basis.
EPIC_FORCE_INLINE uint64_t fnv1a(const void *data, size_t len)
{
    return fnv1aUpdate(kFnvOffsetBasis, data, len);
}

// Value-only, slot-sorted per-record fold. Byte-exact replica of the per-record
// fold previously inlined in TPC-C's logCpuRecordsTableHashes (val_ms) and
// YCSB's hashCpuRecordsValOnly. The two value slots are ordered lower-bytes-
// first via memcmp(value2, value1) < 0, then folded lo-then-hi with FNV-1a from
// the offset basis. Invariant to version tags AND value1<->value2 slot swaps.
// vsz = sizeof(value1) = sizeof(value2). Returns the per-record hash; the caller
// owns the cross-record combine (sum -> multiset).
template <typename ValueType>
EPIC_FORCE_INLINE uint64_t foldRecordValuesOnly(const Record<ValueType> &rec)
{
    constexpr size_t vsz = sizeof(rec.value1);
    const uint8_t *v1 = reinterpret_cast<const uint8_t *>(&rec.value1);
    const uint8_t *v2 = reinterpret_cast<const uint8_t *>(&rec.value2);
    const uint8_t *lo = v1, *hi = v2;
    if (std::memcmp(v2, v1, vsz) < 0) { lo = v2; hi = v1; }
    uint64_t hv = kFnvOffsetBasis;
    hv = fnv1aUpdate(hv, lo, vsz);
    hv = fnv1aUpdate(hv, hi, vsz);
    return hv;
}

// Verification-only negative-control hook. Strictly a NO-OP unless one of the
// two env vars is set, so OFF leaves the Primary Store byte-identical (the
// recovery determinism gate keeps holding). When set, it perturbs the Primary
// Store right before the end-of-run state hashes are computed, to prove the hash
// gate actually CATCHES a corrupted recovery.
//   EPIC_RECOVERY_CORRUPT_ONE : XOR 0xFF into value1 byte 0 of record 0.
//   EPIC_RECOVERY_SWAP_TWO    : swap the FULL value bytes (value1 AND value2)
//                               of record 0 and record 1.
// `recs` points at a live, populated table region of `count` records; `tag`
// names the table in the [NEG-CONTROL] log line. Consolidates the two previous
// corruptPrimaryStoreForNegControl bodies (TP-CC stock table; YCSB record 0/1).
template <typename ValueType>
inline void negControlPerturb(Record<ValueType> *recs, size_t count, const char *tag)
{
    auto &logger = Logger::GetInstance();
    const bool do_corrupt = std::getenv("EPIC_RECOVERY_CORRUPT_ONE") != nullptr;
    const bool do_swap    = std::getenv("EPIC_RECOVERY_SWAP_TWO")    != nullptr;
    if (!do_corrupt && !do_swap) return;   // default: byte-identical no-op

    if (!recs || count < 2) {
        logger.Error("[NEG-CONTROL] {} records unavailable/too small; hook skipped", tag);
        return;
    }
    constexpr size_t vsz = sizeof(recs->value1);
    if (do_corrupt) {
        uint8_t *vb = reinterpret_cast<uint8_t *>(&recs[0].value1);
        vb[0] ^= 0xFFu;   // flip one value byte of record 0
        logger.Info("[NEG-CONTROL] corrupted 1 byte of {} CRID {}", tag, 0);
    }
    if (do_swap) {
        ValueType tmp;
        std::memcpy(&tmp,            &recs[0].value1, vsz);
        std::memcpy(&recs[0].value1, &recs[1].value1, vsz);
        std::memcpy(&recs[1].value1, &tmp,            vsz);
        std::memcpy(&tmp,            &recs[0].value2, vsz);
        std::memcpy(&recs[0].value2, &recs[1].value2, vsz);
        std::memcpy(&recs[1].value2, &tmp,            vsz);
        logger.Info("[NEG-CONTROL] swapped full values of {} CRID {} and {}", tag, 0, 1);
    }
}

#endif // EGAD_VALIDATION

} // namespace epic

#endif // EPIC_BENCHMARKS_RECOVERY_VERIFY_H
