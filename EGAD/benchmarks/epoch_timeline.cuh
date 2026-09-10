//
// epoch_timeline.cuh -- serial-order resolution of inserts and deletes
// inside one epoch. Shared by the YCSB index and the TPC-C growing tables.
//
// Every operation of an epoch has a serial position (transaction order,
// then program order inside the transaction). The INSERT and DELETE
// operations on a key, sorted by position, form the key's timeline: live
// with the epoch-start record until a delete, absent until the next insert,
// live with a fresh record afterwards. Each operation resolves to the record
// that is live at its own position, or to kAbsent, and the index keeps each
// key's final life at the epoch boundary. Execution never sees a delete.
//
// Rules: a delete of a live key ends its life (an "effective" delete); a
// delete of an absent key is a no-op; an insert of an absent key creates a
// life with a fresh CRID (a "life-creating" insert); an insert of a live key
// is a write to the live record. Fresh CRIDs are minted for the
// life-creating inserts in entry order, so workloads without deletes or
// duplicate inserts keep exactly the allocation they had before.
//
// Inputs are per entry: the benchmark fills insKeys()[e] / delKeys()[e] with
// the key of the INSERT / DELETE at entry e (kNoKey elsewhere) and pos()[e]
// with the entry's serial position. Several entries may share a position
// (a transaction's fifteen order lines are one position); ties are broken
// by entry order. Packed sort key = (key << PosBits) | pos, so keys must fit
// in 64 - PosBits bits and positions in PosBits bits.
//

#ifndef EPIC_BENCHMARKS_EPOCH_TIMELINE_CUH
#define EPIC_BENCHMARKS_EPOCH_TIMELINE_CUH

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef EPIC_CUDA_AVAILABLE

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <util_gpu_error_check.cuh>

namespace epic::timeline {

constexpr uint32_t kAbsent   = 0xffffffffu;
constexpr uint8_t  kEvInsert = 0;
constexpr uint8_t  kEvDelete = 1;

template <int PosBits>
__host__ __device__ __forceinline__ uint64_t packEvent(uint64_t key, uint32_t pos)
{
    return (key << PosBits) | static_cast<uint64_t>(pos);
}
template <int PosBits>
__host__ __device__ __forceinline__ uint64_t eventKey(uint64_t packed) { return packed >> PosBits; }
template <int PosBits>
__host__ __device__ __forceinline__ uint32_t eventPos(uint64_t packed)
{
    return static_cast<uint32_t>(packed & ((uint64_t{1} << PosBits) - 1u));
}

// Device-side view of one table's timelines for the epoch.
template <int PosBits>
struct TimelineView
{
    const uint64_t* sorted = nullptr;     // packed (key, pos), sorted
    const uint32_t* entry  = nullptr;     // entry index of each sorted event
    const uint8_t*  type   = nullptr;     // kEvInsert / kEvDelete
    const uint32_t* c0     = nullptr;     // epoch-start record of the key (kAbsent if none)
    const uint32_t* entry_crid = nullptr; // minted CRID at a life-creating insert's entry; ended CRID at an effective delete's
    uint32_t n = 0;
    // Validation control: ignore positions and resolve every operation the
    // way the epoch-boundary semantics did (the epoch-start record, else the
    // first record created this epoch). The oracle must reject it.
    bool order_blind = false;

    // Resolve the operation at (pos, self_entry) on `key`: the state after
    // every event of the key ordered before it; fallback (the epoch-start
    // lookup) when the key has no event this epoch. A life-creating INSERT
    // resolves to its own minted CRID, an INSERT of a live key to that
    // record, a DELETE to the record it ends (kAbsent when it ends nothing).
    __device__ __forceinline__ uint32_t resolve(uint64_t key, uint32_t pos, uint32_t self_entry,
                                                bool is_insert, uint32_t fallback) const
    {
        if (n == 0) return fallback;
        const uint64_t probe = packEvent<PosBits>(key, 0u);
        uint32_t lo = 0, hi = n;
        while (lo < hi) {
            const uint32_t mid = lo + ((hi - lo) >> 1);
            if (sorted[mid] < probe) lo = mid + 1; else hi = mid;
        }
        if (lo == n || eventKey<PosBits>(sorted[lo]) != key) return fallback;
        uint32_t state = c0[lo];
        if (order_blind) {
            if (state != kAbsent) return state;
            for (uint32_t j = lo; j < n && eventKey<PosBits>(sorted[j]) == key; ++j) {
                if (type[j] == kEvInsert) return entry_crid[entry[j]];
            }
            return fallback;
        }
        for (uint32_t j = lo; j < n && eventKey<PosBits>(sorted[j]) == key; ++j) {
            const uint32_t p = eventPos<PosBits>(sorted[j]);
            if (p > pos || (p == pos && entry[j] >= self_entry)) break;
            if (type[j] == kEvDelete) { state = kAbsent; }
            else if (state == kAbsent) { state = entry_crid[entry[j]]; }
        }
        if (is_insert && state == kAbsent) return entry_crid[self_entry];
        return state;
    }
};

namespace detail {

template <typename KeyT>
struct EntryHasKey {
    const KeyT* keys;
    __device__ bool operator()(uint32_t e) const { return keys[e] != static_cast<KeyT>(-1); }
};
struct FlagSet {
    const uint8_t* flags;
    __device__ bool operator()(uint32_t e) const { return flags[e] != 0; }
};

template <typename KeyT>
__global__ void k_gather_events(const uint32_t* __restrict__ ev_entry, uint32_t n,
                                const KeyT* __restrict__ ins_key, const KeyT* __restrict__ del_key,
                                KeyT* __restrict__ ev_key, uint8_t* __restrict__ ev_type)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t e = ev_entry[i];
    const KeyT ins = ins_key[e];
    if (ins != static_cast<KeyT>(-1)) { ev_key[i] = ins; ev_type[i] = kEvInsert; }
    else                               { ev_key[i] = del_key[e]; ev_type[i] = kEvDelete; }
}

template <typename KeyT, int PosBits>
__global__ void k_pack_events(const KeyT* __restrict__ ev_key, const uint32_t* __restrict__ ev_entry,
                              const uint32_t* __restrict__ pos, uint32_t n,
                              uint64_t* __restrict__ packed, uint32_t* __restrict__ idx)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    packed[i] = packEvent<PosBits>(static_cast<uint64_t>(ev_key[i]), pos[ev_entry[i]]);
    idx[i] = i;
}

static __global__ void k_permute_events(const uint32_t* __restrict__ sorted_idx, uint32_t n,
                                        const uint8_t* __restrict__ ev_type, const uint32_t* __restrict__ ev_c0,
                                        const uint32_t* __restrict__ ev_entry,
                                        uint8_t* __restrict__ s_type, uint32_t* __restrict__ s_c0,
                                        uint32_t* __restrict__ s_entry)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t src = sorted_idx[i];
    s_type[i] = ev_type[src];
    s_c0[i] = ev_c0[src];
    s_entry[i] = ev_entry[src];
}

// Pass 1: flag the life-creating inserts and the effective deletes, one
// byte per entry. One thread per sorted event; the first event of each
// key walks the key's timeline.
template <int PosBits>
__global__ void k_walk(const uint64_t* __restrict__ sorted, const uint32_t* __restrict__ s_entry,
                       const uint8_t* __restrict__ s_type, const uint32_t* __restrict__ s_c0, uint32_t n,
                       uint8_t* __restrict__ life_flag, uint8_t* __restrict__ edel_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t key = eventKey<PosBits>(sorted[i]);
    if (i > 0 && eventKey<PosBits>(sorted[i - 1]) == key) return;
    bool live = s_c0[i] != kAbsent;
    for (uint32_t j = i; j < n && eventKey<PosBits>(sorted[j]) == key; ++j) {
        const uint32_t e = s_entry[j];
        if (s_type[j] == kEvDelete) {
            if (live) { edel_flag[e] = 1; live = false; }
        } else {
            if (!live) { life_flag[e] = 1; live = true; }
        }
    }
}

// entry_crid[ins_entry[j]] = crid_begin + j for the minted inserts.
static __global__ void k_scatter_minted(const uint32_t* __restrict__ ins_entry, uint32_t n,
                                        uint32_t crid_begin, uint32_t* __restrict__ entry_crid)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    entry_crid[ins_entry[j]] = crid_begin + j;
}

// Pass 2, after minting: the CRID each effective delete ends (stored at
// the delete's entry) and the key's index update at the epoch boundary.
// Written on the first event of each key: erase when the key existed and
// its final life is not the epoch-start record; insert when the final life
// is a record minted this epoch.
template <int PosBits>
__global__ void k_assign(const uint64_t* __restrict__ sorted, const uint32_t* __restrict__ s_entry,
                         const uint8_t* __restrict__ s_type, const uint32_t* __restrict__ s_c0, uint32_t n,
                         uint32_t* __restrict__ entry_crid,
                         uint64_t* __restrict__ fin_key, uint32_t* __restrict__ fin_crid,
                         uint8_t* __restrict__ fin_erase, uint8_t* __restrict__ fin_insert)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fin_erase[i] = 0; fin_insert[i] = 0;
    const uint64_t key = eventKey<PosBits>(sorted[i]);
    if (i > 0 && eventKey<PosBits>(sorted[i - 1]) == key) return;
    const uint32_t start = s_c0[i];
    uint32_t state = start;
    for (uint32_t j = i; j < n && eventKey<PosBits>(sorted[j]) == key; ++j) {
        const uint32_t e = s_entry[j];
        if (s_type[j] == kEvDelete) {
            if (state != kAbsent) { entry_crid[e] = state; state = kAbsent; }
        } else {
            if (state == kAbsent) state = entry_crid[e];
        }
    }
    fin_key[i] = key;
    fin_crid[i] = state;
    fin_erase[i]  = (start != kAbsent && state != start) ? 1 : 0;
    fin_insert[i] = (state != kAbsent && state != start) ? 1 : 0;
}

template <typename KeyT>
__global__ void k_narrow_keys(const uint64_t* __restrict__ wide, uint32_t n, KeyT* __restrict__ out)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<KeyT>(wide[i]);
}

} // namespace detail

namespace detail {

template <typename KeyT>
__global__ void k_flag_events(const KeyT* __restrict__ ins, const KeyT* __restrict__ del, uint32_t n,
                              uint8_t* __restrict__ ev)
{
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    ev[e] = (ins[e] != static_cast<KeyT>(-1) || del[e] != static_cast<KeyT>(-1)) ? 1 : 0;
}

template <typename KeyT>
inline void flagEvents(const KeyT* ins, const KeyT* del, uint32_t n, uint8_t* ev)
{
    k_flag_events<KeyT><<<(n + 255u) / 256u, 256>>>(ins, del, n, ev);
    gpu_err_check(cudaPeekAtLastError());
}

} // namespace detail

// One table's per-epoch timeline machinery. Device buffers are allocated
// once by ensure(); build() runs the epoch. Outputs stay valid until the
// next build().
template <typename KeyT, int PosBits>
class EpochTimeline
{
public:
    static constexpr KeyT kNoKey = static_cast<KeyT>(-1);
    using View = TimelineView<PosBits>;

    // n_entries = per-epoch input entries; e_max = the most events an epoch can hold.
    void ensure(uint32_t n_entries, uint32_t e_max)
    {
        if (ready_) return;
        if (e_max > n_entries) e_max = n_entries;
        n_entries_ = n_entries; e_max_ = e_max;
        gpu_err_check(cudaMalloc(&d_ins_key_, sizeof(KeyT) * n_entries));
        gpu_err_check(cudaMalloc(&d_del_key_, sizeof(KeyT) * n_entries));
        gpu_err_check(cudaMalloc(&d_pos_, sizeof(uint32_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_ev_flag_, sizeof(uint8_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_life_flag_, sizeof(uint8_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_edel_flag_, sizeof(uint8_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_entry_crid_, sizeof(uint32_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_ins_entry_, sizeof(uint32_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_minted_keys_, sizeof(KeyT) * n_entries));
        gpu_err_check(cudaMalloc(&d_edel_keys_, sizeof(KeyT) * n_entries));
        gpu_err_check(cudaMalloc(&d_edel_crids_, sizeof(uint32_t) * n_entries));
        gpu_err_check(cudaMalloc(&d_ev_entry_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_ev_key_, sizeof(KeyT) * e_max));
        gpu_err_check(cudaMalloc(&d_ev_type_, sizeof(uint8_t) * e_max));
        gpu_err_check(cudaMalloc(&d_ev_c0_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_ev_idx_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_ev_packed_, sizeof(uint64_t) * e_max));
        gpu_err_check(cudaMalloc(&d_sorted_, sizeof(uint64_t) * e_max));
        gpu_err_check(cudaMalloc(&d_sorted_idx_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_s_type_, sizeof(uint8_t) * e_max));
        gpu_err_check(cudaMalloc(&d_s_c0_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_s_entry_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_fin_key_, sizeof(uint64_t) * e_max));
        gpu_err_check(cudaMalloc(&d_fin_crid_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaMalloc(&d_fin_erase_, sizeof(uint8_t) * e_max));
        gpu_err_check(cudaMalloc(&d_fin_insert_, sizeof(uint8_t) * e_max));
        gpu_err_check(cudaMalloc(&d_erase_wide_, sizeof(uint64_t) * e_max));
        gpu_err_check(cudaMalloc(&d_insert_wide_, sizeof(uint64_t) * e_max));
        gpu_err_check(cudaMalloc(&d_erase_keys_, sizeof(KeyT) * e_max));
        gpu_err_check(cudaMalloc(&d_insert_keys_, sizeof(KeyT) * e_max));
        gpu_err_check(cudaMalloc(&d_insert_crids_, sizeof(uint32_t) * e_max));
        gpu_err_check(cudaHostAlloc(&h_counts_, 8 * sizeof(uint32_t), cudaHostAllocMapped));
        for (int i = 0; i < 8; ++i) h_counts_[i] = 0;
        gpu_err_check(cudaHostGetDevicePointer(&d_counts_, h_counts_, 0));
        // cub temp storage: the largest of the calls below.
        size_t need = 0, bytes = 0;
        cub::DeviceSelect::If(nullptr, bytes, thrust::counting_iterator<uint32_t>(0), d_ev_entry_, d_counts_,
            n_entries, detail::FlagSet{d_ev_flag_});
        need = std::max(need, bytes); bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, bytes, d_ins_key_, d_life_flag_, d_minted_keys_, d_counts_, n_entries);
        need = std::max(need, bytes); bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, bytes, d_entry_crid_, d_edel_flag_, d_edel_crids_, d_counts_, n_entries);
        need = std::max(need, bytes); bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, bytes, d_fin_key_, d_fin_erase_, d_erase_wide_, d_counts_, e_max);
        need = std::max(need, bytes); bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, d_ev_packed_, d_sorted_, d_ev_idx_, d_sorted_idx_, e_max);
        need = std::max(need, bytes);
        temp_bytes_ = need;
        gpu_err_check(cudaMalloc(&d_temp_, temp_bytes_));
        ready_ = true;
    }

    bool ready() const { return ready_; }
    uint32_t numEntries() const { return n_entries_; }

    // Per-epoch inputs (device pointers). clearInputs() resets both key
    // arrays to kNoKey; the caller fills the entries it uses and pos()[e]
    // for every entry it may flag.
    KeyT* insKeys() { return d_ins_key_; }
    KeyT* delKeys() { return d_del_key_; }
    uint32_t* pos() { return d_pos_; }
    void clearInputs(cudaStream_t s = 0)
    {
        gpu_err_check(cudaMemsetAsync(d_ins_key_, 0xff, sizeof(KeyT) * n_entries_, s));
        gpu_err_check(cudaMemsetAsync(d_del_key_, 0xff, sizeof(KeyT) * n_entries_, s));
    }

    // Build the epoch's timelines. lookup(keys, n, out) must write the
    // epoch-start record of each key (kAbsent on a miss); crid_begin is the
    // first CRID to mint. Synchronous on stream 0.
    template <typename LookupFn>
    void build(LookupFn lookup, uint32_t crid_begin)
    {
        auto blocks = [](uint32_t n) { return (n + 255u) / 256u; };
        const uint32_t n_entries = n_entries_;
        crid_begin_ = crid_begin;
        // Events in entry order.
        detail::flagEvents(d_ins_key_, d_del_key_, n_entries, d_ev_flag_);
        cub::DeviceSelect::If(d_temp_, temp_bytes_, thrust::counting_iterator<uint32_t>(0), d_ev_entry_, d_counts_,
            n_entries, detail::FlagSet{d_ev_flag_});
        gpu_err_check(cudaStreamSynchronize(0));
        n_ev_ = h_counts_[0];
        if (n_ev_ > e_max_) {
            throw std::runtime_error("EpochTimeline: " + std::to_string(n_ev_) + " events exceed the capacity of " +
                                     std::to_string(e_max_));
        }
        n_minted_ = 0; n_edel_ = 0; n_erase_ = 0; n_insert_ = 0;
        if (n_ev_ == 0) return;
        const uint32_t n = n_ev_;
        detail::k_gather_events<KeyT><<<blocks(n), 256>>>(d_ev_entry_, n, d_ins_key_, d_del_key_, d_ev_key_, d_ev_type_);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        lookup(d_ev_key_, n, d_ev_c0_);
        detail::k_pack_events<KeyT, PosBits><<<blocks(n), 256>>>(d_ev_key_, d_ev_entry_, d_pos_, n, d_ev_packed_, d_ev_idx_);
        gpu_err_check(cudaPeekAtLastError());
        cub::DeviceRadixSort::SortPairs(d_temp_, temp_bytes_, d_ev_packed_, d_sorted_, d_ev_idx_, d_sorted_idx_, n);
        detail::k_permute_events<<<blocks(n), 256>>>(d_sorted_idx_, n, d_ev_type_, d_ev_c0_, d_ev_entry_,
            d_s_type_, d_s_c0_, d_s_entry_);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaMemsetAsync(d_life_flag_, 0, n_entries, 0));
        gpu_err_check(cudaMemsetAsync(d_edel_flag_, 0, n_entries, 0));
        detail::k_walk<PosBits><<<blocks(n), 256>>>(d_sorted_, d_s_entry_, d_s_type_, d_s_c0_, n, d_life_flag_, d_edel_flag_);
        gpu_err_check(cudaPeekAtLastError());
        // Mint: life-creating inserts in entry order.
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_ins_key_, d_life_flag_, d_minted_keys_, d_counts_ + 1, n_entries);
        cub::DeviceSelect::If(d_temp_, temp_bytes_, thrust::counting_iterator<uint32_t>(0), d_ins_entry_, d_counts_ + 1,
            n_entries, detail::FlagSet{d_life_flag_});
        gpu_err_check(cudaStreamSynchronize(0));
        n_minted_ = h_counts_[1];
        if (n_minted_ > 0) {
            detail::k_scatter_minted<<<blocks(n_minted_), 256>>>(d_ins_entry_, n_minted_, crid_begin, d_entry_crid_);
            gpu_err_check(cudaPeekAtLastError());
        }
        detail::k_assign<PosBits><<<blocks(n), 256>>>(d_sorted_, d_s_entry_, d_s_type_, d_s_c0_, n, d_entry_crid_,
            d_fin_key_, d_fin_crid_, d_fin_erase_, d_fin_insert_);
        gpu_err_check(cudaPeekAtLastError());
        // Effective deletes (entry order): keys and the CRIDs they end.
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_del_key_, d_edel_flag_, d_edel_keys_, d_counts_ + 2, n_entries);
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_entry_crid_, d_edel_flag_, d_edel_crids_, d_counts_ + 2, n_entries);
        // Index update lists.
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_fin_key_, d_fin_erase_, d_erase_wide_, d_counts_ + 3, n);
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_fin_key_, d_fin_insert_, d_insert_wide_, d_counts_ + 4, n);
        cub::DeviceSelect::Flagged(d_temp_, temp_bytes_, d_fin_crid_, d_fin_insert_, d_insert_crids_, d_counts_ + 4, n);
        gpu_err_check(cudaStreamSynchronize(0));
        n_edel_ = h_counts_[2]; n_erase_ = h_counts_[3]; n_insert_ = h_counts_[4];
        if (n_erase_ > 0) {
            detail::k_narrow_keys<KeyT><<<blocks(n_erase_), 256>>>(d_erase_wide_, n_erase_, d_erase_keys_);
            gpu_err_check(cudaPeekAtLastError());
        }
        if (n_insert_ > 0) {
            detail::k_narrow_keys<KeyT><<<blocks(n_insert_), 256>>>(d_insert_wide_, n_insert_, d_insert_keys_);
            gpu_err_check(cudaPeekAtLastError());
        }
        gpu_err_check(cudaStreamSynchronize(0));
    }

    View view() const
    {
        View v;
        v.sorted = d_sorted_; v.entry = d_s_entry_; v.type = d_s_type_; v.c0 = d_s_c0_;
        v.entry_crid = d_entry_crid_; v.n = n_ev_; v.order_blind = order_blind_;
        return v;
    }
    void setOrderBlind(bool blind) { order_blind_ = blind; }

    uint32_t numEvents() const { return n_ev_; }
    uint32_t numMinted() const { return n_minted_; }
    uint32_t numEffectiveDeletes() const { return n_edel_; }
    uint32_t numErase() const { return n_erase_; }
    uint32_t numInsert() const { return n_insert_; }
    uint32_t cridBegin() const { return crid_begin_; }
    const KeyT* mintedKeys() const { return d_minted_keys_; }      // entry order; j-th <-> crid_begin + j
    const uint32_t* insEntries() const { return d_ins_entry_; }    // entry of the j-th minted insert
    const KeyT* edelKeys() const { return d_edel_keys_; }
    const uint32_t* edelCrids() const { return d_edel_crids_; }
    const KeyT* eraseKeys() const { return d_erase_keys_; }
    const KeyT* insertKeys() const { return d_insert_keys_; }
    const uint32_t* insertCrids() const { return d_insert_crids_; }

private:
    bool ready_ = false;
    bool order_blind_ = false;
    uint32_t n_entries_ = 0, e_max_ = 0;
    uint32_t n_ev_ = 0, n_minted_ = 0, n_edel_ = 0, n_erase_ = 0, n_insert_ = 0, crid_begin_ = 0;
    KeyT *d_ins_key_ = nullptr, *d_del_key_ = nullptr;
    uint32_t *d_pos_ = nullptr;
    uint8_t *d_ev_flag_ = nullptr, *d_life_flag_ = nullptr, *d_edel_flag_ = nullptr;
    uint32_t *d_entry_crid_ = nullptr, *d_ins_entry_ = nullptr;
    KeyT *d_minted_keys_ = nullptr, *d_edel_keys_ = nullptr;
    uint32_t *d_edel_crids_ = nullptr;
    uint32_t *d_ev_entry_ = nullptr, *d_ev_c0_ = nullptr, *d_ev_idx_ = nullptr;
    KeyT *d_ev_key_ = nullptr;
    uint8_t *d_ev_type_ = nullptr;
    uint64_t *d_ev_packed_ = nullptr, *d_sorted_ = nullptr;
    uint32_t *d_sorted_idx_ = nullptr, *d_s_c0_ = nullptr, *d_s_entry_ = nullptr;
    uint8_t *d_s_type_ = nullptr;
    uint64_t *d_fin_key_ = nullptr;
    uint32_t *d_fin_crid_ = nullptr;
    uint8_t *d_fin_erase_ = nullptr, *d_fin_insert_ = nullptr;
    uint64_t *d_erase_wide_ = nullptr, *d_insert_wide_ = nullptr;
    KeyT *d_erase_keys_ = nullptr, *d_insert_keys_ = nullptr;
    uint32_t *d_insert_crids_ = nullptr;
    uint32_t *h_counts_ = nullptr, *d_counts_ = nullptr;
    void *d_temp_ = nullptr;
    size_t temp_bytes_ = 0;
};


} // namespace epic::timeline

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_BENCHMARKS_EPOCH_TIMELINE_CUH
