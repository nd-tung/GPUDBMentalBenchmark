#include <metal_stdlib>
using namespace metal;

// Generic operator kernels (scaffold)
// These are minimal, compilable stubs to be expanded.

namespace ops {

struct ColumnViewUInt32 {
    device const uint32_t* data;
    uint32_t count;
};

struct ColumnViewFloat {
    device const float* data;
    uint32_t count;
};

struct ColumnOutUInt32 {
    device uint32_t* data;
    uint32_t count;
};

struct RowMask {
    device uint8_t* mask; // 0/1 per row
    uint32_t count;
};

struct HashTable {
    device uint32_t* keys;
    device uint32_t* payloads;
    uint32_t capacity;
};

kernel void filter_eq_u32(const device uint32_t* in,
                          device uint8_t* out_mask,
                          constant uint32_t& eq_value,
                          uint gid [[thread_position_in_grid]],
                          uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    out_mask[gid] = (in[gid] == eq_value) ? 1 : 0;
}

kernel void project_select_u32(const device uint32_t* in,
                               const device uint8_t* mask,
                               device uint32_t* out,
                               uint gid [[thread_position_in_grid]],
                               uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Simple pass-through respecting mask; real impl will compact
    out[gid] = mask[gid] ? in[gid] : 0u;
}

kernel void hash_build_u32(const device uint32_t* keys,
                           const device uint32_t* payloads,
                           device uint32_t* ht_keys,
                           device uint32_t* ht_vals,
                           constant uint32_t& capacity,
                           uint gid [[thread_position_in_grid]],
                           uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Stub: place directly with modulo, no collision handling (to be replaced)
    uint32_t k = keys[gid];
    uint32_t v = payloads[gid];
    uint32_t slot = k % capacity;
    ht_keys[slot] = k;
    ht_vals[slot] = v;
}

kernel void hash_probe_u32(const device uint32_t* probe_keys,
                           const device uint32_t* ht_keys,
                           const device uint32_t* ht_vals,
                           device uint32_t* out_payload,
                           constant uint32_t& capacity,
                           uint gid [[thread_position_in_grid]],
                           uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    // Stub: single-slot probe
    uint32_t k = probe_keys[gid];
    uint32_t slot = k % capacity;
    out_payload[gid] = (ht_keys[slot] == k) ? ht_vals[slot] : 0u;
}

struct GroupByBucketF32 {
    atomic_uint key; // simple key
    atomic_uint count;
    atomic_uint sum_bits; // reinterpret float
};

inline float atomicLoadF32Bits(const device atomic_uint* a) {
    return as_type<float>(atomic_load_explicit((device atomic_uint*)a, memory_order_relaxed));
}

inline void atomicAddF32Bits(device atomic_uint* a, float v) {
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(expected);
        float nxt = cur + v;
        uint desired = as_type<uint>(nxt);
        if (atomic_compare_exchange_weak_explicit(a, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

kernel void groupby_sum_f32(const device uint32_t* keys,
                            const device float* vals,
                            device atomic_uint* bucket_keys,
                            device atomic_uint* bucket_counts,
                            device atomic_uint* bucket_sumbits,
                            constant uint32_t& bucket_mask,
                            uint gid [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]]) {
    if (gid >= grid_size) return;
    uint32_t k = keys[gid];
    float v = vals[gid];
    uint32_t slot = k & bucket_mask; // power-of-two buckets
    // Very naive: set key if empty, increment count, add sum
    atomic_store_explicit(&bucket_keys[slot], k, memory_order_relaxed);
    atomic_fetch_add_explicit(&bucket_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&bucket_sumbits[slot], v);
}

// Packed predicate clause (host must ensure alignment)
struct PredicateClause {
    uint colIndex;   // Column index among provided buffers
    uint op;         // 0:LT 1:LE 2:GT 3:GE 4:EQ
    uint isDate;     // 0 numeric 1 date/int
    int64_t value;   // encoded literal (date as YYYYMMDD or bitcasted double via int64)
};

// Kernel: scan + predicates + sum over single float column
// Simplified kernel: only target column + predicate array (all numeric comparisons).
// Optimized version: per-threadgroup partial reduction to minimize global atomics.
// Each thread writes its passing value into threadgroup memory, performs parallel reduction,
// and a single atomicAdd updates the global accumulator.
kernel void scan_filter_sum_f32(const device float* target_col,
                                constant PredicateClause* clauses,
                                constant uint& clause_count,
                                constant uint& row_count,
                                device atomic_uint* out_sum_bits,
                                uint gid [[thread_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    // Cap threadgroup size we actually use (static alloc below)
    if (tgSize > 1024) tgSize = 1024;
    threadgroup float localVals[1024];

    // Evaluate predicates; if any fail we contribute 0.
    float v = target_col[gid];
    bool passes = true;
    for (uint c = 0; c < clause_count && passes; ++c) {
        PredicateClause pc = clauses[c];
        // Numeric float literal in lower 32 bits
        union { uint32_t u; float f; } conv; conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
        float lit = conv.f;
        switch (pc.op) {
            case 0: passes = v < lit; break;
            case 1: passes = v <= lit; break;
            case 2: passes = v > lit; break;
            case 3: passes = v >= lit; break;
            case 4: passes = v == lit; break;
        }
    }
    localVals[tid] = (passes ? v : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction inside threadgroup
    // Assumes tgSize is power-of-two or handles leftover by ignoring out-of-range indices.
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            localVals[tid] += localVals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomicAddF32Bits(out_sum_bits, localVals[0]);
    }
}

} // namespace ops
