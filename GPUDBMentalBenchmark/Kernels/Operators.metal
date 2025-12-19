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
    uint isDate;     // 0 numeric/string, 1 date/int
    uint isString;   // 0 numeric/date, 1 string (hash comparison)
    uint isOrNext;   // 0 next clause is AND'd, 1 next clause is OR'd
    uint _pad;       // padding for alignment
    int64_t value;   // encoded literal (date as YYYYMMDD, float bits, or string hash)
};

// RPN expression token for arithmetic evaluation
struct ExprToken {
    uint type;       // 0:column_ref 1:literal 2:operator
    uint colIndex;   // if type==0, column index
    float literal;   // if type==1, literal value
    uint op;         // if type==2, operator: 0:+ 1:- 2:* 3:/
};

// Multi-column scan+filter+sum kernel
// Accepts up to 8 column buffers via [[buffer(0..7)]]
// Predicates reference columns by index (0..7)
// Target aggregation column is always buffer(0)
kernel void scan_filter_sum_f32(const device float* col0 [[buffer(0)]],
                                const device float* col1 [[buffer(1)]],
                                const device float* col2 [[buffer(2)]],
                                const device float* col3 [[buffer(3)]],
                                const device float* col4 [[buffer(4)]],
                                const device float* col5 [[buffer(5)]],
                                const device float* col6 [[buffer(6)]],
                                const device float* col7 [[buffer(7)]],
                                constant PredicateClause* clauses [[buffer(8)]],
                                constant uint& col_count [[buffer(9)]],
                                constant uint& clause_count [[buffer(10)]],
                                constant uint& row_count [[buffer(11)]],
                                device atomic_uint* out_sum_bits [[buffer(12)]],
                                uint gid [[thread_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    threadgroup float localVals[1024];

    // Build local column pointer array for dynamic indexing
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    
    // Target column for aggregation is always col0
    float target_val = cols[0][gid];
    
    // Evaluate predicates with dynamic column access and OR/AND logic
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            // Date stored as YYYYMMDD integer in column, compare as integers
            int date_val = as_type<int>(col_val);  // reinterpret float bits as int
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);  // lower 32 bits
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else if (pc.isString) {
            // String comparison via hash - only equality supported
            uint col_hash = as_type<uint>(col_val);  // reinterpret float as hash
            uint lit_hash = (uint)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 4: clauseResult = (col_hash == lit_hash); break;  // Equal
                default: clauseResult = false; break;  // Other ops not supported for strings
            }
        } else {
            // Numeric comparison
            union { uint32_t u; float f; } conv; 
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            groupResult = groupResult || clauseResult;
        } else {
            passes = passes && groupResult;
            if (!passes) break;
            groupResult = clauseResult;
        }
    }
    if (clause_count > 0) passes = passes && groupResult;
    
    localVals[tid] = (passes ? target_val : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
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

// Kernel: scan + filter + evaluate RPN expression + sum
// Supports arithmetic expressions like l_extendedprice * (1 - l_discount)
kernel void scan_filter_eval_sum(const device float* col0 [[buffer(0)]],
                                  const device float* col1 [[buffer(1)]],
                                  const device float* col2 [[buffer(2)]],
                                  const device float* col3 [[buffer(3)]],
                                  const device float* col4 [[buffer(4)]],
                                  const device float* col5 [[buffer(5)]],
                                  const device float* col6 [[buffer(6)]],
                                  const device float* col7 [[buffer(7)]],
                                  constant PredicateClause* clauses [[buffer(8)]],
                                  constant ExprToken* expr_rpn [[buffer(9)]],
                                  constant uint& col_count [[buffer(10)]],
                                  constant uint& clause_count [[buffer(11)]],
                                  constant uint& expr_length [[buffer(12)]],
                                  constant uint& row_count [[buffer(13)]],
                                  device atomic_uint* out_sum_bits [[buffer(14)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    threadgroup float localVals[1024];
    
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    
    // Evaluate predicates first with OR/AND logic
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            int date_val = as_type<int>(col_val);
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else if (pc.isString) {
            uint col_hash = as_type<uint>(col_val);
            uint lit_hash = (uint)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 4: clauseResult = (col_hash == lit_hash); break;
                default: clauseResult = false; break;
            }
        } else {
            union { uint32_t u; float f; } conv;
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            groupResult = groupResult || clauseResult;
        } else {
            passes = passes && groupResult;
            if (!passes) break;
            groupResult = clauseResult;
        }
    }
    if (clause_count > 0) passes = passes && groupResult;
    
    float result_val = 0.0f;
    if (passes) {
        // Evaluate RPN expression using stack
        float stack[32];  // Support expressions up to 32 tokens deep
        uint sp = 0;
        
        for (uint i = 0; i < expr_length; ++i) {
            ExprToken tok = expr_rpn[i];
            if (tok.type == 0) {
                // Column reference
                if (tok.colIndex < col_count && sp < 32) {
                    stack[sp++] = cols[tok.colIndex][gid];
                }
            } else if (tok.type == 1) {
                // Literal
                if (sp < 32) {
                    stack[sp++] = tok.literal;
                }
            } else if (tok.type == 2) {
                // Operator - pop two operands, apply, push result
                if (sp >= 2) {
                    float b = stack[--sp];
                    float a = stack[--sp];
                    float res = 0.0f;
                    switch (tok.op) {
                        case 0: res = a + b; break;  // ADD
                        case 1: res = a - b; break;  // SUB
                        case 2: res = a * b; break;  // MUL
                        case 3: res = (b != 0.0f) ? a / b : 0.0f; break;  // DIV
                    }
                    if (sp < 32) {
                        stack[sp++] = res;
                    }
                }
            }
        }
        
        // Final result is top of stack
        if (sp > 0) {
            result_val = stack[sp - 1];
        }
    }
    
    localVals[tid] = result_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
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

// Simple bitonic sort kernel for small arrays (ORDER BY support)
// For production use, implement radix sort or thrust-style parallel sort
kernel void bitonic_sort_step(device float* data [[buffer(0)]],
                               device uint* indices [[buffer(1)]],
                               constant uint& stage [[buffer(2)]],
                               constant uint& pass [[buffer(3)]],
                               constant uint& count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint pairDist = 1 << (stage - pass);
    uint blockWidth = 2 * pairDist;
    uint leftId = (gid % pairDist) + (gid / pairDist) * blockWidth;
    uint rightId = leftId + pairDist;
    
    if (rightId >= count) return;
    
    float leftVal = data[leftId];
    float rightVal = data[rightId];
    bool ascending = ((leftId / (1 << stage)) % 2) == 0;
    
    if ((leftVal > rightVal) == ascending) {
        // Swap
        data[leftId] = rightVal;
        data[rightId] = leftVal;
        uint tmpIdx = indices[leftId];
        indices[leftId] = indices[rightId];
        indices[rightId] = tmpIdx;
    }
}

// LIMIT kernel: copy first N elements
kernel void limit_copy(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& limit [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < limit) {
        output[gid] = input[gid];
    }
}

// Multi-column GROUP BY with multiple aggregates
// Supports up to 4 group key columns and 4 aggregate columns
kernel void groupby_agg_multi_key(const device uint* key_col0 [[buffer(0)]],
                                   const device uint* key_col1 [[buffer(1)]],
                                   const device uint* key_col2 [[buffer(2)]],
                                   const device uint* key_col3 [[buffer(3)]],
                                   const device float* agg_col0 [[buffer(4)]],
                                   const device float* agg_col1 [[buffer(5)]],
                                   const device float* agg_col2 [[buffer(6)]],
                                   const device float* agg_col3 [[buffer(7)]],
                                   device atomic_uint* ht_keys [[buffer(8)]],   // Flattened: capacity * 4 uint32s
                                   device atomic_uint* ht_agg_bits [[buffer(9)]], // Flattened: capacity * 4 floats as uint32
                                   constant uint& capacity [[buffer(10)]],
                                   constant uint& row_count [[buffer(11)]],
                                   constant uint& num_keys [[buffer(12)]],      // Number of group keys (1-4)
                                   constant uint& num_aggs [[buffer(13)]],      // Number of aggregates (1-4)
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;

    constexpr uint IN_PROGRESS = 0xFFFFFFFFu;
    
    // Read composite key
    uint keys[4];
    keys[0] = (num_keys > 0) ? key_col0[gid] : 0;
    keys[1] = (num_keys > 1) ? key_col1[gid] : 0;
    keys[2] = (num_keys > 2) ? key_col2[gid] : 0;
    keys[3] = (num_keys > 3) ? key_col3[gid] : 0;
    
    // Read aggregate values
    float aggs[4];
    aggs[0] = (num_aggs > 0) ? agg_col0[gid] : 0.0f;
    aggs[1] = (num_aggs > 1) ? agg_col1[gid] : 0.0f;
    aggs[2] = (num_aggs > 2) ? agg_col2[gid] : 0.0f;
    aggs[3] = (num_aggs > 3) ? agg_col3[gid] : 0.0f;
    
    // Compute composite hash using FNV-1a
    uint hash = 2166136261u;
    for (uint i = 0; i < num_keys; ++i) {
        hash ^= keys[i];
        hash *= 16777619u;
    }
    uint slot = hash % capacity;
    
    // Linear probing to find or insert key
    for (uint probe = 0; probe < capacity; ++probe) {
        uint probe_slot = (slot + probe) % capacity;
        uint base_idx = probe_slot * 4;  // 4 key columns per slot

        // Use the first key as the slot state/ownership indicator.
        // 0           => empty
        // IN_PROGRESS => being initialized by another thread
        // otherwise   => occupied with a valid key[0]
        uint ht_k0 = atomic_load_explicit(&ht_keys[base_idx + 0], memory_order_relaxed);

        // Empty slot - try to claim it
        if (ht_k0 == 0u) {
            uint expected = 0u;
            if (atomic_compare_exchange_weak_explicit(&ht_keys[base_idx + 0], &expected, IN_PROGRESS,
                                                      memory_order_relaxed, memory_order_relaxed)) {
                // We own this slot now. Initialize the remaining keys then publish key0.
                for (uint k = 1; k < num_keys; ++k) {
                    atomic_store_explicit(&ht_keys[base_idx + k], keys[k], memory_order_relaxed);
                }
                atomic_store_explicit(&ht_keys[base_idx + 0], keys[0], memory_order_relaxed);

                uint agg_base = probe_slot * 4;
                for (uint a = 0; a < num_aggs; ++a) {
                    atomicAddF32Bits(&ht_agg_bits[agg_base + a], aggs[a]);
                }
                return;
            }
            // Someone else raced us; retry this probe slot
            continue;
        }

        // Skip slots that are still being initialized
        if (ht_k0 == IN_PROGRESS) {
            continue;
        }

        // Check if slot matches our full key
        if (ht_k0 == keys[0]) {
            bool match = true;
            for (uint k = 1; k < num_keys; ++k) {
                uint ht_k = atomic_load_explicit(&ht_keys[base_idx + k], memory_order_relaxed);
                if (ht_k != keys[k]) { match = false; break; }
            }
            if (match) {
                uint agg_base = probe_slot * 4;
                for (uint a = 0; a < num_aggs; ++a) {
                    atomicAddF32Bits(&ht_agg_bits[agg_base + a], aggs[a]);
                }
                return;
            }
        }
        
        // Collision - continue probing
    }
}

// Multi-key GROUP BY with aggregation
// Simplified version for single uint32 key
kernel void groupby_agg_single_key(const device uint* keys [[buffer(0)]],
                                    const device float* values [[buffer(1)]],
                                    device atomic_uint* ht_keys [[buffer(2)]],
                                    device atomic_uint* ht_counts [[buffer(3)]],
                                    device atomic_uint* ht_sum_bits [[buffer(4)]],
                                    constant uint& capacity [[buffer(5)]],
                                    constant uint& row_count [[buffer(6)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= row_count) return;
    
    uint key = keys[gid];
    float val = values[gid];
    
    // Simple hash to slot
    uint slot = key % capacity;
    
    // Atomic insert/update (simplified, no collision handling)
    atomic_store_explicit(&ht_keys[slot], key, memory_order_relaxed);
    atomic_fetch_add_explicit(&ht_counts[slot], 1u, memory_order_relaxed);
    atomicAddF32Bits(&ht_sum_bits[slot], val);
}


// Hash join: Build phase
kernel void hash_join_build(const device uint* keys [[buffer(0)]],
                             const device uint* payloads [[buffer(1)]],
                             device atomic_uint* ht_keys [[buffer(2)]],
                             device atomic_uint* ht_payloads [[buffer(3)]],
                             constant uint& capacity [[buffer(4)]],
                             constant uint& build_count [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= build_count) return;
    
    uint key = keys[gid];
    uint payload = payloads[gid];
    uint slot = key % capacity;
    
    // Linear probing to find empty slot
    for (uint i = 0; i < capacity; ++i) {
        uint probe_slot = (slot + i) % capacity;
        uint expected = 0;  // Empty slot marker
        
        // Try to claim this slot atomically
        if (atomic_compare_exchange_weak_explicit(&ht_keys[probe_slot], &expected, key,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            // Successfully claimed slot, write payload
            atomic_store_explicit(&ht_payloads[probe_slot], payload, memory_order_relaxed);
            return;
        }
        
        // If slot has same key (duplicate), just update payload and return
        if (atomic_load_explicit(&ht_keys[probe_slot], memory_order_relaxed) == key) {
            atomic_store_explicit(&ht_payloads[probe_slot], payload, memory_order_relaxed);
            return;
        }
    }
}

// Hash join: Probe phase
kernel void hash_join_probe(const device uint* probe_keys [[buffer(0)]],
                             const device uint* ht_keys [[buffer(1)]],
                             const device uint* ht_payloads [[buffer(2)]],
                             device uint* output_matches [[buffer(3)]],
                             device uint* output_payloads [[buffer(4)]],
                             constant uint& capacity [[buffer(5)]],
                             constant uint& probe_count [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= probe_count) return;
    
    uint key = probe_keys[gid];
    uint slot = key % capacity;
    
    // Linear probing to find matching key
    for (uint i = 0; i < capacity; ++i) {
        uint probe_slot = (slot + i) % capacity;
        uint ht_key = atomic_load_explicit((device atomic_uint*)&ht_keys[probe_slot], memory_order_relaxed);
        
        if (ht_key == key) {
            // Match found
            output_matches[gid] = 1;
            output_payloads[gid] = atomic_load_explicit((device atomic_uint*)&ht_payloads[probe_slot], memory_order_relaxed);
            return;
        }
        
        if (ht_key == 0) {
            // Empty slot found, key doesn't exist in hash table
            output_matches[gid] = 0;
            output_payloads[gid] = 0;
            return;
        }
    }
    
    // Shouldn't reach here unless table is completely full
    output_matches[gid] = 0;
    output_payloads[gid] = 0;
}

// Generic aggregation kernel supporting COUNT, SUM, AVG, MIN, MAX
// aggType: 0=COUNT, 1=SUM, 2=AVG, 3=MIN, 4=MAX
kernel void scan_filter_aggregate(const device float* col0 [[buffer(0)]],
                                  const device float* col1 [[buffer(1)]],
                                  const device float* col2 [[buffer(2)]],
                                  const device float* col3 [[buffer(3)]],
                                  const device float* col4 [[buffer(4)]],
                                  const device float* col5 [[buffer(5)]],
                                  const device float* col6 [[buffer(6)]],
                                  const device float* col7 [[buffer(7)]],
                                  constant PredicateClause* clauses [[buffer(8)]],
                                  constant uint& col_count [[buffer(9)]],
                                  constant uint& clause_count [[buffer(10)]],
                                  constant uint& row_count [[buffer(11)]],
                                  constant uint& aggType [[buffer(12)]],
                                  device atomic_uint* out_result_bits [[buffer(13)]],
                                  device atomic_uint* out_count [[buffer(14)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint tgSize [[threads_per_threadgroup]]) {
    if (gid >= row_count) return;
    if (tgSize > 1024) tgSize = 1024;
    
    threadgroup float localVals[1024];
    threadgroup uint localCounts[1024];
    
    const device float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
    float target_val = cols[0][gid];
    
    // Evaluate predicates with OR/AND logic
    // Group consecutive clauses connected by OR, evaluate groups with AND
    bool passes = true;
    bool groupResult = true;
    
    for (uint c = 0; c < clause_count; ++c) {
        PredicateClause pc = clauses[c];
        if (pc.colIndex >= col_count) { passes = false; break; }
        float col_val = cols[pc.colIndex][gid];
        
        bool clauseResult;
        if (pc.isDate) {
            int date_val = as_type<int>(col_val);
            int date_lit = (int)(pc.value & 0xFFFFFFFFull);
            switch (pc.op) {
                case 0: clauseResult = date_val < date_lit; break;
                case 1: clauseResult = date_val <= date_lit; break;
                case 2: clauseResult = date_val > date_lit; break;
                case 3: clauseResult = date_val >= date_lit; break;
                case 4: clauseResult = date_val == date_lit; break;
                default: clauseResult = false; break;
            }
        } else {
            union { uint32_t u; float f; } conv;
            conv.u = (uint32_t)(pc.value & 0xFFFFFFFFull);
            float lit = conv.f;
            switch (pc.op) {
                case 0: clauseResult = col_val < lit; break;
                case 1: clauseResult = col_val <= lit; break;
                case 2: clauseResult = col_val > lit; break;
                case 3: clauseResult = col_val >= lit; break;
                case 4: clauseResult = col_val == lit; break;
                default: clauseResult = false; break;
            }
        }
        
        if (c == 0) {
            groupResult = clauseResult;
        } else if (clauses[c-1].isOrNext) {
            // Previous clause was OR'd with this one
            groupResult = groupResult || clauseResult;
        } else {
            // Previous clause was AND'd - finish previous group
            passes = passes && groupResult;
            if (!passes) break; // Short circuit
            groupResult = clauseResult;
        }
    }
    // Don't forget the last group
    if (clause_count > 0) passes = passes && groupResult;
    
    // Initialize local values based on aggregation type
    if (aggType == 0) {
        // COUNT
        localVals[tid] = passes ? 1.0f : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    } else if (aggType == 3) {
        // MIN
        localVals[tid] = passes ? target_val : FLT_MAX;
    } else if (aggType == 4) {
        // MAX
        localVals[tid] = passes ? target_val : -FLT_MAX;
    } else {
        // SUM or AVG
        localVals[tid] = passes ? target_val : 0.0f;
        localCounts[tid] = passes ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (aggType == 3) {
                // MIN
                localVals[tid] = min(localVals[tid], localVals[tid + stride]);
            } else if (aggType == 4) {
                // MAX
                localVals[tid] = max(localVals[tid], localVals[tid + stride]);
            } else {
                // COUNT, SUM, AVG
                localVals[tid] += localVals[tid + stride];
                if (aggType == 0 || aggType == 2) {
                    localCounts[tid] += localCounts[tid + stride];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 in each threadgroup atomically updates global result
    if (tid == 0) {
        union { uint32_t u; float f; } conv;
        conv.f = localVals[0];
        
        if (aggType == 3) {
            // MIN: atomic min
            uint expected = atomic_load_explicit(out_result_bits, memory_order_relaxed);
            while (true) {
                union { uint32_t u; float f; } current;
                current.u = expected;
                float new_val = min(current.f, conv.f);
                union { uint32_t u; float f; } new_conv;
                new_conv.f = new_val;
                if (atomic_compare_exchange_weak_explicit(out_result_bits, &expected, new_conv.u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        } else if (aggType == 4) {
            // MAX: atomic max
            uint expected = atomic_load_explicit(out_result_bits, memory_order_relaxed);
            while (true) {
                union { uint32_t u; float f; } current;
                current.u = expected;
                float new_val = max(current.f, conv.f);
                union { uint32_t u; float f; } new_conv;
                new_conv.f = new_val;
                if (atomic_compare_exchange_weak_explicit(out_result_bits, &expected, new_conv.u,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        } else {
            // COUNT, SUM, AVG: atomic add
            atomicAddF32Bits(out_result_bits, conv.f);
        }
        
        if (aggType == 0 || aggType == 2) {
            atomic_fetch_add_explicit(out_count, localCounts[0], memory_order_relaxed);
        }
    }
}

} // namespace ops
