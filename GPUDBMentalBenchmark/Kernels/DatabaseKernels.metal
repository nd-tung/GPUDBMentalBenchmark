#include <metal_stdlib>
using namespace metal;

// --- SELECTION KERNELS ---
// SELECT * FROM lineitem WHERE column < filterValue;
kernel void selection_kernel(const device int  *inData,        // Input data column
                           device uint *result,       // Output bitmap (0 or 1)
                           constant int &filterValue, // The value to compare against
                           uint index [[thread_position_in_grid]]) {

    // Each thread performs one comparison
    if (inData[index] < filterValue) {
        result[index] = 1;
    } else {
        result[index] = 0;
    }
}


// --- AGGREGATION KERNELS ---
// SELECT SUM(l_quantity) FROM lineitem;

// Stage 1: Reduces a partition of the input data into a single partial sum per threadgroup.
kernel void sum_kernel_stage1(const device float* inData,
                              device float* partialSums,
                              constant uint& dataSize, // <-- NEW: Pass in the actual data size
                              uint group_id [[threadgroup_position_in_grid]],
                              uint thread_id_in_group [[thread_index_in_threadgroup]],
                              uint threads_per_group [[threads_per_threadgroup]])
{
    // 1. Each thread computes a local sum
    float local_sum = 0.0f;
    uint grid_size = threads_per_group * 2048; // Total threads in the grid
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
              index < dataSize;
              index += grid_size) {
        local_sum += inData[index];
    }
    
    // 2. Reduce within the threadgroup using shared memory. (This part was already correct)
    threadgroup float shared_memory[1024];
    shared_memory[thread_id_in_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_id_in_group < stride) {
            shared_memory[thread_id_in_group] += shared_memory[thread_id_in_group + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. The first thread in each group writes the final partial sum.
    if (thread_id_in_group == 0) {
        partialSums[group_id] = shared_memory[0];
    }
}


// Stage 2: Reduces the buffer of partial sums into a final single result.
kernel void sum_kernel_stage2(const device float* partialSums,
                              device float* finalResult,
                              uint index [[thread_position_in_grid]])
{
    // A single thread iterates through the partial sums and calculates the final total.
    if (index == 0) {
        float total_sum = 0.0f;
        // The number of partial sums must match the number of threadgroups from Stage 1.
        for(int i = 0; i < 2048; ++i) {
            total_sum += partialSums[i];
        }
        finalResult[0] = total_sum;
    }
}

// --- HASH JOIN KERNELS ---
// SELECT * FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey;

// Represents an entry in our simple hash table
struct HashTableEntry {
    atomic_int key;   // o_orderkey
    atomic_int value; // Row ID (payload)
};

// Phase 1: Builds a hash table from the smaller table (orders)
kernel void hash_join_build(const device int* inKeys,      // Input: o_orderkey
                            const device int* inValues,    // Input: Row IDs
                            device HashTableEntry* hashTable,
                            constant uint& dataSize,
                            constant uint& hashTableSize,
                            uint index [[thread_position_in_grid]])
{
    if (index >= dataSize) {
        return;
    }

    int key = inKeys[index];
    int value = inValues[index];

    // Simple hash function
    uint hash_index = (uint)key % hashTableSize;

    // Linear probing with atomic operations to handle collisions
    for (uint i = 0; i < hashTableSize; ++i) {
        uint probe_index = (hash_index + i) % hashTableSize;

        // Try to insert the key if the slot is empty (key == -1)
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&hashTable[probe_index].key,
                                                    &expected,
                                                    key,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            // If we successfully claimed the slot, write the value
            atomic_store_explicit(&hashTable[probe_index].value, value, memory_order_relaxed);
            return; // Exit the loop after successful insertion
        }
    }
}

// Phase 2: Probes the hash table using keys from the larger table (lineitem)
kernel void hash_join_probe(const device int* probeKeys,        // Input: l_orderkey
                            const device HashTableEntry* hashTable,
                            device atomic_uint* match_count, // Output: counter for successful joins
                            constant uint& probeDataSize,
                            constant uint& hashTableSize,
                            uint index [[thread_position_in_grid]])
{
    if (index >= probeDataSize) {
        return;
    }

    int key_to_find = probeKeys[index];

    // Simple hash function (must be identical to the build phase)
    uint hash_index = (uint)key_to_find % hashTableSize;

    // Linear probing to find the key
    for (uint i = 0; i < hashTableSize; ++i) {
        uint probe_index = (hash_index + i) % hashTableSize;
        
        int table_key = atomic_load_explicit(&hashTable[probe_index].key, memory_order_relaxed);

        // If we find our key, we have a match.
        if (table_key == key_to_find) {
            atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            return; // Found a match, this thread is done.
        }

        // If we find an empty slot, the key is not in the table.
        if (table_key == -1) {
            return; // Key not found, this thread is done.
        }
    }
}


// --- TPC-H Q1 KERNELS ---

// This struct holds all the running totals for a single group in our aggregation.
// Only the final global hash table needs atomics.
struct Q1Aggregates {
    atomic_int   key;
    atomic_float sum_qty;
    atomic_float sum_base_price;
    atomic_float sum_disc_price;
    atomic_float sum_charge;
    atomic_float sum_discount;
    atomic_uint  count;
};

// A non-atomic version for use in fast threadgroup memory
struct Q1Aggregates_Local {
    int   key;
    float sum_qty;
    float sum_base_price;
    float sum_disc_price;
    float sum_charge;
    float sum_discount;
    uint  count;
};

// STAGE 1: Each threadgroup creates its own private hash table in threadgroup memory
// and performs a local aggregation.
kernel void q1_local_aggregation_kernel(
    const device uint* selection_bitmap,
    const device char* l_returnflag,
    const device char* l_linestatus,
    const device float* l_quantity,
    const device float* l_extendedprice,
    const device float* l_discount,
    const device float* l_tax,
    device Q1Aggregates_Local* intermediate_results, // Output buffer for all local results
    constant uint& data_size,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    // 1. Create a private hash table for this threadgroup.
    const int local_ht_size = 16;
    thread Q1Aggregates_Local local_ht[local_ht_size];

    // Initialize the private hash table.
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].key = -1;
        local_ht[i].count = 0;
        // other fields are implicitly zeroed by their accumulation logic
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Each thread processes its assigned rows and updates the LOCAL hash table.
    uint start_index = (group_id * threads_per_group) + thread_id_in_group;
    uint grid_size = threads_per_group * 2048; // Assume 2048 groups, adjust as needed

    for (uint i = start_index; i < data_size; i += grid_size) {
        if (selection_bitmap[i] == 1) {
            int key = (l_returnflag[i] << 8) | l_linestatus[i];
            uint hash_index = (uint)key % local_ht_size;

            // Simple linear probe on the local hash table. No atomics needed!
            for (int j = 0; j < local_ht_size; ++j) {
                uint probe_index = (hash_index + j) % local_ht_size;
                if (local_ht[probe_index].key == -1 || local_ht[probe_index].key == key) {
                    local_ht[probe_index].key = key;
                    local_ht[probe_index].sum_qty += l_quantity[i];
                    local_ht[probe_index].sum_base_price += l_extendedprice[i];
                    local_ht[probe_index].sum_disc_price += l_extendedprice[i] * (1.0f - l_discount[i]);
                    local_ht[probe_index].sum_charge += l_extendedprice[i] * (1.0f - l_discount[i]) * (1.0f + l_tax[i]);
                    local_ht[probe_index].sum_discount += l_discount[i];
                    local_ht[probe_index].count++;
                    break;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Each thread writes its portion of the local hash table to global memory.
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        if (local_ht[i].key != -1) {
            intermediate_results[group_id * local_ht_size + i] = local_ht[i];
        }
    }
}


// STAGE 2: A second kernel merges the many small local results into one final hash table.
kernel void q1_merge_kernel(
    const device Q1Aggregates_Local* intermediate_results,
    device Q1Aggregates* final_hash_table,
    constant uint& intermediate_data_size,
    constant uint& final_hash_table_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= intermediate_data_size) {
        return;
    }

    Q1Aggregates_Local local_result = intermediate_results[index];
    if (local_result.key == -1) {
        return;
    }

    int key = local_result.key;
    uint hash_index = (uint)key % final_hash_table_size;

    // Probe and update the FINAL hash table using atomics.
    // Contention is much lower here because the number of items to merge is small.
    for (uint i = 0; i < final_hash_table_size; ++i) {
        uint probe_index = (hash_index + i) % final_hash_table_size;

        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&final_hash_table[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            // Success! This thread is the first to claim this slot for this group.
            // Atomically initialize all aggregate values to zero before any additions happen.
            atomic_store_explicit(&final_hash_table[probe_index].sum_qty, 0.0f, memory_order_relaxed);
            atomic_store_explicit(&final_hash_table[probe_index].sum_base_price, 0.0f, memory_order_relaxed);
            atomic_store_explicit(&final_hash_table[probe_index].sum_disc_price, 0.0f, memory_order_relaxed);
            atomic_store_explicit(&final_hash_table[probe_index].sum_charge, 0.0f, memory_order_relaxed);
            atomic_store_explicit(&final_hash_table[probe_index].sum_discount, 0.0f, memory_order_relaxed);
            atomic_store_explicit(&final_hash_table[probe_index].count, 0, memory_order_relaxed);
        }

        if (atomic_load_explicit(&final_hash_table[probe_index].key, memory_order_relaxed) == key) {
            // This is the right group. Atomically update all the aggregates.
            atomic_fetch_add_explicit(&final_hash_table[probe_index].sum_qty, local_result.sum_qty, memory_order_relaxed);
            atomic_fetch_add_explicit(&final_hash_table[probe_index].sum_base_price, local_result.sum_base_price, memory_order_relaxed);
            atomic_fetch_add_explicit(&final_hash_table[probe_index].sum_disc_price, local_result.sum_disc_price, memory_order_relaxed);
            atomic_fetch_add_explicit(&final_hash_table[probe_index].sum_charge, local_result.sum_charge, memory_order_relaxed);
            atomic_fetch_add_explicit(&final_hash_table[probe_index].sum_discount, local_result.sum_discount, memory_order_relaxed);
            atomic_fetch_add_explicit(&final_hash_table[probe_index].count, local_result.count, memory_order_relaxed);
            return; // Done
        }
    }
}
