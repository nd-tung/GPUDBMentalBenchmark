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
