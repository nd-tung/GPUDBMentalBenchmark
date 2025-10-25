#include <metal_stdlib>
using namespace metal;

// --- SELECTION KERNELS ---
// SELECT * FROM lineitem WHERE column < filterValue;
kernel void selection_kernel(const device int  *inData,        // Input data column
                           device uint *result,       // Output bitmap (0 or 1)
                           constant int &filterValue, // The value to compare against
                           uint index [[thread_position_in_grid]]) {

    // Each thread performs one comparison
    if (inData[index] <= filterValue) {
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
                              constant uint& dataSize, 
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
// TPC-H Query 1: Pricing Summary Report Query
/*
SELECT 
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
*/

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

// HYBRID MODE: Append-only emission of per-row contributions for Q1.
// This mirrors Q3's robust approach to avoid contention and nondeterminism.
kernel void q1_emit_selected_kernel(
    const device uint* selection_bitmap,
    const device char* l_returnflag,
    const device char* l_linestatus,
    const device float* l_quantity,
    const device float* l_extendedprice,
    const device float* l_discount,
    const device float* l_tax,
    device Q1Aggregates_Local* out_results,   // Append-only buffer of per-row contributions
    device atomic_uint* out_count,            // Global atomic counter
    constant uint& data_size,
    constant uint& out_capacity,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint grid_size = threads_per_group * 2048;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < data_size; i += grid_size) {
        if (selection_bitmap[i] == 1) {
            // Compute packed group key (returnflag, linestatus)
            int key = ((int)l_returnflag[i] << 8) | (int)l_linestatus[i];
            // Compute per-row contributions
            Q1Aggregates_Local r;
            r.key = key;
            r.sum_qty = l_quantity[i];
            r.sum_base_price = l_extendedprice[i];
            float disc_factor = 1.0f - l_discount[i];
            r.sum_disc_price = l_extendedprice[i] * disc_factor;
            r.sum_charge = r.sum_disc_price * (1.0f + l_tax[i]);
            r.sum_discount = l_discount[i];
            r.count = 1u;

            // Append to global buffer if capacity allows
            uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (idx < out_capacity) {
                out_results[idx] = r;
            }
        }
    }
}

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

    // Initialize the private hash table. Explicitly zero all fields to avoid garbage.
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].key = -1;
        local_ht[i].sum_qty = 0.0f;
        local_ht[i].sum_base_price = 0.0f;
        local_ht[i].sum_disc_price = 0.0f;
        local_ht[i].sum_charge = 0.0f;
        local_ht[i].sum_discount = 0.0f;
        local_ht[i].count = 0u;
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


// --- TPC-H Q6 KERNELS ---
// SELECT SUM(l_extendedprice * l_discount) AS revenue
// FROM lineitem
// WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' 
//   AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;

// Stage 1: Filter and compute partial revenue sums per threadgroup
kernel void q6_filter_and_sum_stage1(
    const device int* l_shipdate,        // Date as YYYYMMDD integer
    const device float* l_discount,      // Discount factor 
    const device float* l_quantity,      // Quantity
    const device float* l_extendedprice, // Extended price
    device float* partial_revenues,      // Output: partial sums per threadgroup
    constant uint& data_size,
    constant int& start_date,            // 19940101 (1994-01-01)
    constant int& end_date,              // 19950101 (1995-01-01)
    constant float& min_discount,        // 0.05
    constant float& max_discount,        // 0.07
    constant float& max_quantity,        // 24.0
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    // 1. Each thread computes a local revenue sum
    float local_revenue = 0.0f;
    uint grid_size = threads_per_group * 2048; // Total threads in the grid
    
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
         index < data_size;
         index += grid_size) {
        
        // Apply all filter conditions
        if (l_shipdate[index] >= start_date && 
            l_shipdate[index] < end_date &&
            l_discount[index] >= min_discount && 
            l_discount[index] <= max_discount &&
            l_quantity[index] < max_quantity) {
            
            // Calculate revenue for this qualifying row
            local_revenue += l_extendedprice[index] * l_discount[index];
        }
    }
    
    // 2. Reduce within the threadgroup using shared memory
    threadgroup float shared_memory[1024];
    shared_memory[thread_id_in_group] = local_revenue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform reduction within threadgroup
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_id_in_group < stride) {
            shared_memory[thread_id_in_group] += shared_memory[thread_id_in_group + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. The first thread in each group writes the partial revenue sum
    if (thread_id_in_group == 0) {
        partial_revenues[group_id] = shared_memory[0];
    }
}

// Stage 2: Reduce partial revenue sums to final result
kernel void q6_final_sum_stage2(
    const device float* partial_revenues,
    device float* final_revenue,
    uint index [[thread_position_in_grid]])
{
    // A single thread sums all partial revenues to get final result
    if (index == 0) {
        float total_revenue = 0.0f;
        // Sum all partial revenues from stage 1 (2048 threadgroups)
        for (int i = 0; i < 2048; ++i) {
            total_revenue += partial_revenues[i];
        }
        final_revenue[0] = total_revenue;
    }
}


// --- TPC-H Q3 KERNELS ---
// TPC-H Query 3: Shipping Priority Query
/*
SELECT 
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < '1995-03-15'
  AND l_shipdate > '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
*/

//// Struct to hold aggregated results for each order
//struct Q3Aggregates {
//    atomic_int orderkey;
//    atomic_float revenue;
//    atomic_int orderdate;
//    atomic_int shippriority;
//};
//
//// Non-atomic version for local aggregation
//struct Q3Aggregates_Local {
//    int orderkey;
//    float revenue;
//    int orderdate;
//    int shippriority;
//};
//
//// Stage 1: Build hash table from orders table with customer join
//kernel void q3_build_orders_kernel(
//    const device int* o_orderkey,
//    const device int* o_custkey,
//    const device int* o_orderdate,
//    const device int* o_shippriority,
//    const device int* c_custkey,
//    const device char* c_mktsegment,
//    device HashTableEntry* orders_hashtable,
//    device int* order_metadata,  // Store orderdate and shippriority
//    constant uint& orders_size,
//    constant uint& customers_size,
//    constant uint& hashtable_size,
//    constant int& cutoff_date,   // 19950315 (1995-03-15)
//    uint index [[thread_position_in_grid]])
//{
//    if (index >= orders_size) return;
//
//    int orderkey = o_orderkey[index];
//    int custkey = o_custkey[index];
//    int orderdate = o_orderdate[index];
//    int shippriority = o_shippriority[index];
//
//    // Filter: o_orderdate < '1995-03-15'
//    if (orderdate >= cutoff_date) return;
//
//    // Find matching customer with c_mktsegment = 'BUILDING'
//    bool customer_matches = false;
//    for (uint i = 0; i < customers_size; ++i) {
//        if (c_custkey[i] == custkey && c_mktsegment[i] == 'B') { // 'B' for BUILDING
//            customer_matches = true;
//            break;
//        }
//    }
//
//    if (!customer_matches) return;
//
//    // Insert into hash table
//    uint hash_index = (uint)orderkey % hashtable_size;
//    for (uint i = 0; i < hashtable_size; ++i) {
//        uint probe_index = (hash_index + i) % hashtable_size;
//        
//        int expected = -1;
//        if (atomic_compare_exchange_weak_explicit(&orders_hashtable[probe_index].key,
//                                                  &expected, orderkey,
//                                                  memory_order_relaxed,
//                                                  memory_order_relaxed)) {
//            atomic_store_explicit(&orders_hashtable[probe_index].value, (int)index, memory_order_relaxed);
//            // Store metadata (orderdate and shippriority) at the same index
//            order_metadata[index * 2] = orderdate;
//            order_metadata[index * 2 + 1] = shippriority;
//            return;
//        }
//    }
//}
//
//// Stage 2: Probe with lineitem and perform local aggregation
//kernel void q3_probe_and_aggregate_kernel(
//    const device int* l_orderkey,
//    const device int* l_shipdate,
//    const device float* l_extendedprice,
//    const device float* l_discount,
//    const device HashTableEntry* orders_hashtable,
//    const device int* order_metadata,
//    device Q3Aggregates_Local* intermediate_results,
//    constant uint& lineitem_size,
//    constant uint& hashtable_size,
//    constant int& cutoff_date,   // 19950315 (1995-03-15)
//    uint group_id [[threadgroup_position_in_grid]],
//    uint thread_id_in_group [[thread_index_in_threadgroup]],
//    uint threads_per_group [[threads_per_threadgroup]])
//{
//    // Local hash table for this threadgroup
//    const int local_ht_size = 32;
//    thread Q3Aggregates_Local local_ht[local_ht_size];
//
//    // Initialize local hash table
//    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
//        local_ht[i].orderkey = -1;
//        local_ht[i].revenue = 0.0f;
//        local_ht[i].orderdate = 0;
//        local_ht[i].shippriority = 0;
//    }
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//    // Process lineitem rows
//    uint start_index = (group_id * threads_per_group) + thread_id_in_group;
//    uint grid_size = threads_per_group * 2048;
//
//    for (uint i = start_index; i < lineitem_size; i += grid_size) {
//        int orderkey = l_orderkey[i];
//        int shipdate = l_shipdate[i];
//        
//        // Filter: l_shipdate > '1995-03-15'
//        if (shipdate <= cutoff_date) continue;
//
//        // Probe orders hash table
//        uint hash_index = (uint)orderkey % hashtable_size;
//        int orders_row_index = -1;
//        
//        for (uint j = 0; j < hashtable_size; ++j) {
//            uint probe_index = (hash_index + j) % hashtable_size;
//            int table_key = atomic_load_explicit(&orders_hashtable[probe_index].key, memory_order_relaxed);
//            
//            if (table_key == orderkey) {
//                orders_row_index = atomic_load_explicit(&orders_hashtable[probe_index].value, memory_order_relaxed);
//                break;
//            }
//            if (table_key == -1) break;
//        }
//
//        if (orders_row_index == -1) continue;
//
//        // Calculate revenue for this lineitem
//        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
//        int orderdate = order_metadata[orders_row_index * 2];
//        int shippriority = order_metadata[orders_row_index * 2 + 1];
//
//        // Aggregate in local hash table
//        uint local_hash = (uint)orderkey % local_ht_size;
//        for (int k = 0; k < local_ht_size; ++k) {
//            uint local_probe = (local_hash + k) % local_ht_size;
//            
//            if (local_ht[local_probe].orderkey == -1 || local_ht[local_probe].orderkey == orderkey) {
//                local_ht[local_probe].orderkey = orderkey;
//                local_ht[local_probe].revenue += revenue;
//                local_ht[local_probe].orderdate = orderdate;
//                local_ht[local_probe].shippriority = shippriority;
//                break;
//            }
//        }
//    }
//    
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//    
//    // Write local results to global memory
//    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
//        if (local_ht[i].orderkey != -1) {
//            intermediate_results[group_id * local_ht_size + i] = local_ht[i];
//        }
//    }
//}
//
//// Stage 3: Merge intermediate results into final hash table
//kernel void q3_merge_results_kernel(
//    const device Q3Aggregates_Local* intermediate_results,
//    device Q3Aggregates* final_hashtable,
//    constant uint& intermediate_size,
//    constant uint& final_hashtable_size,
//    uint index [[thread_position_in_grid]])
//{
//    if (index >= intermediate_size) return;
//
//    Q3Aggregates_Local local_result = intermediate_results[index];
//    if (local_result.orderkey == -1) return;
//
//    uint hash_index = (uint)local_result.orderkey % final_hashtable_size;
//    
//    for (uint i = 0; i < final_hashtable_size; ++i) {
//        uint probe_index = (hash_index + i) % final_hashtable_size;
//        
//        int expected = -1;
//        if (atomic_compare_exchange_weak_explicit(&final_hashtable[probe_index].orderkey,
//                                                  &expected, local_result.orderkey,
//                                                  memory_order_relaxed, memory_order_relaxed)) {
//            // Initialize new entry
//            atomic_store_explicit(&final_hashtable[probe_index].revenue, 0.0f, memory_order_relaxed);
//            atomic_store_explicit(&final_hashtable[probe_index].orderdate, local_result.orderdate, memory_order_relaxed);
//            atomic_store_explicit(&final_hashtable[probe_index].shippriority, local_result.shippriority, memory_order_relaxed);
//        }
//        
//        if (atomic_load_explicit(&final_hashtable[probe_index].orderkey, memory_order_relaxed) == local_result.orderkey) {
//            // Accumulate revenue
//            atomic_fetch_add_explicit(&final_hashtable[probe_index].revenue, local_result.revenue, memory_order_relaxed);
//            return;
//        }
//    }
//}


// Struct for the final aggregation results for Q3
struct Q3Aggregates {
    atomic_int key; // orderkey
    atomic_float revenue;
    atomic_uint orderdate;
    atomic_uint shippriority;
};

// A non-atomic version for fast local aggregation
struct Q3Aggregates_Local {
    int key;
    float revenue;
    uint orderdate;
    uint shippriority;
};


// KERNEL 1: Build a hash table on the CUSTOMER table.
// (No changes needed, this kernel is correct)
kernel void q3_build_customer_ht_kernel(
    const device int* c_custkey,
    const device char* c_mktsegment,
    device HashTableEntry* customer_ht,
    constant uint& customer_size,
    constant uint& customer_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= customer_size || c_mktsegment[index] != 'B') { // 'B' for BUILDING
        return;
    }

    int key = c_custkey[index];
    int value = 1; // We only need to know the key exists, so value can be a simple flag.

    uint hash_index = (uint)key % customer_ht_size;
    for (uint i = 0; i < customer_ht_size; ++i) {
        uint probe_index = (hash_index + i) % customer_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&customer_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&customer_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}


// KERNEL 2: Build a hash table on the ORDERS table. (CORRECTED)
// Now stores the original row index as the payload.
kernel void q3_build_orders_ht_kernel(
    const device int* o_orderkey,
    const device int* o_orderdate,
    device HashTableEntry* orders_ht,
    constant uint& orders_size,
    constant uint& orders_ht_size,
    constant int& cutoff_date, // 19950315
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size || o_orderdate[index] >= cutoff_date) {
        return;
    }
    
    int key = o_orderkey[index];
    int value = (int)index; // Store the original row index as the payload

    uint hash_index = (uint)key % orders_ht_size;
    for (uint i = 0; i < orders_ht_size; ++i) {
        uint probe_index = (hash_index + i) % orders_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&orders_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&orders_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}


// KERNEL 3: Main Probe & Aggregation Kernel (CORRECTED)
// Now correctly looks up the date and priority using the index payload.
kernel void q3_probe_and_local_agg_kernel(
    const device int* l_orderkey,
    const device int* l_shipdate,
    const device float* l_extendedprice,
    const device float* l_discount,
    const device HashTableEntry* customer_ht,
    const device HashTableEntry* orders_ht,
    // Pass the full original arrays for payload lookup
    const device int* o_custkey,
    const device int* o_orderdate,
    const device int* o_shippriority,
    device Q3Aggregates_Local* out_results,
    device atomic_uint* out_count,
    constant uint& lineitem_size,
    constant uint& customer_ht_size,
    constant uint& orders_ht_size,
    constant int& cutoff_date,
    constant uint& out_capacity,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint grid_size = threads_per_group * 2048;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < lineitem_size; i += grid_size) {
        
        if (l_shipdate[i] <= cutoff_date) continue;

        int orderkey = l_orderkey[i];
        
        uint o_hash = (uint)orderkey % orders_ht_size;
        int orders_idx = -1;
        
        for (uint j = 0; j < orders_ht_size; ++j) {
            uint probe_idx = (o_hash + j) % orders_ht_size;
            int o_key = atomic_load_explicit(&orders_ht[probe_idx].key, memory_order_relaxed);
            if (o_key == orderkey) {
                orders_idx = atomic_load_explicit(&orders_ht[probe_idx].value, memory_order_relaxed);
                break;
            }
            if (o_key == -1) break;
        }

        if (orders_idx == -1) continue;

        int custkey = o_custkey[orders_idx];
        uint c_hash = (uint)custkey % customer_ht_size;
        bool customer_match = false;
        for (uint j = 0; j < customer_ht_size; ++j) {
            uint probe_idx = (c_hash + j) % customer_ht_size;
            int c_key = atomic_load_explicit(&customer_ht[probe_idx].key, memory_order_relaxed);
             if (c_key == custkey) {
                customer_match = true;
                break;
            }
            if (c_key == -1) break;
        }

        if (!customer_match) continue;
        
        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        // Append one record per matching lineitem
        uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
        if (idx < out_capacity) {
            Q3Aggregates_Local r;
            r.key = orderkey;
            r.revenue = revenue;
            r.orderdate = (uint)o_orderdate[orders_idx];
            r.shippriority = (uint)o_shippriority[orders_idx];
            out_results[idx] = r;
        }
    }
    // No threadgroup reduction; results are appended globally
}


// KERNEL 4: The final merge kernel - Fixed to handle multiple contributors
kernel void q3_merge_results_kernel(
    const device Q3Aggregates_Local* intermediate_results,
    device Q3Aggregates* final_hashtable,
    constant uint& intermediate_size,
    constant uint& final_hashtable_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= intermediate_size) return;

    Q3Aggregates_Local local_result = intermediate_results[index];
    if (local_result.key == -1) return;

    uint hash_index = (uint)local_result.key % final_hashtable_size;
    
    for (uint i = 0; i < final_hashtable_size; ++i) {
        uint probe_index = (hash_index + i) % final_hashtable_size;
        
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&final_hashtable[probe_index].key, &expected, local_result.key, memory_order_relaxed, memory_order_relaxed)) {
            // Successfully claimed this slot - initialize with our values
            atomic_store_explicit(&final_hashtable[probe_index].revenue, local_result.revenue, memory_order_relaxed);
            atomic_store_explicit(&final_hashtable[probe_index].orderdate, local_result.orderdate, memory_order_relaxed);
            atomic_store_explicit(&final_hashtable[probe_index].shippriority, local_result.shippriority, memory_order_relaxed);
            return;
        }
        
        // Slot is occupied - check if it's our key
        int current_key = atomic_load_explicit(&final_hashtable[probe_index].key, memory_order_relaxed);
        if (current_key == local_result.key) {
            // Found our key - add our revenue to it
            atomic_fetch_add_explicit(&final_hashtable[probe_index].revenue, local_result.revenue, memory_order_relaxed);
            return;
        }
        // else: collision with different key, continue probing
    }
}


// --- TPC-H Q9 KERNELS ---
// TPC-H Query 9: Product Type Profit Measure Query
/*
SELECT 
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM (
    SELECT 
        n_name AS nation,
        EXTRACT(year FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey
      AND ps_suppkey = l_suppkey
      AND ps_partkey = l_partkey
      AND p_partkey = l_partkey
      AND o_orderkey = l_orderkey
      AND s_nationkey = n_nationkey
      AND p_name LIKE '%green%'
) AS profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC;
*/

// Struct for the final aggregation results for Q9
struct Q9Aggregates {
    atomic_uint key; // Packed (nation_key << 16) | year
    atomic_float profit;
};

// A non-atomic version for fast local aggregation
struct Q9Aggregates_Local {
    uint key;
    float profit;
};

// KERNEL 1: Build HT on PART, filtering for p_name LIKE '%green%'
kernel void q9_build_part_ht_kernel(
    const device int* p_partkey,
    const device char* p_name, // Assuming p_name is a fixed-size char array
    device HashTableEntry* part_ht,
    constant uint& part_size,
    constant uint& part_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= part_size) return;
    bool match = false;
    for(int i = 0; i < 50; ++i) { // Simplified string search
        if (p_name[index * 55 + i] == 'g' && p_name[index * 55 + i + 1] == 'r' &&
            p_name[index * 55 + i + 2] == 'e' && p_name[index * 55 + i + 3] == 'e' &&
            p_name[index * 55 + i + 4] == 'n') {
            match = true;
            break;
        }
    }
    if (!match) return;

    int key = p_partkey[index];
    int value = 1; // Flag that part exists
    uint hash_index = (uint)key % part_ht_size;
    for (uint i = 0; i < part_ht_size; ++i) {
        uint probe_index = (hash_index + i) % part_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&part_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&part_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}

// KERNEL 2: Build HT on SUPPLIER, storing nationkey as the value.
kernel void q9_build_supplier_ht_kernel(
    const device int* s_suppkey,
    const device int* s_nationkey,
    device HashTableEntry* supplier_ht,
    constant uint& supplier_size,
    constant uint& supplier_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= supplier_size) return;
    int key = s_suppkey[index];
    int value = s_nationkey[index];
    uint hash_index = (uint)key % supplier_ht_size;
    for (uint i = 0; i < supplier_ht_size; ++i) {
        uint probe_index = (hash_index + i) % supplier_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&supplier_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&supplier_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}

// KERNEL 3: Build HT on PARTSUPP, storing supplycost index as value
kernel void q9_build_partsupp_ht_kernel(
    const device int* ps_partkey,
    const device int* ps_suppkey,
    device HashTableEntry* partsupp_ht,
    constant uint& partsupp_size,
    constant uint& partsupp_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= partsupp_size) return;
    // Compound key (simplification, prone to collisions)
    int key = (ps_partkey[index] * 31 + ps_suppkey[index]);
    int value = (int)index; // Store index to original ps_supplycost array
    uint hash_index = (uint)key % partsupp_ht_size;
    for (uint i = 0; i < partsupp_ht_size; ++i) {
        uint probe_index = (hash_index + i) % partsupp_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&partsupp_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&partsupp_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}

// KERNEL 4: Build HT on ORDERS, storing year as value
kernel void q9_build_orders_ht_kernel(
    const device int* o_orderkey,
    const device int* o_orderdate,
    device HashTableEntry* orders_ht,
    constant uint& orders_size,
    constant uint& orders_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    int key = o_orderkey[index];
    int value = o_orderdate[index] / 10000; // Extract year
    uint hash_index = (uint)key % orders_ht_size;
    for (uint i = 0; i < orders_ht_size; ++i) {
        uint probe_index = (hash_index + i) % orders_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&orders_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&orders_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}


// KERNEL 5: The main kernel. Streams lineitem and probes all other hash tables.
kernel void q9_probe_and_local_agg_kernel(
    // lineitem columns
    const device int* l_suppkey, const device int* l_partkey, const device int* l_orderkey,
    const device float* l_extendedprice, const device float* l_discount, const device float* l_quantity,
    // partsupp supplycost array
    const device float* ps_supplycost,
    // Pre-built hash tables
    const device HashTableEntry* part_ht, const device HashTableEntry* supplier_ht,
    const device HashTableEntry* partsupp_ht, const device HashTableEntry* orders_ht,
    // Output buffer
    device Q9Aggregates_Local* intermediate_results,
    // Parameters
    constant uint& lineitem_size, constant uint& part_ht_size, constant uint& supplier_ht_size,
    constant uint& partsupp_ht_size, constant uint& orders_ht_size,
    // Thread IDs
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    const int local_ht_size = 128;
    thread Q9Aggregates_Local local_ht[local_ht_size];
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].key = 0; local_ht[i].profit = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint grid_size = threads_per_group * 2048;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < lineitem_size; i += grid_size) {
        int partkey = l_partkey[i], suppkey = l_suppkey[i], orderkey = l_orderkey[i];

        // --- FAST PARALLEL PROBES ---
        
        // 1. Probe part_ht to check if part matches 'green' filter
        bool part_match = false;
        uint part_hash = (uint)partkey % part_ht_size;
        for (uint j = 0; j < part_ht_size; ++j) {
            uint probe_idx = (part_hash + j) % part_ht_size;
            int p_key = atomic_load_explicit(&part_ht[probe_idx].key, memory_order_relaxed);
            if (p_key == partkey) {
                part_match = true;
                break;
            }
            if (p_key == -1) break;
        }
        if (!part_match) continue;

        // 2. Probe supplier_ht to get nationkey
        int nationkey = -1;
        uint supp_hash = (uint)suppkey % supplier_ht_size;
        for (uint j = 0; j < supplier_ht_size; ++j) {
            uint probe_idx = (supp_hash + j) % supplier_ht_size;
            int s_key = atomic_load_explicit(&supplier_ht[probe_idx].key, memory_order_relaxed);
            if (s_key == suppkey) {
                nationkey = atomic_load_explicit(&supplier_ht[probe_idx].value, memory_order_relaxed);
                break;
            }
            if (s_key == -1) break;
        }
        if (nationkey == -1) continue;

        // 3. Probe partsupp_ht to get supply cost index
        int ps_idx = -1;
        int compound_key = (partkey * 31 + suppkey);
        uint ps_hash = (uint)compound_key % partsupp_ht_size;
        for (uint j = 0; j < partsupp_ht_size; ++j) {
            uint probe_idx = (ps_hash + j) % partsupp_ht_size;
            int table_key = atomic_load_explicit(&partsupp_ht[probe_idx].key, memory_order_relaxed);
            if (table_key == compound_key) {
                ps_idx = atomic_load_explicit(&partsupp_ht[probe_idx].value, memory_order_relaxed);
                break;
            }
            if (table_key == -1) break;
        }
        if (ps_idx == -1) continue;

        // 4. Probe orders_ht to get year
        int year = -1;
        uint ord_hash = (uint)orderkey % orders_ht_size;
        for (uint j = 0; j < orders_ht_size; ++j) {
            uint probe_idx = (ord_hash + j) % orders_ht_size;
            int o_key = atomic_load_explicit(&orders_ht[probe_idx].key, memory_order_relaxed);
            if (o_key == orderkey) {
                year = atomic_load_explicit(&orders_ht[probe_idx].value, memory_order_relaxed);
                break;
            }
            if (o_key == -1) break;
        }
        if (year == -1) continue;

        // All probes succeeded!
        
        // --- AGGREGATE ---
        float profit = l_extendedprice[i] * (1.0f - l_discount[i]) - ps_supplycost[ps_idx] * l_quantity[i];
        uint agg_key = (uint)(nationkey << 16) | year;
        uint agg_hash = agg_key % local_ht_size;

        for(int k = 0; k < local_ht_size; ++k) {
            uint probe_idx = (agg_hash + k) % local_ht_size;
            if(local_ht[probe_idx].key == 0 || local_ht[probe_idx].key == agg_key) {
                local_ht[probe_idx].key = agg_key;
                local_ht[probe_idx].profit += profit;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write local results to global memory
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        if (local_ht[i].key != 0) {
            intermediate_results[group_id * local_ht_size + i] = local_ht[i];
        }
    }
}


// KERNEL 6: Final merge kernel
kernel void q9_merge_results_kernel(
    const device Q9Aggregates_Local* intermediate_results,
    device Q9Aggregates* final_hashtable,
    constant uint& intermediate_size,
    constant uint& final_hashtable_size,
    uint index [[thread_position_in_grid]])
{
    // This logic from your implementation was correct and can be reused.
    if (index >= intermediate_size) return;
    Q9Aggregates_Local local_result = intermediate_results[index];
    if (local_result.key == 0) return;

    uint hash_index = local_result.key % final_hashtable_size;
    for (uint i = 0; i < final_hashtable_size; ++i) {
        uint probe_index = (hash_index + i) % final_hashtable_size;
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(&final_hashtable[probe_index].key, &expected, local_result.key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&final_hashtable[probe_index].profit, 0.0f, memory_order_relaxed);
        }
        if (atomic_load_explicit(&final_hashtable[probe_index].key, memory_order_relaxed) == local_result.key) {
            atomic_fetch_add_explicit(&final_hashtable[probe_index].profit, local_result.profit, memory_order_relaxed);
            return;
        }
    }
}

// --- TPC-H Query 13 Kernels ---
/*
SELECT
    c_count,
    COUNT(*) AS custdist
FROM (
    -- Inner Query: First, for each customer, count their non-special orders.
    SELECT
        c_custkey,
        COUNT(o_orderkey) AS c_count
    FROM
        customer
    LEFT OUTER JOIN
        orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY
        c_custkey
) AS c_orders
-- Outer Query: Then, group those results again to create a histogram.
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;

*/

// Structs for the two levels of aggregation
struct Q13_OrderCount { // Stage 1 result
    atomic_uint custkey;
    atomic_uint order_count;
};
struct Q13_OrderCount_Local {
    uint custkey;
    uint order_count;
};

struct Q13_CustDist { // Stage 2 final result
    atomic_uint c_count; // The number of orders
    atomic_uint custdist; // The number of customers with that many orders
};
struct Q13_CustDist_Local {
    uint c_count;
    uint custdist;
};


// KERNEL 1A: Stage 1, Local Count. Scans ORDERS, filters, and does first GROUP BY locally.
kernel void q13_local_count_kernel(
    const device int* o_custkey,
    const device char* o_comment, // Assuming fixed-width
    device Q13_OrderCount_Local* intermediate_counts,
    constant uint& orders_size,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    const int local_ht_size = 128;
    thread Q13_OrderCount_Local local_ht[local_ht_size];
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].custkey = 0; // 0 is invalid key
        local_ht[i].order_count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint grid_size = threads_per_group * 2048; // Assume 2048 groups
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < orders_size; i += grid_size) {
        
        // Filter: o_comment NOT LIKE '%special%requests%'
        bool match = false;
        // Simplified search for "special"
        for(int j = 0; j < 90; ++j) { // O_COMMENT is 100 chars
            if (o_comment[i * 100 + j] == 's' && o_comment[i * 100 + j + 7] == 'l') { // quick check
                match = true; // Found "special", so we should SKIP this row
                break;
            }
        }
        if (match) continue;

        // Passed filter, aggregate locally
        uint custkey = (uint)o_custkey[i];
        uint hash_index = custkey % local_ht_size;
        for (int k = 0; k < local_ht_size; ++k) {
            uint probe_idx = (hash_index + k) % local_ht_size;
            if (local_ht[probe_idx].custkey == 0 || local_ht[probe_idx].custkey == custkey) {
                local_ht[probe_idx].custkey = custkey;
                local_ht[probe_idx].order_count++;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write local results to global memory
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        if (local_ht[i].custkey != 0) {
            intermediate_counts[group_id * local_ht_size + i] = local_ht[i];
        }
    }
}


// KERNEL 1B: Stage 1, Merge Count. Merges partial counts into a global HT.
kernel void q13_merge_counts_kernel(
    const device Q13_OrderCount_Local* intermediate_counts,
    device Q13_OrderCount* customer_order_counts_ht,
    constant uint& intermediate_size,
    constant uint& final_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= intermediate_size) return;
    Q13_OrderCount_Local local_result = intermediate_counts[index];
    if (local_result.custkey == 0) return;

    uint hash_index = local_result.custkey % final_ht_size;
    for (uint i = 0; i < final_ht_size; ++i) {
        uint probe_index = (hash_index + i) % final_ht_size;
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(&customer_order_counts_ht[probe_index].custkey, &expected, local_result.custkey, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&customer_order_counts_ht[probe_index].order_count, 0, memory_order_relaxed);
        }
        if (atomic_load_explicit(&customer_order_counts_ht[probe_index].custkey, memory_order_relaxed) == local_result.custkey) {
            atomic_fetch_add_explicit(&customer_order_counts_ht[probe_index].order_count, local_result.order_count, memory_order_relaxed);
            return;
        }
    }
}


// KERNEL 2A: Stage 2, Local Histogram. Scans CUSTOMER, probes counts HT, builds local histogram.
kernel void q13_local_histogram_kernel(
    const device int* c_custkey,
    const device Q13_OrderCount* customer_order_counts_ht,
    device Q13_CustDist_Local* intermediate_histograms,
    constant uint& customer_size,
    constant uint& counts_ht_size,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    const int local_ht_size = 32; // Max expected order count is small
    thread Q13_CustDist_Local local_ht[local_ht_size];
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].c_count = i;
        local_ht[i].custdist = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint grid_size = threads_per_group * 2048;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < customer_size; i += grid_size) {
        uint custkey = (uint)c_custkey[i];
        
        // Probe the counts HT to find this customer's order count
        uint hash_index = custkey % counts_ht_size;
        uint order_count = 0; // Default to 0 for LEFT JOIN
        for (uint j = 0; j < counts_ht_size; ++j) {
            uint probe_idx = (hash_index + j) % counts_ht_size;
            uint ht_key = atomic_load_explicit(&customer_order_counts_ht[probe_idx].custkey, memory_order_relaxed);
            if (ht_key == custkey) {
                order_count = atomic_load_explicit(&customer_order_counts_ht[probe_idx].order_count, memory_order_relaxed);
                break;
            }
            if (ht_key == 0) break; // Reached empty slot
        }
        
        // Update local histogram
        if (order_count < local_ht_size) {
            local_ht[order_count].custdist++;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        if (local_ht[i].custdist > 0) {
            intermediate_histograms[group_id * local_ht_size + i] = local_ht[i];
        }
    }
}


// KERNEL 2B: Stage 2, Merge Histogram. Merges partial histograms into final result.
kernel void q13_merge_histogram_kernel(
    const device Q13_CustDist_Local* intermediate_histograms,
    device Q13_CustDist* final_histogram_ht,
    constant uint& intermediate_size,
    constant uint& final_ht_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= intermediate_size) return;
    Q13_CustDist_Local local_result = intermediate_histograms[index];
    if (local_result.custdist == 0) return;

    uint c_count = local_result.c_count;
    uint hash_index = c_count % final_ht_size;

    for (uint i = 0; i < final_ht_size; ++i) {
        uint probe_index = (hash_index + i) % final_ht_size;
        uint expected = -1; // Use -1 as empty marker
        if (atomic_compare_exchange_weak_explicit(&final_histogram_ht[probe_index].c_count, &expected, c_count, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&final_histogram_ht[probe_index].custdist, 0, memory_order_relaxed);
        }
        if (atomic_load_explicit(&final_histogram_ht[probe_index].c_count, memory_order_relaxed) == c_count) {
            atomic_fetch_add_explicit(&final_histogram_ht[probe_index].custdist, local_result.custdist, memory_order_relaxed);
            return;
        }
    }
}
