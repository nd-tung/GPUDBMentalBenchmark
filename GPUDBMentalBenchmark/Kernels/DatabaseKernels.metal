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
                              uint group_id [[threadgroup_position_in_grid]],
                              uint thread_id_in_group [[thread_index_in_threadgroup]],
                              uint threads_per_group [[threads_per_threadgroup]])
{
    // 1. Each thread computes a local sum from its assigned elements in global memory.
    // A grid-stride loop allows each thread to process multiple elements.
    float local_sum = 0.0f;
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
              index < 60000000; // Total number of elements in the input data.
              index += threads_per_group * 2048) { // Stride is based on the total number of threads in the grid.
        if(index < 60000000) { // Bounds check to prevent reading past the end of the input buffer.
           local_sum += inData[index];
        }
    }
    
    // 2. Perform a parallel reduction within the threadgroup using shared memory.
    threadgroup float shared_memory[1024]; // Max threadgroup size on Apple Silicon.
    shared_memory[thread_id_in_group] = local_sum;

    // Synchronize threads within the group to ensure all writes to shared memory are complete.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform a parallel sum reduction on the data in shared memory.
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_id_in_group < stride) {
            shared_memory[thread_id_in_group] += shared_memory[thread_id_in_group + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Thread 0 of each group writes the group's partial sum to global memory.
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
