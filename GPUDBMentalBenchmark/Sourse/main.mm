#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

// --- Helper to Load Integer Column ---
std::vector<int> loadIntColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stoi(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// --- Helper to Load Float Column ---
std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stof(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// Helper to Load char columns
std::vector<char> loadCharColumn(const std::string& filePath, int columnIndex) {
    std::vector<char> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(token[0]); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// Helper to Load date columns (as integers for simplicity, e.g., 19980315)
std::vector<int> loadDateColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) {
                token = line.substr(start, end - start);
                token.erase(std::remove(token.begin(), token.end(), '-'), token.end());
                data.push_back(std::stoi(token));
                break;
            }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}


// --- Selection Benchmark Test Function ---
void runSingleSelectionTest(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLComputePipelineState> pipelineState,
                            id<MTLBuffer> inBuffer, id<MTLBuffer> resultBuffer,
                            const std::vector<int>& cpuData, int filterValue) {
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:inBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:resultBuffer offset:0 atIndex:1];
    [commandEncoder setBytes:&filterValue length:sizeof(filterValue) atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(cpuData.size(), 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > cpuData.size()) { threadGroupSize = cpuData.size(); }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer.contents;
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) { if (resultData[i] == 1) { passCount++; } }
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    NSLog(@"--- Filter Value: < %d ---", filterValue);
    NSLog(@"Selectivity: %.2f%% (%u rows matched)", selectivity, passCount);
    NSLog(@"GPU execution time: %f ms", gpuExecutionTime * 1000.0);
    NSLog(@"Effective Bandwidth: %.2f GB/s\n", bandwidth);
}

// --- Main Function for Selection Benchmark ---
void runSelectionBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running Selection Benchmark ---");

    //Select tpch data file
    std::vector<int> cpuData = loadIntColumn("Data/SF-10/lineitem.tbl", 1);
    if (cpuData.empty()) { return; }
    NSLog(@"Loaded %lu rows for selection.", cpuData.size());

    NSError *error = nil;
    id<MTLFunction> selectionFunction = [library newFunctionWithName:@"selection_kernel"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:selectionFunction error:&error];
    if (!pipelineState) { NSLog(@"Failed to create selection pipeline state: %@", error); return; }

    const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
    id<MTLBuffer> inBuffer = [device newBufferWithBytes:cpuData.data() length:dataSizeBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:cpuData.size() * sizeof(unsigned int) options:MTLResourceStorageModeShared];

    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
}


// --- Main Function for Aggregation Benchmark ---
void runAggregationBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running Aggregation Benchmark ---");

    //Select tpch data file
    std::vector<float> cpuData = loadFloatColumn("Data/SF-10/lineitem.tbl", 4);
    if (cpuData.empty()) return;
    NSLog(@"Loaded %lu rows for aggregation.", cpuData.size());
    const unsigned long dataSizeBytes = cpuData.size() * sizeof(float);
    uint dataSize = (uint)cpuData.size(); // The actual number of elements

    NSError *error = nil;
    id<MTLFunction> stage1Function = [library newFunctionWithName:@"sum_kernel_stage1"];
    id<MTLComputePipelineState> stage1Pipeline = [device newComputePipelineStateWithFunction:stage1Function error:&error];
    if (!stage1Pipeline) { NSLog(@"Failed to create stage 1 pipeline state: %@", error); return; }

    id<MTLFunction> stage2Function = [library newFunctionWithName:@"sum_kernel_stage2"];
    id<MTLComputePipelineState> stage2Pipeline = [device newComputePipelineStateWithFunction:stage2Function error:&error];
    if (!stage2Pipeline) { NSLog(@"Failed to create stage 2 pipeline state: %@", error); return; }

    const int numThreadgroups = 2048;
    id<MTLBuffer> inBuffer = [device newBufferWithBytes:cpuData.data() length:dataSizeBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> partialSumsBuffer = [device newBufferWithLength:numThreadgroups * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    id<MTLComputeCommandEncoder> stage1Encoder = [commandBuffer computeCommandEncoder];
    [stage1Encoder setComputePipelineState:stage1Pipeline];
    [stage1Encoder setBuffer:inBuffer offset:0 atIndex:0];

    [stage1Encoder setBuffer:partialSumsBuffer offset:0 atIndex:1];
    [stage1Encoder setBytes:&dataSize length:sizeof(dataSize) atIndex:2]; // <-- THIS IS THE NEW LINE
    
    NSUInteger stage1ThreadGroupSize = [stage1Pipeline maxTotalThreadsPerThreadgroup];
    MTLSize stage1GridSize = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize stage1GroupSize = MTLSizeMake(stage1ThreadGroupSize, 1, 1);
    [stage1Encoder dispatchThreadgroups:stage1GridSize threadsPerThreadgroup:stage1GroupSize];
    [stage1Encoder endEncoding];

    id<MTLComputeCommandEncoder> stage2Encoder = [commandBuffer computeCommandEncoder];
    [stage2Encoder setComputePipelineState:stage2Pipeline];
    [stage2Encoder setBuffer:partialSumsBuffer offset:0 atIndex:0];
    [stage2Encoder setBuffer:resultBuffer offset:0 atIndex:1];
    [stage2Encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [stage2Encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    double dataSizeGB = (double)dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;

    float *finalSum = (float *)resultBuffer.contents;
    NSLog(@"Final SUM(l_quantity): %.2f", finalSum[0]);
    NSLog(@"GPU execution time: %f ms", gpuExecutionTime * 1000.0);
    NSLog(@"Effective Bandwidth: %.2f GB/s\n", bandwidth);
}


// --- Main Function for Join Benchmark ---
void runJoinBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running Join Benchmark ---");
    
    // =================================================================
    // PHASE 1: BUILD
    // =================================================================
    
    // 1. Load Data for the build side (orders table)
    std::vector<int> buildKeys = loadIntColumn("Data/SF-10/orders.tbl", 0);
    if (buildKeys.empty()) {
        std::cerr << "Error: Could not open 'orders.tbl'. Make sure it's in your Data/SF-1 folder." << std::endl;
        return;
    }
    const uint buildDataSize = (uint)buildKeys.size();
    NSLog(@"Loaded %u rows from orders.tbl for build phase.", buildDataSize);

    // 2. Setup Hash Table
    const uint hashTableSize = buildDataSize * 2;
    const unsigned long hashTableSizeBytes = hashTableSize * sizeof(int) * 2;
    std::vector<int> cpuHashTable(hashTableSize * 2, -1);

    // 3. Setup Build Kernel and Pipeline State
    NSError *error = nil;
    id<MTLFunction> buildFunction = [library newFunctionWithName:@"hash_join_build"];
    id<MTLComputePipelineState> buildPipeline = [device newComputePipelineStateWithFunction:buildFunction error:&error];
    if (!buildPipeline) { NSLog(@"Failed to create build pipeline state: %@", error); return; }

    // 4. Create Build Buffers
    id<MTLBuffer> buildKeysBuffer = [device newBufferWithBytes:buildKeys.data() length:buildKeys.size() * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buildValuesBuffer = [device newBufferWithBytes:buildKeys.data() length:buildKeys.size() * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hashTableBuffer = [device newBufferWithBytes:cpuHashTable.data() length:hashTableSizeBytes options:MTLResourceStorageModeShared];

    // 5. Encode and Dispatch Build Kernel
    id<MTLCommandBuffer> buildCommandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> buildEncoder = [buildCommandBuffer computeCommandEncoder];
    
    [buildEncoder setComputePipelineState:buildPipeline];
    [buildEncoder setBuffer:buildKeysBuffer offset:0 atIndex:0];
    [buildEncoder setBuffer:buildValuesBuffer offset:0 atIndex:1];
    [buildEncoder setBuffer:hashTableBuffer offset:0 atIndex:2];
    [buildEncoder setBytes:&buildDataSize length:sizeof(buildDataSize) atIndex:3];
    [buildEncoder setBytes:&hashTableSize length:sizeof(hashTableSize) atIndex:4];

    MTLSize buildGridSize = MTLSizeMake(buildDataSize, 1, 1);
    NSUInteger buildThreadGroupSize = [buildPipeline maxTotalThreadsPerThreadgroup];
    if (buildThreadGroupSize > buildDataSize) { buildThreadGroupSize = buildDataSize; }
    MTLSize buildGroupSize = MTLSizeMake(buildThreadGroupSize, 1, 1);

    [buildEncoder dispatchThreads:buildGridSize threadsPerThreadgroup:buildGroupSize];
    [buildEncoder endEncoding];
    
    // 6. Execute Build Phase
    [buildCommandBuffer commit];

    // =================================================================
    // PHASE 2: PROBE
    // =================================================================
    
    // 7. Load Data for the probe side (lineitem table)
    // l_orderkey is the 1st column (index 0)
    std::vector<int> probeKeys = loadIntColumn("Data/SF-10/lineitem.tbl", 0);
    if (probeKeys.empty()) {
        std::cerr << "Error: Could not open 'lineitem.tbl' for probe phase." << std::endl;
        return;
    }
    const uint probeDataSize = (uint)probeKeys.size();
    NSLog(@"Loaded %u rows from lineitem.tbl for probe phase.", probeDataSize);

    // 8. Setup Probe Kernel and Pipeline State
    id<MTLFunction> probeFunction = [library newFunctionWithName:@"hash_join_probe"];
    id<MTLComputePipelineState> probePipeline = [device newComputePipelineStateWithFunction:probeFunction error:&error];
    if (!probePipeline) { NSLog(@"Failed to create probe pipeline state: %@", error); return; }

    // 9. Create Probe Buffers
    id<MTLBuffer> probeKeysBuffer = [device newBufferWithBytes:probeKeys.data() length:probeKeys.size() * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> matchCountBuffer = [device newBufferWithLength:sizeof(unsigned int) options:MTLResourceStorageModeShared];
    // Clear the match count to zero
    memset(matchCountBuffer.contents, 0, sizeof(unsigned int));

    // 10. Wait for build to finish, then start probe
    [buildCommandBuffer waitUntilCompleted]; // Ensure build is done before probe starts
    
    id<MTLCommandBuffer> probeCommandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> probeEncoder = [probeCommandBuffer computeCommandEncoder];

    [probeEncoder setComputePipelineState:probePipeline];
    [probeEncoder setBuffer:probeKeysBuffer offset:0 atIndex:0];
    [probeEncoder setBuffer:hashTableBuffer offset:0 atIndex:1]; // Reuse the hash table from build
    [probeEncoder setBuffer:matchCountBuffer offset:0 atIndex:2];
    [probeEncoder setBytes:&probeDataSize length:sizeof(probeDataSize) atIndex:3];
    [probeEncoder setBytes:&hashTableSize length:sizeof(hashTableSize) atIndex:4];

    MTLSize probeGridSize = MTLSizeMake(probeDataSize, 1, 1);
    NSUInteger probeThreadGroupSize = [probePipeline maxTotalThreadsPerThreadgroup];
    if (probeThreadGroupSize > probeDataSize) { probeThreadGroupSize = probeDataSize; }
    MTLSize probeGroupSize = MTLSizeMake(probeThreadGroupSize, 1, 1);
    [probeEncoder dispatchThreads:probeGridSize threadsPerThreadgroup:probeGroupSize];
    [probeEncoder endEncoding];
    
    // 11. Execute Probe Phase
    [probeCommandBuffer commit];
    [probeCommandBuffer waitUntilCompleted];

    // =================================================================
    // FINAL RESULTS
    // =================================================================
    
    double buildTime = buildCommandBuffer.GPUEndTime - buildCommandBuffer.GPUStartTime;
    double probeTime = probeCommandBuffer.GPUEndTime - probeCommandBuffer.GPUStartTime;
    
    unsigned int* matchCount = (unsigned int*)matchCountBuffer.contents;

    NSLog(@"Join complete. Found %u total matches.", *matchCount);
    NSLog(@"Build Phase GPU time: %f ms", buildTime * 1000.0);
    NSLog(@"Probe Phase GPU time: %f ms", probeTime * 1000.0);
    NSLog(@"Total Join GPU time: %f ms\n", (buildTime + probeTime) * 1000.0);
}


// C++ equivalent of the Metal struct for reading results.
// Note: no atomics here, as we are just reading the final values.
struct Q1Aggregates_CPU {
    int   key;
    float sum_qty;
    float sum_base_price;
    float sum_disc_price;
    float sum_charge;
    float sum_discount;
    unsigned int  count;
};

// --- Main Function for TPC-H Q1 Benchmark (High-Performance Version) ---
void runQ1Benchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running TPC-H Query 1 Benchmark (High-Performance) ---");

    const std::string filepath = "Data/SF-10/lineitem.tbl";
    auto l_returnflag = loadCharColumn(filepath, 8), l_linestatus = loadCharColumn(filepath, 9);
    auto l_quantity = loadFloatColumn(filepath, 4), l_extendedprice = loadFloatColumn(filepath, 5);
    auto l_discount = loadFloatColumn(filepath, 6), l_tax = loadFloatColumn(filepath, 7);
    auto l_shipdate = loadDateColumn(filepath, 10);
    const uint data_size = (uint)l_shipdate.size();

    NSError* error = nil;
    id<MTLFunction> selectionFunction = [library newFunctionWithName:@"selection_kernel"];
    id<MTLComputePipelineState> selectionPipeline = [device newComputePipelineStateWithFunction:selectionFunction error:&error];

    id<MTLFunction> localAggFunction = [library newFunctionWithName:@"q1_local_aggregation_kernel"];
    id<MTLComputePipelineState> localAggPipeline = [device newComputePipelineStateWithFunction:localAggFunction error:&error];

    id<MTLFunction> mergeFunction = [library newFunctionWithName:@"q1_merge_kernel"];
    id<MTLComputePipelineState> mergePipeline = [device newComputePipelineStateWithFunction:mergeFunction error:&error];

    id<MTLBuffer> shipdateBuffer = [device newBufferWithBytes:l_shipdate.data() length:data_size * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bitmapBuffer = [device newBufferWithLength:data_size * sizeof(unsigned int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> flagBuffer = [device newBufferWithBytes:l_returnflag.data() length:data_size * sizeof(char) options:MTLResourceStorageModeShared];
    id<MTLBuffer> statusBuffer = [device newBufferWithBytes:l_linestatus.data() length:data_size * sizeof(char) options:MTLResourceStorageModeShared];
    id<MTLBuffer> qtyBuffer = [device newBufferWithBytes:l_quantity.data() length:data_size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> priceBuffer = [device newBufferWithBytes:l_extendedprice.data() length:data_size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> discBuffer = [device newBufferWithBytes:l_discount.data() length:data_size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> taxBuffer = [device newBufferWithBytes:l_tax.data() length:data_size * sizeof(float) options:MTLResourceStorageModeShared];
    
    const uint num_threadgroups = 2048;
    const uint local_ht_size = 16;
    const uint intermediate_size = num_threadgroups * local_ht_size;
    id<MTLBuffer> intermediateBuffer = [device newBufferWithLength:intermediate_size * sizeof(Q1Aggregates_CPU) options:MTLResourceStorageModeShared];

    const uint final_ht_size = 64;
    std::vector<int> cpuFinalHashTable(final_ht_size * (sizeof(Q1Aggregates_CPU)/sizeof(int)), -1);
    id<MTLBuffer> finalHashTableBuffer = [device newBufferWithBytes:cpuFinalHashTable.data() length:final_ht_size * sizeof(Q1Aggregates_CPU) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Stage 1: Selection
    id<MTLComputeCommandEncoder> selectionEncoder = [commandBuffer computeCommandEncoder];
    int filterDate = 19980902; // Corresponds to DATE '1998-12-01' - INTERVAL '90' DAY
    [selectionEncoder setComputePipelineState:selectionPipeline];
    [selectionEncoder setBuffer:shipdateBuffer offset:0 atIndex:0];
    [selectionEncoder setBuffer:bitmapBuffer offset:0 atIndex:1];
    [selectionEncoder setBytes:&filterDate length:sizeof(filterDate) atIndex:2];
    [selectionEncoder dispatchThreads:MTLSizeMake(data_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [selectionEncoder endEncoding];
    
    // Stage 2: Local Aggregation
    id<MTLComputeCommandEncoder> localAggEncoder = [commandBuffer computeCommandEncoder];
    [localAggEncoder setComputePipelineState:localAggPipeline];
    [localAggEncoder setBuffer:bitmapBuffer offset:0 atIndex:0];
    [localAggEncoder setBuffer:flagBuffer offset:0 atIndex:1];
    [localAggEncoder setBuffer:statusBuffer offset:0 atIndex:2];
    [localAggEncoder setBuffer:qtyBuffer offset:0 atIndex:3];
    [localAggEncoder setBuffer:priceBuffer offset:0 atIndex:4];
    [localAggEncoder setBuffer:discBuffer offset:0 atIndex:5];
    [localAggEncoder setBuffer:taxBuffer offset:0 atIndex:6];
    [localAggEncoder setBuffer:intermediateBuffer offset:0 atIndex:7];
    [localAggEncoder setBytes:&data_size length:sizeof(data_size) atIndex:8];
    [localAggEncoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [localAggEncoder endEncoding];

    // Stage 3: Merge
    id<MTLComputeCommandEncoder> mergeEncoder = [commandBuffer computeCommandEncoder];
    [mergeEncoder setComputePipelineState:mergePipeline];
    [mergeEncoder setBuffer:intermediateBuffer offset:0 atIndex:0];
    [mergeEncoder setBuffer:finalHashTableBuffer offset:0 atIndex:1];
    [mergeEncoder setBytes:&intermediate_size length:sizeof(intermediate_size) atIndex:2];
    [mergeEncoder setBytes:&final_ht_size length:sizeof(final_ht_size) atIndex:3];
    [mergeEncoder dispatchThreads:MTLSizeMake(intermediate_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [mergeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;

    // Print Results (same logic as before)
    struct Q1Result { float sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc; uint count; };
    Q1Aggregates_CPU* results = (Q1Aggregates_CPU*)finalHashTableBuffer.contents;
    std::map<std::pair<char, char>, Q1Result> final_results;
    for (int i = 0; i < final_ht_size; ++i) {
        if (results[i].key != -1) {
            char flag = (results[i].key >> 8) & 0xFF, status = results[i].key & 0xFF;
            Q1Result res;
            res.sum_qty = results[i].sum_qty; res.sum_base_price = results[i].sum_base_price;
            res.sum_disc_price = results[i].sum_disc_price; res.sum_charge = results[i].sum_charge;
            res.count = results[i].count;
            res.avg_qty = res.sum_qty / res.count; res.avg_price = res.sum_base_price / res.count;
            res.avg_disc = results[i].sum_discount / res.count;
            final_results[{flag, status}] = res;
        }
    }
    printf("\n+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    for(auto const& [key, val] : final_results) {
        printf("| %8c | %8c | %10.2f | %14.2f | %14.2f | %14.2f | %10.2f | %10.2f | %10.2f | %8u |\n",
               key.first, key.second, val.sum_qty, val.sum_base_price, val.sum_disc_price, val.sum_charge,
               val.avg_qty, val.avg_price, val.avg_disc, val.count);
    }
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    NSLog(@"Total TPC-H Q1 GPU time (High-Performance): %f ms", gpuExecutionTime * 1000.0);
}


// --- Main Entry Point ---
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLLibrary> library;
        @try {
            library = [device newDefaultLibrary];
        } @catch (NSException *exception) {
            NSLog(@"Error loading .metal library: %@", exception.reason);
            return 1;
        }

        // Run all four benchmarks
        // runSelectionBenchmark(device, commandQueue, library);
        // runAggregationBenchmark(device, commandQueue, library);
        // runJoinBenchmark(device, commandQueue, library);
        runQ1Benchmark(device, commandQueue, library);
    }
    return 0;
}
