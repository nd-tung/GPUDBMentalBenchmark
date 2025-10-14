#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

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
void runSingleSelectionTest(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::ComputePipelineState* pipelineState,
                            MTL::Buffer* inBuffer, MTL::Buffer* resultBuffer,
                            const std::vector<int>& cpuData, int filterValue) {
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
    commandEncoder->setComputePipelineState(pipelineState);
    commandEncoder->setBuffer(inBuffer, 0, 0);
    commandEncoder->setBuffer(resultBuffer, 0, 1);
    commandEncoder->setBytes(&filterValue, sizeof(filterValue), 2);
    
    MTL::Size gridSize = MTL::Size::Make(cpuData.size(), 1, 1);
    NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > cpuData.size()) { threadGroupSize = cpuData.size(); }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
    commandEncoder->dispatchThreads(gridSize, threadgroupSize);
    commandEncoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    double gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer->contents();
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) { if (resultData[i] == 1) { passCount++; } }
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    std::cout << "--- Filter Value: < " << filterValue << " ---" << std::endl;
    std::cout << "Selectivity: " << selectivity << "% (" << passCount << " rows matched)" << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
}

// --- Main Function for Selection Benchmark ---
void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Selection Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<int> cpuData = loadIntColumn("Data/SF-10/lineitem.tbl", 1);
    if (cpuData.empty()) { return; }
    std::cout << "Loaded " << cpuData.size() << " rows for selection." << std::endl;

    NS::Error* error = nullptr;
    NS::String* functionName = NS::String::string("selection_kernel", NS::UTF8StringEncoding);
    MTL::Function* selectionFunction = library->newFunction(functionName);
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(selectionFunction, &error);
    if (!pipelineState) { 
        std::cerr << "Failed to create selection pipeline state" << std::endl; 
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(cpuData.size() * sizeof(unsigned int), MTL::ResourceStorageModeShared);

    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
    
    // Cleanup
    selectionFunction->release();
    pipelineState->release();
    inBuffer->release();
    resultBuffer->release();
    functionName->release();
}


// --- Main Function for Aggregation Benchmark ---
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Aggregation Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<float> cpuData = loadFloatColumn("Data/SF-10/lineitem.tbl", 4);
    if (cpuData.empty()) return;
    std::cout << "Loaded " << cpuData.size() << " rows for aggregation." << std::endl;
    const unsigned long dataSizeBytes = cpuData.size() * sizeof(float);
    uint dataSize = (uint)cpuData.size(); // The actual number of elements

    NS::Error* error = nullptr;
    NS::String* stage1FunctionName = NS::String::string("sum_kernel_stage1", NS::UTF8StringEncoding);
    MTL::Function* stage1Function = library->newFunction(stage1FunctionName);
    MTL::ComputePipelineState* stage1Pipeline = device->newComputePipelineState(stage1Function, &error);
    if (!stage1Pipeline) { 
        std::cerr << "Failed to create stage 1 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    NS::String* stage2FunctionName = NS::String::string("sum_kernel_stage2", NS::UTF8StringEncoding);
    MTL::Function* stage2Function = library->newFunction(stage2FunctionName);
    MTL::ComputePipelineState* stage2Pipeline = device->newComputePipelineState(stage2Function, &error);
    if (!stage2Pipeline) { 
        std::cerr << "Failed to create stage 2 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    const int numThreadgroups = 2048;
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* partialSumsBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    MTL::ComputeCommandEncoder* stage1Encoder = commandBuffer->computeCommandEncoder();
    stage1Encoder->setComputePipelineState(stage1Pipeline);
    stage1Encoder->setBuffer(inBuffer, 0, 0);
    stage1Encoder->setBuffer(partialSumsBuffer, 0, 1);
    stage1Encoder->setBytes(&dataSize, sizeof(dataSize), 2);
    
    NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
    MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
    MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
    stage1Encoder->dispatchThreadgroups(stage1GridSize, stage1GroupSize);
    stage1Encoder->endEncoding();

    MTL::ComputeCommandEncoder* stage2Encoder = commandBuffer->computeCommandEncoder();
    stage2Encoder->setComputePipelineState(stage2Pipeline);
    stage2Encoder->setBuffer(partialSumsBuffer, 0, 0);
    stage2Encoder->setBuffer(resultBuffer, 0, 1);
    stage2Encoder->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
    stage2Encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    double gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
    double dataSizeGB = (double)dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;

    float *finalSum = (float *)resultBuffer->contents();
    std::cout << "Final SUM(l_quantity): " << finalSum[0] << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
    
    // Cleanup
    stage1Function->release();
    stage1Pipeline->release();
    stage2Function->release();
    stage2Pipeline->release();
    inBuffer->release();
    partialSumsBuffer->release();
    resultBuffer->release();
    stage1FunctionName->release();
    stage2FunctionName->release();
}


// --- Main Function for Join Benchmark ---
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Join Benchmark ---" << std::endl;
    
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
    std::cout << "Loaded " << buildDataSize << " rows from orders.tbl for build phase." << std::endl;

    // 2. Setup Hash Table
    const uint hashTableSize = buildDataSize * 2;
    const unsigned long hashTableSizeBytes = hashTableSize * sizeof(int) * 2;
    std::vector<int> cpuHashTable(hashTableSize * 2, -1);

    // 3. Setup Build Kernel and Pipeline State
    NS::Error* error = nullptr;
    NS::String* buildFunctionName = NS::String::string("hash_join_build", NS::UTF8StringEncoding);
    MTL::Function* buildFunction = library->newFunction(buildFunctionName);
    MTL::ComputePipelineState* buildPipeline = device->newComputePipelineState(buildFunction, &error);
    if (!buildPipeline) { 
        std::cerr << "Failed to create build pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    // 4. Create Build Buffers
    MTL::Buffer* buildKeysBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* buildValuesBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* hashTableBuffer = device->newBuffer(cpuHashTable.data(), hashTableSizeBytes, MTL::ResourceStorageModeShared);

    // 5. Encode and Dispatch Build Kernel
    MTL::CommandBuffer* buildCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* buildEncoder = buildCommandBuffer->computeCommandEncoder();
    
    buildEncoder->setComputePipelineState(buildPipeline);
    buildEncoder->setBuffer(buildKeysBuffer, 0, 0);
    buildEncoder->setBuffer(buildValuesBuffer, 0, 1);
    buildEncoder->setBuffer(hashTableBuffer, 0, 2);
    buildEncoder->setBytes(&buildDataSize, sizeof(buildDataSize), 3);
    buildEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size buildGridSize = MTL::Size::Make(buildDataSize, 1, 1);
    NS::UInteger buildThreadGroupSize = buildPipeline->maxTotalThreadsPerThreadgroup();
    if (buildThreadGroupSize > buildDataSize) { buildThreadGroupSize = buildDataSize; }
    MTL::Size buildGroupSize = MTL::Size::Make(buildThreadGroupSize, 1, 1);

    buildEncoder->dispatchThreads(buildGridSize, buildGroupSize);
    buildEncoder->endEncoding();
    
    // 6. Execute Build Phase
    buildCommandBuffer->commit();

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
    std::cout << "Loaded " << probeDataSize << " rows from lineitem.tbl for probe phase." << std::endl;

    // 8. Setup Probe Kernel and Pipeline State
    NS::String* probeFunctionName = NS::String::string("hash_join_probe", NS::UTF8StringEncoding);
    MTL::Function* probeFunction = library->newFunction(probeFunctionName);
    MTL::ComputePipelineState* probePipeline = device->newComputePipelineState(probeFunction, &error);
    if (!probePipeline) { 
        std::cerr << "Failed to create probe pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    // 9. Create Probe Buffers
    MTL::Buffer* probeKeysBuffer = device->newBuffer(probeKeys.data(), probeKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchCountBuffer = device->newBuffer(sizeof(unsigned int), MTL::ResourceStorageModeShared);
    // Clear the match count to zero
    memset(matchCountBuffer->contents(), 0, sizeof(unsigned int));

    // 10. Wait for build to finish, then start probe
    buildCommandBuffer->waitUntilCompleted(); // Ensure build is done before probe starts
    
    MTL::CommandBuffer* probeCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* probeEncoder = probeCommandBuffer->computeCommandEncoder();

    probeEncoder->setComputePipelineState(probePipeline);
    probeEncoder->setBuffer(probeKeysBuffer, 0, 0);
    probeEncoder->setBuffer(hashTableBuffer, 0, 1); // Reuse the hash table from build
    probeEncoder->setBuffer(matchCountBuffer, 0, 2);
    probeEncoder->setBytes(&probeDataSize, sizeof(probeDataSize), 3);
    probeEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size probeGridSize = MTL::Size::Make(probeDataSize, 1, 1);
    NS::UInteger probeThreadGroupSize = probePipeline->maxTotalThreadsPerThreadgroup();
    if (probeThreadGroupSize > probeDataSize) { probeThreadGroupSize = probeDataSize; }
    MTL::Size probeGroupSize = MTL::Size::Make(probeThreadGroupSize, 1, 1);
    probeEncoder->dispatchThreads(probeGridSize, probeGroupSize);
    probeEncoder->endEncoding();
    
    // 11. Execute Probe Phase
    probeCommandBuffer->commit();
    probeCommandBuffer->waitUntilCompleted();

    // =================================================================
    // FINAL RESULTS
    // =================================================================
    
    double buildTime = buildCommandBuffer->GPUEndTime() - buildCommandBuffer->GPUStartTime();
    double probeTime = probeCommandBuffer->GPUEndTime() - probeCommandBuffer->GPUStartTime();
    
    unsigned int* matchCount = (unsigned int*)matchCountBuffer->contents();

    std::cout << "Join complete. Found " << *matchCount << " total matches." << std::endl;
    std::cout << "Build Phase GPU time: " << buildTime * 1000.0 << " ms" << std::endl;
    std::cout << "Probe Phase GPU time: " << probeTime * 1000.0 << " ms" << std::endl;
    std::cout << "Total Join GPU time: " << (buildTime + probeTime) * 1000.0 << " ms" << std::endl << std::endl;
    
    // Cleanup
    buildFunction->release();
    buildPipeline->release();
    probeFunction->release();
    probePipeline->release();
    buildKeysBuffer->release();
    buildValuesBuffer->release();
    hashTableBuffer->release();
    probeKeysBuffer->release();
    matchCountBuffer->release();
    buildFunctionName->release();
    probeFunctionName->release();
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

// --- Main Function for TPC-H Q1 Benchmark ---
void runQ1Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 1 Benchmark ---" << std::endl;

    const std::string filepath = "Data/SF-10/lineitem.tbl";
    auto l_returnflag = loadCharColumn(filepath, 8), l_linestatus = loadCharColumn(filepath, 9);
    auto l_quantity = loadFloatColumn(filepath, 4), l_extendedprice = loadFloatColumn(filepath, 5);
    auto l_discount = loadFloatColumn(filepath, 6), l_tax = loadFloatColumn(filepath, 7);
    auto l_shipdate = loadDateColumn(filepath, 10);
    const uint data_size = (uint)l_shipdate.size();

    NS::Error* error = nullptr;
    NS::String* selectionFunctionName = NS::String::string("selection_kernel", NS::UTF8StringEncoding);
    MTL::Function* selectionFunction = library->newFunction(selectionFunctionName);
    MTL::ComputePipelineState* selectionPipeline = device->newComputePipelineState(selectionFunction, &error);

    NS::String* localAggFunctionName = NS::String::string("q1_local_aggregation_kernel", NS::UTF8StringEncoding);
    MTL::Function* localAggFunction = library->newFunction(localAggFunctionName);
    MTL::ComputePipelineState* localAggPipeline = device->newComputePipelineState(localAggFunction, &error);

    NS::String* mergeFunctionName = NS::String::string("q1_merge_kernel", NS::UTF8StringEncoding);
    MTL::Function* mergeFunction = library->newFunction(mergeFunctionName);
    MTL::ComputePipelineState* mergePipeline = device->newComputePipelineState(mergeFunction, &error);

    MTL::Buffer* shipdateBuffer = device->newBuffer(l_shipdate.data(), data_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* bitmapBuffer = device->newBuffer(data_size * sizeof(unsigned int), MTL::ResourceStorageModeShared);
    MTL::Buffer* flagBuffer = device->newBuffer(l_returnflag.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* statusBuffer = device->newBuffer(l_linestatus.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* qtyBuffer = device->newBuffer(l_quantity.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* priceBuffer = device->newBuffer(l_extendedprice.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* discBuffer = device->newBuffer(l_discount.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* taxBuffer = device->newBuffer(l_tax.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    
    const uint num_threadgroups = 2048;
    const uint local_ht_size = 16;
    const uint intermediate_size = num_threadgroups * local_ht_size;
    MTL::Buffer* intermediateBuffer = device->newBuffer(intermediate_size * sizeof(Q1Aggregates_CPU), MTL::ResourceStorageModeShared);

    const uint final_ht_size = 64;
    std::vector<int> cpuFinalHashTable(final_ht_size * (sizeof(Q1Aggregates_CPU)/sizeof(int)), -1);
    MTL::Buffer* finalHashTableBuffer = device->newBuffer(cpuFinalHashTable.data(), final_ht_size * sizeof(Q1Aggregates_CPU), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Stage 1: Selection
    MTL::ComputeCommandEncoder* selectionEncoder = commandBuffer->computeCommandEncoder();
    int filterDate = 19980902; // Corresponds to DATE '1998-12-01' - INTERVAL '90' DAY
    selectionEncoder->setComputePipelineState(selectionPipeline);
    selectionEncoder->setBuffer(shipdateBuffer, 0, 0);
    selectionEncoder->setBuffer(bitmapBuffer, 0, 1);
    selectionEncoder->setBytes(&filterDate, sizeof(filterDate), 2);
    selectionEncoder->dispatchThreads(MTL::Size::Make(data_size, 1, 1), MTL::Size::Make(1024, 1, 1));
    selectionEncoder->endEncoding();
    
    // Stage 2: Local Aggregation
    MTL::ComputeCommandEncoder* localAggEncoder = commandBuffer->computeCommandEncoder();
    localAggEncoder->setComputePipelineState(localAggPipeline);
    localAggEncoder->setBuffer(bitmapBuffer, 0, 0);
    localAggEncoder->setBuffer(flagBuffer, 0, 1);
    localAggEncoder->setBuffer(statusBuffer, 0, 2);
    localAggEncoder->setBuffer(qtyBuffer, 0, 3);
    localAggEncoder->setBuffer(priceBuffer, 0, 4);
    localAggEncoder->setBuffer(discBuffer, 0, 5);
    localAggEncoder->setBuffer(taxBuffer, 0, 6);
    localAggEncoder->setBuffer(intermediateBuffer, 0, 7);
    localAggEncoder->setBytes(&data_size, sizeof(data_size), 8);
    localAggEncoder->dispatchThreadgroups(MTL::Size::Make(num_threadgroups, 1, 1), MTL::Size::Make(1024, 1, 1));
    localAggEncoder->endEncoding();

    // Stage 3: Merge
    MTL::ComputeCommandEncoder* mergeEncoder = commandBuffer->computeCommandEncoder();
    mergeEncoder->setComputePipelineState(mergePipeline);
    mergeEncoder->setBuffer(intermediateBuffer, 0, 0);
    mergeEncoder->setBuffer(finalHashTableBuffer, 0, 1);
    mergeEncoder->setBytes(&intermediate_size, sizeof(intermediate_size), 2);
    mergeEncoder->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
    mergeEncoder->dispatchThreads(MTL::Size::Make(intermediate_size, 1, 1), MTL::Size::Make(1024, 1, 1));
    mergeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    double gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();

    // Print Results (same logic as before)
    struct Q1Result { float sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc; uint count; };
    Q1Aggregates_CPU* results = (Q1Aggregates_CPU*)finalHashTableBuffer->contents();
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
    std::cout << "Total TPC-H Q1 GPU time (High-Performance): " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    
    // Cleanup
    selectionFunction->release();
    selectionPipeline->release();
    localAggFunction->release();
    localAggPipeline->release();
    mergeFunction->release();
    mergePipeline->release();
    shipdateBuffer->release();
    bitmapBuffer->release();
    flagBuffer->release();
    statusBuffer->release();
    qtyBuffer->release();
    priceBuffer->release();
    discBuffer->release();
    taxBuffer->release();
    intermediateBuffer->release();
    finalHashTableBuffer->release();
    selectionFunctionName->release();
    localAggFunctionName->release();
    mergeFunctionName->release();
}


// --- Main Entry Point ---
int main(int argc, const char * argv[]) {
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    
    NS::Error* error = nullptr;
    MTL::Library* library = device->newDefaultLibrary();
    if (!library) {
        // Try to load from specific path
        NS::String* libraryPath = NS::String::string("default.metallib", NS::UTF8StringEncoding);
        library = device->newLibrary(libraryPath, &error);
        libraryPath->release();
        
        if (!library) {
            std::cerr << "Error loading .metal library from both default and file path" << std::endl;
            if (error) {
                std::cerr << "Error details: " << error->localizedDescription()->utf8String() << std::endl;
            }
            return 1;
        }
    }

    // Run all four benchmarks
    runSelectionBenchmark(device, commandQueue, library);
    runAggregationBenchmark(device, commandQueue, library);
    runJoinBenchmark(device, commandQueue, library);
    runQ1Benchmark(device, commandQueue, library);
    
    // Cleanup
    library->release();
    commandQueue->release();
    device->release();
    
    pAutoreleasePool->release();
    return 0;
}
