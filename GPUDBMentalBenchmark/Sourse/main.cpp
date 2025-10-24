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
#include <chrono>
#include <iomanip>
#include <cmath>

// Global dataset configuration
std::string g_dataset_path = "Data/SF-1/"; // Default to SF-10

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
std::vector<char> loadCharColumn(const std::string& filePath, int columnIndex, int fixed_width = 0) {
    std::vector<char> data; std::ifstream file(filePath); if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl; return data;
    }
    std::string line; while (std::getline(file, line)) { std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) { if (currentCol == columnIndex) { token = line.substr(start, end - start);
            if (fixed_width > 0) { for(int i=0; i < fixed_width; ++i) data.push_back(i < token.length() ? token[i] : '\0'); }
            else { data.push_back(token[0]);
            }
            break;
        }
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
    std::vector<int> cpuData = loadIntColumn(g_dataset_path + "lineitem.tbl", 1);
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
    std::vector<float> cpuData = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);
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
    std::vector<int> buildKeys = loadIntColumn(g_dataset_path + "orders.tbl", 0);
    if (buildKeys.empty()) {
        std::cerr << "Error: Could not open 'orders.tbl'. Make sure it's in your " << g_dataset_path << " folder." << std::endl;
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
    std::vector<int> probeKeys = loadIntColumn(g_dataset_path + "lineitem.tbl", 0);
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

    const std::string filepath = g_dataset_path + "lineitem.tbl";
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
    std::cout << "Total TPC-H Q1 GPU time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    
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



// C++ structs for reading final results
struct Q3Result {
    int orderkey;
    float revenue;
    int orderdate;
    int shippriority;
};

struct Q3Aggregates_CPU {
    int key;
    float revenue;
    unsigned int orderdate;
    unsigned int shippriority;
};


// --- Main Function for TPC-H Q3 Benchmark ---
void runQ3Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 3 Benchmark ---" << std::endl;

    // 1. Load data for all three tables
    const std::string sf_path = g_dataset_path;
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto c_mktsegment = loadCharColumn(sf_path + "customer.tbl", 6);

    auto o_orderkey = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_orderdate = loadDateColumn(sf_path + "orders.tbl", 4);
    auto o_shippriority = loadIntColumn(sf_path + "orders.tbl", 7);

    auto l_orderkey = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_shipdate = loadDateColumn(sf_path + "lineitem.tbl", 10);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    
    const uint customer_size = (uint)c_custkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();
    std::cout << "Loaded " << customer_size << " customers, " << orders_size << " orders, " << lineitem_size << " lineitem rows." << std::endl;

    // 2. Setup all kernels
    NS::Error* pError = nullptr;
    MTL::Function* pCustBuildFn = pLibrary->newFunction(NS::String::string("q3_build_customer_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pCustBuildPipe = pDevice->newComputePipelineState(pCustBuildFn, &pError);

    MTL::Function* pOrdersBuildFn = pLibrary->newFunction(NS::String::string("q3_build_orders_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = pDevice->newComputePipelineState(pOrdersBuildFn, &pError);
    
    MTL::Function* pProbeAggFn = pLibrary->newFunction(NS::String::string("q3_probe_and_local_agg_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pProbeAggPipe = pDevice->newComputePipelineState(pProbeAggFn, &pError);

    MTL::Function* pMergeFn = pLibrary->newFunction(NS::String::string("q3_merge_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pMergePipe = pDevice->newComputePipelineState(pMergeFn, &pError);

    // 3. Create Buffers
    const uint customer_ht_size = customer_size * 2;
    std::vector<int> cpu_customer_ht(customer_ht_size * 2, -1);
    MTL::Buffer* pCustKeyBuffer = pDevice->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustMktBuffer = pDevice->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustomerHTBuffer = pDevice->newBuffer(cpu_customer_ht.data(), customer_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);

    const uint orders_ht_size = orders_size * 2;
    std::vector<int> cpu_orders_ht(orders_ht_size * 2, -1);
    MTL::Buffer* pOrdKeyBuffer = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuffer = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPrioBuffer = pDevice->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuffer = pDevice->newBuffer(cpu_orders_ht.data(), orders_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);
    
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineShipDateBuffer = pDevice->newBuffer(l_shipdate.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    
    const uint num_threadgroups = 2048;
    // Allocate intermediate as an append-only buffer up to lineitem_size
    const uint intermediate_capacity = lineitem_size;
    MTL::Buffer* pIntermediateBuffer = pDevice->newBuffer(intermediate_capacity * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOutCountBuffer = pDevice->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    // Initialize out counter to 0
    memset(pOutCountBuffer->contents(), 0, sizeof(uint));

    const uint final_ht_size = orders_size;
    std::vector<int> cpu_final_ht(final_ht_size * (sizeof(Q3Aggregates_CPU)/sizeof(int)), -1);
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(cpu_final_ht.data(), final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);

    // 4. Dispatch full pipeline
    MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
    const int cutoff_date = 19950315;

    MTL::ComputeCommandEncoder* pCustBuildEncoder = pCommandBuffer->computeCommandEncoder();
    pCustBuildEncoder->setComputePipelineState(pCustBuildPipe);
    pCustBuildEncoder->setBuffer(pCustKeyBuffer, 0, 0);
    pCustBuildEncoder->setBuffer(pCustMktBuffer, 0, 1);
    pCustBuildEncoder->setBuffer(pCustomerHTBuffer, 0, 2);
    pCustBuildEncoder->setBytes(&customer_size, sizeof(customer_size), 3);
    pCustBuildEncoder->setBytes(&customer_ht_size, sizeof(customer_ht_size), 4);
    pCustBuildEncoder->dispatchThreads(MTL::Size(customer_size, 1, 1), MTL::Size(1024, 1, 1));
    pCustBuildEncoder->endEncoding();

    MTL::ComputeCommandEncoder* pOrdersBuildEncoder = pCommandBuffer->computeCommandEncoder();
    pOrdersBuildEncoder->setComputePipelineState(pOrdersBuildPipe);
    pOrdersBuildEncoder->setBuffer(pOrdKeyBuffer, 0, 0);
    pOrdersBuildEncoder->setBuffer(pOrdDateBuffer, 0, 1);
    pOrdersBuildEncoder->setBuffer(pOrdersHTBuffer, 0, 2);
    pOrdersBuildEncoder->setBytes(&orders_size, sizeof(orders_size), 3);
    pOrdersBuildEncoder->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
    pOrdersBuildEncoder->setBytes(&cutoff_date, sizeof(cutoff_date), 5);
    pOrdersBuildEncoder->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(1024, 1, 1));
    pOrdersBuildEncoder->endEncoding();

    MTL::ComputeCommandEncoder* pProbeAggEncoder = pCommandBuffer->computeCommandEncoder();
    pProbeAggEncoder->setComputePipelineState(pProbeAggPipe);
    pProbeAggEncoder->setBuffer(pLineOrdKeyBuffer, 0, 0);
    pProbeAggEncoder->setBuffer(pLineShipDateBuffer, 0, 1);
    pProbeAggEncoder->setBuffer(pLinePriceBuffer, 0, 2);
    pProbeAggEncoder->setBuffer(pLineDiscBuffer, 0, 3);
    pProbeAggEncoder->setBuffer(pCustomerHTBuffer, 0, 4);
    pProbeAggEncoder->setBuffer(pOrdersHTBuffer, 0, 5);
    pProbeAggEncoder->setBuffer(pOrdCustKeyBuffer, 0, 6);
    pProbeAggEncoder->setBuffer(pOrdDateBuffer, 0, 7);
    pProbeAggEncoder->setBuffer(pOrdPrioBuffer, 0, 8);
    pProbeAggEncoder->setBuffer(pIntermediateBuffer, 0, 9);
    pProbeAggEncoder->setBuffer(pOutCountBuffer, 0, 10);
    pProbeAggEncoder->setBytes(&lineitem_size, sizeof(lineitem_size), 11);
    pProbeAggEncoder->setBytes(&customer_ht_size, sizeof(customer_ht_size), 12);
    pProbeAggEncoder->setBytes(&orders_ht_size, sizeof(orders_ht_size), 13);
    pProbeAggEncoder->setBytes(&cutoff_date, sizeof(cutoff_date), 14);
    pProbeAggEncoder->setBytes(&intermediate_capacity, sizeof(intermediate_capacity), 15);
    pProbeAggEncoder->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    pProbeAggEncoder->endEncoding();
    
    // Ensure final hash table is initialized to empty (-1 keys) on CPU (shared memory)
    {   
        void* final_ptr = pFinalHTBuffer->contents();
        memset(final_ptr, 0xFF, final_ht_size * sizeof(Q3Aggregates_CPU));
    }

    // NOTE: Skip GPU merge stage for Q3 due to non-determinism; perform final merge on CPU for correctness

    // 5. Execute and time (GPU portion up to local agg)
    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();
    double gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();

    // Debug: count non-empty intermediate results
    {
        uint out_count = *(uint*)pOutCountBuffer->contents();
        Q3Aggregates_CPU* inter_dbg = (Q3Aggregates_CPU*)pIntermediateBuffer->contents();
        size_t non_empty = 0;
        int printed = 0;
        for (uint i = 0; i < out_count && i < 100000; ++i) { // cap scan for speed
            if (inter_dbg[i].key > 0) {
                non_empty++;
                if (printed < 5) {
                    std::cout << "[DEBUG] inter[" << i << "]: key=" << inter_dbg[i].key
                              << ", revenue=" << inter_dbg[i].revenue
                              << ", orderdate=" << inter_dbg[i].orderdate
                              << ", shippriority=" << inter_dbg[i].shippriority << std::endl;
                    printed++;
                }
            }
        }
        std::cout << "[DEBUG] Intermediate non-empty entries: " << non_empty << "/" << out_count << std::endl;
    }

    // 6. CPU merge for determinism and correctness
    auto cpuMergeStart = std::chrono::high_resolution_clock::now();
    std::unordered_map<int, Q3Result> acc;
    uint out_count = *(uint*)pOutCountBuffer->contents();
    Q3Aggregates_CPU* inter = (Q3Aggregates_CPU*)pIntermediateBuffer->contents();
    for (uint i = 0; i < out_count; ++i) {
        if (inter[i].key > 0) {
            auto it = acc.find(inter[i].key);
            if (it == acc.end()) {
                acc.emplace(inter[i].key, Q3Result{inter[i].key, inter[i].revenue, (int)inter[i].orderdate, (int)inter[i].shippriority});
            } else {
                it->second.revenue += inter[i].revenue;
            }
        }
    }
    std::vector<Q3Result> final_results;
    final_results.reserve(acc.size());
    for (auto &kv : acc) final_results.push_back(kv.second);
    std::sort(final_results.begin(), final_results.end(), [](const Q3Result& a, const Q3Result& b) {
        if (a.revenue != b.revenue) return a.revenue > b.revenue;
        return a.orderdate < b.orderdate;
    });
    auto cpuMergeEnd = std::chrono::high_resolution_clock::now();
    double cpuMergeMs = std::chrono::duration<double, std::milli>(cpuMergeEnd - cpuMergeStart).count();

    printf("\nTPC-H Query 3 Results (Top 10):\n");
    printf("+----------+------------+------------+--------------+\n");
    printf("| orderkey |   revenue  | orderdate  | shippriority |\n");
    printf("+----------+------------+------------+--------------+\n");
    for (int i = 0; i < 10 && i < final_results.size(); ++i) {
        printf("| %8d | $%10.2f | %10d | %12d |\n",
               final_results[i].orderkey, final_results[i].revenue, final_results[i].orderdate, final_results[i].shippriority);
    }
    printf("+----------+------------+------------+--------------+\n");
    printf("Total results found: %lu\n", final_results.size());
    printf("Q3 Mode: Hybrid (GPU probe + CPU merge)\n");
    printf("  GPU time (build+probe): %0.3f ms\n", gpuExecutionTime * 1000.0);
    printf("  CPU merge time: %0.3f ms\n", cpuMergeMs);
    printf("  Total hybrid time: %0.3f ms\n", gpuExecutionTime * 1000.0 + cpuMergeMs);
    
    //Cleanup
    pCustBuildFn->release();
    pCustBuildPipe->release();
    pOrdersBuildFn->release();
    pOrdersBuildPipe->release();
    pProbeAggFn->release();
    pProbeAggPipe->release();
    pMergeFn->release();
    pMergePipe->release();

    pCustKeyBuffer->release();
    pCustMktBuffer->release();
    pCustomerHTBuffer->release();
    pOrdKeyBuffer->release();
    pOrdCustKeyBuffer->release();
    pOrdDateBuffer->release();
    pOrdPrioBuffer->release();
    pOrdersHTBuffer->release();
    pLineOrdKeyBuffer->release();
    pLineShipDateBuffer->release();
    pLinePriceBuffer->release();
    pLineDiscBuffer->release();
    pIntermediateBuffer->release();
    pOutCountBuffer->release();
    pFinalHTBuffer->release();
}


// --- Main Function for TPC-H Query 6 Benchmark ---
void runQ6Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 6 Benchmark ---" << std::endl;
    
    // Load required columns from lineitem table
    std::vector<int> l_shipdate = loadDateColumn(g_dataset_path + "lineitem.tbl", 10);    // Column 10: l_shipdate
    std::vector<float> l_discount = loadFloatColumn(g_dataset_path + "lineitem.tbl", 6);  // Column 6: l_discount
    std::vector<float> l_quantity = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);  // Column 4: l_quantity
    std::vector<float> l_extendedprice = loadFloatColumn(g_dataset_path + "lineitem.tbl", 5); // Column 5: l_extendedprice

    if (l_shipdate.empty() || l_discount.empty() || l_quantity.empty() || l_extendedprice.empty()) {
        std::cerr << "Error: Could not load required columns for Q6 benchmark" << std::endl;
        return;
    }

    uint dataSize = (uint)l_shipdate.size();
    std::cout << "Loaded " << dataSize << " rows for TPC-H Query 6." << std::endl;

    // Query parameters
    int start_date = 19940101;   // 1994-01-01
    int end_date = 19950101;     // 1995-01-01
    float min_discount = 0.05f;  // 5%
    float max_discount = 0.07f;  // 7%
    float max_quantity = 24.0f;

    NS::Error* error = nullptr;
    
    // Create stage 1 pipeline (filter and sum)
    NS::String* stage1FunctionName = NS::String::string("q6_filter_and_sum_stage1", NS::UTF8StringEncoding);
    MTL::Function* stage1Function = library->newFunction(stage1FunctionName);
    if (!stage1Function) {
        std::cerr << "Error: Could not find q6_filter_and_sum_stage1 function" << std::endl;
        return;
    }
    MTL::ComputePipelineState* stage1Pipeline = device->newComputePipelineState(stage1Function, &error);
    if (!stage1Pipeline) {
        std::cerr << "Failed to create Q6 stage 1 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return;
    }

    // Create stage 2 pipeline (final sum)
    NS::String* stage2FunctionName = NS::String::string("q6_final_sum_stage2", NS::UTF8StringEncoding);
    MTL::Function* stage2Function = library->newFunction(stage2FunctionName);
    if (!stage2Function) {
        std::cerr << "Error: Could not find q6_final_sum_stage2 function" << std::endl;
        return;
    }
    MTL::ComputePipelineState* stage2Pipeline = device->newComputePipelineState(stage2Function, &error);
    if (!stage2Pipeline) {
        std::cerr << "Failed to create Q6 stage 2 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return;
    }

    // Create GPU buffers
    const int numThreadgroups = 2048;
    MTL::Buffer* shipdateBuffer = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* discountBuffer = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* quantityBuffer = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* extendedpriceBuffer = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialRevenuesBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalRevenueBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Execute GPU kernels
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    // Stage 1: Filter and compute partial revenue sums
    MTL::ComputeCommandEncoder* stage1Encoder = commandBuffer->computeCommandEncoder();
    stage1Encoder->setComputePipelineState(stage1Pipeline);
    stage1Encoder->setBuffer(shipdateBuffer, 0, 0);
    stage1Encoder->setBuffer(discountBuffer, 0, 1);
    stage1Encoder->setBuffer(quantityBuffer, 0, 2);
    stage1Encoder->setBuffer(extendedpriceBuffer, 0, 3);
    stage1Encoder->setBuffer(partialRevenuesBuffer, 0, 4);
    stage1Encoder->setBytes(&dataSize, sizeof(dataSize), 5);
    stage1Encoder->setBytes(&start_date, sizeof(start_date), 6);
    stage1Encoder->setBytes(&end_date, sizeof(end_date), 7);
    stage1Encoder->setBytes(&min_discount, sizeof(min_discount), 8);
    stage1Encoder->setBytes(&max_discount, sizeof(max_discount), 9);
    stage1Encoder->setBytes(&max_quantity, sizeof(max_quantity), 10);

    NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
    MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
    MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
    stage1Encoder->dispatchThreadgroups(stage1GridSize, stage1GroupSize);
    stage1Encoder->endEncoding();

    // Stage 2: Final sum reduction
    MTL::ComputeCommandEncoder* stage2Encoder = commandBuffer->computeCommandEncoder();
    stage2Encoder->setComputePipelineState(stage2Pipeline);
    stage2Encoder->setBuffer(partialRevenuesBuffer, 0, 0);
    stage2Encoder->setBuffer(finalRevenueBuffer, 0, 1);
    
    MTL::Size stage2GridSize = MTL::Size::Make(1, 1, 1);
    MTL::Size stage2GroupSize = MTL::Size::Make(1, 1, 1);
    stage2Encoder->dispatchThreads(stage2GridSize, stage2GroupSize);
    stage2Encoder->endEncoding();

    // Execute and measure time
    auto startTime = std::chrono::high_resolution_clock::now();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    auto endTime = std::chrono::high_resolution_clock::now();
    
    double gpuExecutionTime = std::chrono::duration<double>(endTime - startTime).count();

    // Get result
    float* resultData = (float*)finalRevenueBuffer->contents();
    float totalRevenue = resultData[0];

    std::cout << "TPC-H Query 6 Result:" << std::endl;
    std::cout << "Total Revenue: $" << std::fixed << std::setprecision(2) << totalRevenue << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    
    // Calculate effective bandwidth (rough estimate)
    size_t totalDataBytes = dataSize * (sizeof(int) + 3 * sizeof(float)); // All input columns
    double bandwidth = (totalDataBytes / (1024.0 * 1024.0 * 1024.0)) / gpuExecutionTime;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;

    // Cleanup
    stage1Function->release();
    stage1Pipeline->release();
    stage2Function->release();
    stage2Pipeline->release();
    shipdateBuffer->release();
    discountBuffer->release();
    quantityBuffer->release();
    extendedpriceBuffer->release();
    partialRevenuesBuffer->release();
    finalRevenueBuffer->release();
    stage1FunctionName->release();
    stage2FunctionName->release();
}


// C++ structs for reading final results
struct Q9Result {
    int nationkey;
    int year;
    float profit;
};

struct Q9Aggregates_CPU {
    uint key;
    float profit;
};


// --- Main Function for TPC-H Q9 Benchmark ---
void runQ9Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 9 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data for all SIX tables
    auto p_partkey = loadIntColumn(sf_path + "part.tbl", 0);
    auto p_name = loadCharColumn(sf_path + "part.tbl", 1, 55);
    auto s_suppkey = loadIntColumn(sf_path + "supplier.tbl", 0);
    auto s_nationkey = loadIntColumn(sf_path + "supplier.tbl", 3);
    auto l_partkey = loadIntColumn(sf_path + "lineitem.tbl", 1);
    auto l_suppkey = loadIntColumn(sf_path + "lineitem.tbl", 2);
    auto l_orderkey = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_quantity = loadFloatColumn(sf_path + "lineitem.tbl", 4);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    auto ps_partkey = loadIntColumn(sf_path + "partsupp.tbl", 0);
    auto ps_suppkey = loadIntColumn(sf_path + "partsupp.tbl", 1);
    auto ps_supplycost = loadFloatColumn(sf_path + "partsupp.tbl", 3);
    auto o_orderkey = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_orderdate = loadDateColumn(sf_path + "orders.tbl", 4);
    auto n_nationkey = loadIntColumn(sf_path + "nation.tbl", 0);
    auto n_name = loadCharColumn(sf_path + "nation.tbl", 1, 25);

    // Create a map for nation names
    std::map<int, std::string> nation_names;
    for (size_t i = 0; i < n_nationkey.size(); ++i) {
        nation_names[n_nationkey[i]] = std::string(&n_name[i * 25], 25);
    }
    
    // Get sizes
    const uint part_size = (uint)p_partkey.size(), supplier_size = (uint)s_suppkey.size(), lineitem_size = (uint)l_partkey.size();
    const uint partsupp_size = (uint)ps_partkey.size(), orders_size = (uint)o_orderkey.size();
    std::cout << "Loaded data for all tables." << std::endl;

    // 2. Setup all kernel pipelines
    NS::Error* pError = nullptr;
    MTL::Function* pPartBuildFn = pLibrary->newFunction(NS::String::string("q9_build_part_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartBuildPipe = pDevice->newComputePipelineState(pPartBuildFn, &pError);
    MTL::Function* pSuppBuildFn = pLibrary->newFunction(NS::String::string("q9_build_supplier_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pSuppBuildPipe = pDevice->newComputePipelineState(pSuppBuildFn, &pError);
    MTL::Function* pPartSuppBuildFn = pLibrary->newFunction(NS::String::string("q9_build_partsupp_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartSuppBuildPipe = pDevice->newComputePipelineState(pPartSuppBuildFn, &pError);
    MTL::Function* pOrdersBuildFn = pLibrary->newFunction(NS::String::string("q9_build_orders_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = pDevice->newComputePipelineState(pOrdersBuildFn, &pError);
    MTL::Function* pProbeAggFn = pLibrary->newFunction(NS::String::string("q9_probe_and_local_agg_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pProbeAggPipe = pDevice->newComputePipelineState(pProbeAggFn, &pError);
    MTL::Function* pMergeFn = pLibrary->newFunction(NS::String::string("q9_merge_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pMergePipe = pDevice->newComputePipelineState(pMergeFn, &pError);

    // 3. Create all GPU buffers
    const uint part_ht_size = part_size * 2;
    std::vector<int> cpu_part_ht(part_ht_size * 2, -1);
    MTL::Buffer* pPartKeyBuffer = pDevice->newBuffer(p_partkey.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartNameBuffer = pDevice->newBuffer(p_name.data(), p_name.size() * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartHTBuffer = pDevice->newBuffer(cpu_part_ht.data(), part_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);

    const uint supplier_ht_size = supplier_size * 2;
    std::vector<int> cpu_supplier_ht(supplier_ht_size * 2, -1);
    MTL::Buffer* pSuppKeyBuffer = pDevice->newBuffer(s_suppkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationKeyBuffer = pDevice->newBuffer(s_nationkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSupplierHTBuffer = pDevice->newBuffer(cpu_supplier_ht.data(), supplier_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);
    
    const uint partsupp_ht_size = partsupp_size * 2;
    std::vector<int> cpu_partsupp_ht(partsupp_ht_size * 2, -1);
    MTL::Buffer* pPsPartKeyBuffer = pDevice->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuffer = pDevice->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuffer = pDevice->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSuppHTBuffer = pDevice->newBuffer(cpu_partsupp_ht.data(), partsupp_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);
    
    const uint orders_ht_size = orders_size * 2;
    std::vector<int> cpu_orders_ht(orders_ht_size * 2, -1);
    MTL::Buffer* pOrdKeyBuffer = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuffer = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuffer = pDevice->newBuffer(cpu_orders_ht.data(), orders_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);

    MTL::Buffer* pLinePartKeyBuffer = pDevice->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuffer = pDevice->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuffer = pDevice->newBuffer(l_quantity.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    const uint num_threadgroups = 2048, local_ht_size = 128, intermediate_size = num_threadgroups * local_ht_size;
    MTL::Buffer* pIntermediateBuffer = pDevice->newBuffer(intermediate_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);
    const uint final_ht_size = 25 * 10; // 25 nations * ~10 years
    std::vector<uint> cpu_final_ht(final_ht_size * (sizeof(Q9Aggregates_CPU)/sizeof(uint)), 0);
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(cpu_final_ht.data(), final_ht_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);

    // 4. Dispatch the entire 6-stage pipeline
    MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
    
    MTL::ComputeCommandEncoder* pEnc1 = pCommandBuffer->computeCommandEncoder();
    pEnc1->setComputePipelineState(pPartBuildPipe);
    pEnc1->setBuffer(pPartKeyBuffer, 0, 0); pEnc1->setBuffer(pPartNameBuffer, 0, 1);
    pEnc1->setBuffer(pPartHTBuffer, 0, 2); pEnc1->setBytes(&part_size, sizeof(part_size), 3);
    pEnc1->setBytes(&part_ht_size, sizeof(part_ht_size), 4);
    pEnc1->dispatchThreads(MTL::Size(part_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc1->endEncoding();
    
    MTL::ComputeCommandEncoder* pEnc2 = pCommandBuffer->computeCommandEncoder();
    pEnc2->setComputePipelineState(pSuppBuildPipe);
    pEnc2->setBuffer(pSuppKeyBuffer, 0, 0); pEnc2->setBuffer(pSuppNationKeyBuffer, 0, 1);
    pEnc2->setBuffer(pSupplierHTBuffer, 0, 2); pEnc2->setBytes(&supplier_size, sizeof(supplier_size), 3);
    pEnc2->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 4);
    pEnc2->dispatchThreads(MTL::Size(supplier_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc2->endEncoding();

    MTL::ComputeCommandEncoder* pEnc3 = pCommandBuffer->computeCommandEncoder();
    pEnc3->setComputePipelineState(pPartSuppBuildPipe);
    pEnc3->setBuffer(pPsPartKeyBuffer, 0, 0); pEnc3->setBuffer(pPsSuppKeyBuffer, 0, 1);
    pEnc3->setBuffer(pPartSuppHTBuffer, 0, 2); pEnc3->setBytes(&partsupp_size, sizeof(partsupp_size), 3);
    pEnc3->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
    pEnc3->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc3->endEncoding();

    MTL::ComputeCommandEncoder* pEnc4 = pCommandBuffer->computeCommandEncoder();
    pEnc4->setComputePipelineState(pOrdersBuildPipe);
    pEnc4->setBuffer(pOrdKeyBuffer, 0, 0); pEnc4->setBuffer(pOrdDateBuffer, 0, 1);
    pEnc4->setBuffer(pOrdersHTBuffer, 0, 2); pEnc4->setBytes(&orders_size, sizeof(orders_size), 3);
    pEnc4->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
    pEnc4->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc4->endEncoding();
    
    MTL::ComputeCommandEncoder* pEnc5 = pCommandBuffer->computeCommandEncoder();
    pEnc5->setComputePipelineState(pProbeAggPipe);
    pEnc5->setBuffer(pLineSuppKeyBuffer, 0, 0); pEnc5->setBuffer(pLinePartKeyBuffer, 0, 1);
    pEnc5->setBuffer(pLineOrdKeyBuffer, 0, 2); pEnc5->setBuffer(pLinePriceBuffer, 0, 3);
    pEnc5->setBuffer(pLineDiscBuffer, 0, 4); pEnc5->setBuffer(pLineQtyBuffer, 0, 5);
    pEnc5->setBuffer(pPsSupplyCostBuffer, 0, 6); pEnc5->setBuffer(pPartHTBuffer, 0, 7);
    pEnc5->setBuffer(pSupplierHTBuffer, 0, 8); pEnc5->setBuffer(pPartSuppHTBuffer, 0, 9);
    pEnc5->setBuffer(pOrdersHTBuffer, 0, 10); pEnc5->setBuffer(pIntermediateBuffer, 0, 11);
    pEnc5->setBytes(&lineitem_size, sizeof(lineitem_size), 12); pEnc5->setBytes(&part_ht_size, sizeof(part_ht_size), 13);
    pEnc5->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 14); pEnc5->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 15);
    pEnc5->setBytes(&orders_ht_size, sizeof(orders_ht_size), 16);
    pEnc5->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    pEnc5->endEncoding();
    
    MTL::ComputeCommandEncoder* pEnc6 = pCommandBuffer->computeCommandEncoder();
    pEnc6->setComputePipelineState(pMergePipe);
    pEnc6->setBuffer(pIntermediateBuffer, 0, 0); pEnc6->setBuffer(pFinalHTBuffer, 0, 1);
    pEnc6->setBytes(&intermediate_size, sizeof(intermediate_size), 2); pEnc6->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
    pEnc6->dispatchThreads(MTL::Size(intermediate_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc6->endEncoding();

    // 5. Execute and time
    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();
    double gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();

    // 6. Process and print final results
    Q9Aggregates_CPU* results = (Q9Aggregates_CPU*)pFinalHTBuffer->contents();
    std::vector<Q9Result> final_results;
    for (uint i = 0; i < final_ht_size; ++i) {
        if (results[i].key != 0) {
            int nationkey = (results[i].key >> 16) & 0xFFFF;
            int year = results[i].key & 0xFFFF;
            final_results.push_back({nationkey, year, results[i].profit});
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q9Result& a, const Q9Result& b) {
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        return a.year > b.year;
    });

    printf("\nTPC-H Query 9 Results (Top 15):\n");
    printf("+------------+------+---------------+\n");
    printf("| Nation     | Year |        Profit |\n");
    printf("+------------+------+---------------+\n");
    for (int i = 0; i < 15 && i < final_results.size(); ++i) {
        printf("| %-10s | %4d | $%13.2f |\n",
               nation_names[final_results[i].nationkey].c_str(), final_results[i].year, final_results[i].profit);
    }
    printf("+------------+------+---------------+\n");
    printf("Total results found: %lu\n", final_results.size());
    printf("Total TPC-H Q9 GPU time: %f ms\n", gpuExecutionTime * 1000.0);
    
    // Release all functions and pipelines
    pPartBuildFn->release();
    pPartBuildPipe->release();
    pSuppBuildFn->release();
    pSuppBuildPipe->release();
    pPartSuppBuildFn->release();
    pPartSuppBuildPipe->release();
    pOrdersBuildFn->release();
    pOrdersBuildPipe->release();
    pProbeAggFn->release();
    pProbeAggPipe->release();
    pMergeFn->release();
    pMergePipe->release();
    
    // Release all buffers
    pPartKeyBuffer->release();
    pPartNameBuffer->release();
    pPartHTBuffer->release();
    pSuppKeyBuffer->release();
    pSuppNationKeyBuffer->release();
    pSupplierHTBuffer->release();
    pPsPartKeyBuffer->release();
    pPsSuppKeyBuffer->release();
    pPsSupplyCostBuffer->release();
    pPartSuppHTBuffer->release();
    pOrdKeyBuffer->release();
    pOrdDateBuffer->release();
    pOrdersHTBuffer->release();
    pLinePartKeyBuffer->release();
    pLineSuppKeyBuffer->release();
    pLineOrdKeyBuffer->release();
    pLineQtyBuffer->release();
    pLinePriceBuffer->release();
    pLineDiscBuffer->release();
    pIntermediateBuffer->release();
    pFinalHTBuffer->release();
}


// C++ structs for reading final Q13 results
struct Q13_OrderCount_CPU {
    int custkey;
    uint order_count;
};

struct Q13_CustDist_CPU {
    int c_count;
    uint custdist;
};

struct Q13Result {
    uint c_count;
    uint custdist;
};


// --- Main Function for TPC-H Q13 Benchmark ---
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 13 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_comment = loadCharColumn(sf_path + "orders.tbl", 8, 100);
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);

    const uint orders_size = (uint)o_custkey.size();
    const uint customer_size = (uint)c_custkey.size();
    std::cout << "Loaded " << orders_size << " orders and " << customer_size << " customers." << std::endl;

    // 2. Setup kernels
    NS::Error* pError = nullptr;
    MTL::Function* pLocalCountFn = pLibrary->newFunction(NS::String::string("q13_local_count_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pLocalCountPipe = pDevice->newComputePipelineState(pLocalCountFn, &pError);
    MTL::Function* pMergeCountFn = pLibrary->newFunction(NS::String::string("q13_merge_counts_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pMergeCountPipe = pDevice->newComputePipelineState(pMergeCountFn, &pError);
    MTL::Function* pLocalHistFn = pLibrary->newFunction(NS::String::string("q13_local_histogram_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pLocalHistPipe = pDevice->newComputePipelineState(pLocalHistFn, &pError);
    // NOTE: We no longer need the q13_merge_histogram_kernel pipeline state

    // 3. Create Buffers
    const uint num_threadgroups = 2048;
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCommentBuffer = pDevice->newBuffer(o_comment.data(), o_comment.size() * sizeof(char), MTL::ResourceStorageModeShared);
    
    const uint inter_count_size = num_threadgroups * 128;
    MTL::Buffer* pInterCountsBuffer = pDevice->newBuffer(inter_count_size * sizeof(Q13_OrderCount_CPU), MTL::ResourceStorageModeShared);
    
    const uint final_count_ht_size = customer_size * 2; // Larger to reduce collisions
    // CORRECTED INITIALIZATION: Use 0 to match the merge kernel's expectation for an empty key
    std::vector<uint> cpu_final_counts_ht(final_count_ht_size * (sizeof(Q13_OrderCount_CPU)/sizeof(uint)), 0);
    MTL::Buffer* pFinalCountsHTBuffer = pDevice->newBuffer(cpu_final_counts_ht.data(), final_count_ht_size * sizeof(Q13_OrderCount_CPU), MTL::ResourceStorageModeShared);

    MTL::Buffer* pCustKeyBuffer = pDevice->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    const uint inter_hist_size = num_threadgroups * 32;
    MTL::Buffer* pInterHistBuffer = pDevice->newBuffer(inter_hist_size * sizeof(Q13_CustDist_CPU), MTL::ResourceStorageModeShared);

    // 4. Dispatch the first 3 stages of the pipeline
    MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
    
    MTL::ComputeCommandEncoder* pEnc1 = pCommandBuffer->computeCommandEncoder();
    pEnc1->setComputePipelineState(pLocalCountPipe);
    pEnc1->setBuffer(pOrdCustKeyBuffer, 0, 0); pEnc1->setBuffer(pOrdCommentBuffer, 0, 1);
    pEnc1->setBuffer(pInterCountsBuffer, 0, 2); pEnc1->setBytes(&orders_size, sizeof(orders_size), 3);
    pEnc1->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    pEnc1->endEncoding();
    
    MTL::ComputeCommandEncoder* pEnc2 = pCommandBuffer->computeCommandEncoder();
    pEnc2->setComputePipelineState(pMergeCountPipe);
    pEnc2->setBuffer(pInterCountsBuffer, 0, 0); pEnc2->setBuffer(pFinalCountsHTBuffer, 0, 1);
    pEnc2->setBytes(&inter_count_size, sizeof(inter_count_size), 2);
    pEnc2->setBytes(&final_count_ht_size, sizeof(final_count_ht_size), 3);
    pEnc2->dispatchThreads(MTL::Size(inter_count_size, 1, 1), MTL::Size(1024, 1, 1));
    pEnc2->endEncoding();

    MTL::ComputeCommandEncoder* pEnc3 = pCommandBuffer->computeCommandEncoder();
    pEnc3->setComputePipelineState(pLocalHistPipe);
    pEnc3->setBuffer(pCustKeyBuffer, 0, 0); pEnc3->setBuffer(pFinalCountsHTBuffer, 0, 1);
    pEnc3->setBuffer(pInterHistBuffer, 0, 2); pEnc3->setBytes(&customer_size, sizeof(customer_size), 3);
    pEnc3->setBytes(&final_count_ht_size, sizeof(final_count_ht_size), 4);
    pEnc3->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    pEnc3->endEncoding();
    
    // NOTE: Stage 4 (final merge) is removed from the GPU pipeline

    // 5. Execute GPU work
    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();
    double gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();

    // 6. Perform final merge on CPU (The new, optimized Stage 4)
    Q13_CustDist_CPU* inter_results = (Q13_CustDist_CPU*)pInterHistBuffer->contents();
    std::map<uint, uint> final_histogram;
    for (uint i = 0; i < inter_hist_size; ++i) {
        if (inter_results[i].custdist > 0) {
            final_histogram[inter_results[i].c_count] += inter_results[i].custdist;
        }
    }
    
    // Add customers with 0 orders to correctly handle the LEFT JOIN
    uint total_customers_in_histogram = 0;
    for(auto const& [key, val] : final_histogram) {
        total_customers_in_histogram += val;
    }
    final_histogram[0] += (customer_size - total_customers_in_histogram);

    // 7. Process and print results
    std::vector<Q13Result> final_results;
    for(auto const& [key, val] : final_histogram) {
        final_results.push_back({key, val});
    }

    std::sort(final_results.begin(), final_results.end(), [](const Q13Result& a, const Q13Result& b) {
        if (a.custdist != b.custdist) return a.custdist > b.custdist;
        return a.c_count > b.c_count;
    });

    printf("\nTPC-H Query 13 Results:\n");
    printf("+---------+----------+\n");
    printf("| c_count | custdist |\n");
    printf("+---------+----------+\n");
    for(const auto& res : final_results) {
        printf("| %7u | %8u |\n", res.c_count, res.custdist);
    }
    printf("+---------+----------+\n");
    printf("Total TPC-H Q13 GPU time: %f ms\n", gpuExecutionTime * 1000.0);

    // Release objects...
    pLocalCountFn->release(); pLocalCountPipe->release();
    pMergeCountFn->release(); pMergeCountPipe->release();
    pLocalHistFn->release(); pLocalHistPipe->release();
    pOrdCustKeyBuffer->release(); pOrdCommentBuffer->release();
    pInterCountsBuffer->release(); pFinalCountsHTBuffer->release();
    pCustKeyBuffer->release(); pInterHistBuffer->release();
}


void showHelp() {
    std::cout << "GPU Database Mental Benchmark" << std::endl;
    std::cout << "Usage: GPUDBMentalBenchmark [query]" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Available queries:" << std::endl;
    std::cout << "  all           - Run all benchmarks (default)" << std::endl;
    std::cout << "  selection     - Run selection benchmark" << std::endl;
    std::cout << "  aggregation   - Run aggregation benchmark" << std::endl;
    std::cout << "  join          - Run join benchmark" << std::endl;
    std::cout << "  q1            - Run TPC-H Query 1 (Pricing Summary Report)" << std::endl;
    std::cout << "  q3            - Run TPC-H Query 3 (Shipping Priority)" << std::endl;
    std::cout << "  q6            - Run TPC-H Query 6 (Forecasting Revenue Change)" << std::endl;
    std::cout << "  q9            - Run TPC-H Query 9 (Product Type Profit Measure)" << std::endl;
    std::cout << "  q13           - Run TPC-H Query 13 (Customer Distribution)" << std::endl;
    std::cout << "  help          - Show this help message" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  GPUDBMentalBenchmark        # Run all benchmarks" << std::endl;
    std::cout << "  GPUDBMentalBenchmark q1     # Run only TPC-H Query 1" << std::endl;
    std::cout << "  GPUDBMentalBenchmark q3     # Run only TPC-H Query 3" << std::endl;
}

// --- Main Entry Point ---
int main(int argc, const char * argv[]) {
    // Parse command line arguments
    std::string query = "all"; // default to running all benchmarks
    if (argc > 1) {
        query = std::string(argv[1]);
        if (query == "help" || query == "--help" || query == "-h") {
            showHelp();
            return 0;
        }
        // Set dataset path based on argument
        if (query == "sf1") {
            g_dataset_path = "Data/SF-1/";
            query = "all"; // Run all benchmarks with SF-1
        } else if (query == "sf10") {
            g_dataset_path = "Data/SF-10/";
            query = "all"; // Run all benchmarks with SF-10
        }
    }

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
            pAutoreleasePool->release();
            return 1;
        }
    }

    // Run benchmarks based on command line argument
    if (query == "all") {
        // Run all benchmarks
        runSelectionBenchmark(device, commandQueue, library);
        runAggregationBenchmark(device, commandQueue, library);
        runJoinBenchmark(device, commandQueue, library);
        runQ1Benchmark(device, commandQueue, library);
        runQ3Benchmark(device, commandQueue, library);
        runQ6Benchmark(device, commandQueue, library);
        runQ9Benchmark(device, commandQueue, library);
        runQ13Benchmark(device, commandQueue, library);
    } else if (query == "selection") {
        runSelectionBenchmark(device, commandQueue, library);
    } else if (query == "aggregation") {
        runAggregationBenchmark(device, commandQueue, library);
    } else if (query == "join") {
        runJoinBenchmark(device, commandQueue, library);
    } else if (query == "q1") {
        runQ1Benchmark(device, commandQueue, library);
    } else if (query == "q3") {
        runQ3Benchmark(device, commandQueue, library);
    } else if (query == "q6") {
        runQ6Benchmark(device, commandQueue, library);
    } else if (query == "q9") {
        runQ9Benchmark(device, commandQueue, library);
    } else if (query == "q13") {
        runQ13Benchmark(device, commandQueue, library);
    } else {
        std::cerr << "Unknown query: " << query << std::endl;
        std::cerr << "Use 'help' to see available options." << std::endl;
        // Cleanup and exit
        library->release();
        commandQueue->release();
        device->release();
        pAutoreleasePool->release();
        return 1;
    }
    
    // Cleanup
    library->release();
    commandQueue->release();
    device->release();
    
    pAutoreleasePool->release();
    return 0;
}
