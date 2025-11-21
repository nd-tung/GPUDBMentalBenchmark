#include "JoinExecutor.hpp"
#include "ColumnStoreGPU.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <Metal/Metal.hpp>

namespace engine {

// Table schemas - map column names to indices
static const std::map<std::string, int> lineitem_schema = {
    {"l_orderkey", 0}, {"l_partkey", 1}, {"l_suppkey", 2}, {"l_linenumber", 3},
    {"l_quantity", 4}, {"l_extendedprice", 5}, {"l_discount", 6}, {"l_tax", 7},
    {"l_returnflag", 8}, {"l_linestatus", 9}, {"l_shipdate", 10}
};

static const std::map<std::string, int> orders_schema = {
    {"o_orderkey", 0}, {"o_custkey", 1}, {"o_orderstatus", 2}, 
    {"o_totalprice", 3}, {"o_orderdate", 4}
};

// Load integer key column
static std::vector<uint32_t> loadIntColumn(const std::string& filePath, int columnIndex) {
    std::vector<uint32_t> data;
    std::ifstream file(filePath);
    if (!file.is_open()) return data;
    
    std::string line;
    while (std::getline(file, line)) {
        std::string token;
        int col = 0;
        size_t s = 0, e = line.find('|');
        while (e != std::string::npos) {
            if (col == columnIndex) {
                token = line.substr(s, e - s);
                data.push_back(static_cast<uint32_t>(std::stoul(token)));
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

// Load float column (reuse from GpuExecutor)
static std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data;
    std::ifstream file(filePath);
    if (!file.is_open()) return data;
    
    std::string line;
    while (std::getline(file, line)) {
        std::string token;
        int col = 0;
        size_t s = 0, e = line.find('|');
        while (e != std::string::npos) {
            if (col == columnIndex) {
                token = line.substr(s, e - s);
                data.push_back(std::stof(token));
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

bool JoinExecutor::isEligible(const std::string& leftTable, const std::string& rightTable) {
    // For now, only support lineitem-orders join
    return (leftTable == "lineitem" && rightTable == "orders") ||
           (leftTable == "orders" && rightTable == "lineitem");
}

JoinResult JoinExecutor::runHashJoin(
    const std::string& dataset_path,
    const std::string& leftTable,
    const std::string& rightTable,
    const std::string& leftKeyColumn,
    const std::string& rightKeyColumn,
    const std::string& aggColumn,
    const std::vector<std::string>& /*predicateColumns*/)
{
    auto& store = ColumnStoreGPU::instance();
    store.initialize();  // Ensure Metal device is initialized
    
    if (!store.device() || !store.library()) {
        return {0.0, 0.0, 0.0, 0};
    }

    auto uploadStart = std::chrono::high_resolution_clock::now();

    // Load build side (smaller table - orders)
    std::string buildPath = dataset_path + rightTable + ".tbl";
    std::string probePath = dataset_path + leftTable + ".tbl";
    
    // Get column indices
    int buildKeyIdx = rightTable == "orders" ? orders_schema.at(rightKeyColumn) : lineitem_schema.at(rightKeyColumn);
    int probeKeyIdx = leftTable == "lineitem" ? lineitem_schema.at(leftKeyColumn) : orders_schema.at(leftKeyColumn);
    int probeValueIdx = leftTable == "lineitem" ? lineitem_schema.at(aggColumn) : orders_schema.at(aggColumn);
    
    // Load data
    auto buildKeys = loadIntColumn(buildPath, buildKeyIdx);
    auto probeKeys = loadIntColumn(probePath, probeKeyIdx);
    auto probeValues = loadFloatColumn(probePath, probeValueIdx);
    
    if (buildKeys.empty() || probeKeys.empty() || probeValues.empty()) {
        return {0.0, 0.0, 0.0, 0};
    }
    
    // Build side is smaller table (orders), probe side is larger (lineitem)
    
    // Create GPU buffers
    uint32_t buildCount = static_cast<uint32_t>(buildKeys.size());
    uint32_t probeCount = static_cast<uint32_t>(probeKeys.size());
    uint32_t htCapacity = buildCount * 2;  // 50% load factor
    
    auto buildKeyBuf = store.device()->newBuffer(buildKeys.data(), buildCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto probeKeyBuf = store.device()->newBuffer(probeKeys.data(), probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto probeValBuf = store.device()->newBuffer(probeValues.data(), probeCount * sizeof(float), MTL::ResourceStorageModeShared);
    
    // Hash table buffers
    auto htKeyBuf = store.device()->newBuffer(htCapacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto htPayloadBuf = store.device()->newBuffer(htCapacity * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(htKeyBuf->contents(), 0, htCapacity * sizeof(uint32_t));
    std::memset(htPayloadBuf->contents(), 0, htCapacity * sizeof(uint32_t));
    
    // Output buffers
    auto matchBuf = store.device()->newBuffer(probeCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto resultBuf = store.device()->newBuffer(probeCount * sizeof(float), MTL::ResourceStorageModeShared);
    std::memset(matchBuf->contents(), 0, probeCount * sizeof(uint32_t));
    
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    // Get kernels
    auto buildFnName = NS::String::string("ops::hash_join_build", NS::UTF8StringEncoding);
    auto probeFnName = NS::String::string("ops::hash_join_probe", NS::UTF8StringEncoding);
    
    MTL::Function* buildFn = store.library()->newFunction(buildFnName);
    MTL::Function* probeFn = store.library()->newFunction(probeFnName);
    
    if (!buildFn || !probeFn) {
        std::cerr << "[JOIN] Kernels not found\n";
        buildFnName->release();
        probeFnName->release();
        return {0.0, 0.0, upload_ms, 0};
    }
    
    NS::Error* error = nullptr;
    auto buildPipeline = store.device()->newComputePipelineState(buildFn, &error);
    auto probePipeline = store.device()->newComputePipelineState(probeFn, &error);
    
    buildFn->release();
    probeFn->release();
    buildFnName->release();
    probeFnName->release();
    
    if (!buildPipeline || !probePipeline) {
        if (error) std::cerr << "[JOIN] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
        return {0.0, 0.0, upload_ms, 0};
    }
    
    auto kernelStart = std::chrono::high_resolution_clock::now();
    
    // Build phase
    MTL::CommandBuffer* buildCmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* buildEnc = buildCmd->computeCommandEncoder();
    buildEnc->setComputePipelineState(buildPipeline);
    buildEnc->setBuffer(buildKeyBuf, 0, 0);
    buildEnc->setBuffer(buildKeyBuf, 0, 1);  // Use keys as payloads for simplicity
    buildEnc->setBuffer(htKeyBuf, 0, 2);
    buildEnc->setBuffer(htPayloadBuf, 0, 3);
    buildEnc->setBytes(&htCapacity, sizeof(htCapacity), 4);
    buildEnc->setBytes(&buildCount, sizeof(buildCount), 5);
    
    MTL::Size buildGrid = MTL::Size::Make(buildCount, 1, 1);
    MTL::Size buildThreadgroup = MTL::Size::Make(256, 1, 1);
    buildEnc->dispatchThreads(buildGrid, buildThreadgroup);
    buildEnc->endEncoding();
    buildCmd->commit();
    buildCmd->waitUntilCompleted();
    
    // Probe phase
    MTL::CommandBuffer* probeCmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* probeEnc = probeCmd->computeCommandEncoder();
    probeEnc->setComputePipelineState(probePipeline);
    probeEnc->setBuffer(probeKeyBuf, 0, 0);
    probeEnc->setBuffer(htKeyBuf, 0, 1);
    probeEnc->setBuffer(htPayloadBuf, 0, 2);
    probeEnc->setBuffer(matchBuf, 0, 3);
    probeEnc->setBuffer(resultBuf, 0, 4);
    probeEnc->setBytes(&htCapacity, sizeof(htCapacity), 5);
    probeEnc->setBytes(&probeCount, sizeof(probeCount), 6);
    
    MTL::Size probeGrid = MTL::Size::Make(probeCount, 1, 1);
    MTL::Size probeThreadgroup = MTL::Size::Make(256, 1, 1);
    probeEnc->dispatchThreads(probeGrid, probeThreadgroup);
    probeEnc->endEncoding();
    probeCmd->commit();
    probeCmd->waitUntilCompleted();
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    
    // Aggregate matched rows (CPU side for now)
    uint32_t* matches = reinterpret_cast<uint32_t*>(matchBuf->contents());
    float* values = reinterpret_cast<float*>(probeValBuf->contents());
    
    double sum = 0.0;
    uint32_t matchCount = 0;
    for (uint32_t i = 0; i < probeCount; ++i) {
        if (matches[i]) {
            sum += values[i];
            matchCount++;
        }
    }
    
    // Cleanup
    buildKeyBuf->release();
    probeKeyBuf->release();
    probeValBuf->release();
    htKeyBuf->release();
    htPayloadBuf->release();
    matchBuf->release();
    resultBuf->release();
    buildPipeline->release();
    probePipeline->release();
    
    // Aggregation complete
    
    return {sum, gpu_ms, upload_ms, matchCount};
}

} // namespace engine
