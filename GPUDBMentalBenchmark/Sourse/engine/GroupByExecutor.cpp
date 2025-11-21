#include "GroupByExecutor.hpp"
#include "ColumnStoreGPU.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <iostream>
#include <Metal/Metal.hpp>

static const char* KERNEL_GROUPBY = "ops::groupby_agg_single_key";

namespace engine {

static std::vector<uint32_t> loadUInt32Column(const std::string& filePath, int columnIndex) {
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
                data.push_back(std::stoul(token));
                break;
            }
            s = e + 1;
            e = line.find('|', s);
            ++col;
        }
    }
    return data;
}

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

bool GroupByExecutor::isEligible(const std::string& groupByColumn, const std::string& aggColumn) {
    if (groupByColumn.empty() || aggColumn.empty()) return false;
    // For now, only support integer group keys
    return true;
}

GroupByResult GroupByExecutor::runGroupBySum(const std::string& dataset_path,
                                             const std::string& table,
                                             const std::string& groupByColumn,
                                             const std::string& aggColumn) {
    static const std::map<std::string, std::map<std::string, int>> column_indices = {
        {"lineitem", {
            {"l_orderkey", 0}, {"l_partkey", 1}, {"l_suppkey", 2}, {"l_linenumber", 3},
            {"l_quantity", 4}, {"l_extendedprice", 5}, {"l_discount", 6}, {"l_tax", 7},
            {"l_returnflag", 8}, {"l_linestatus", 9}
        }},
        {"orders", {
            {"o_orderkey", 0}, {"o_custkey", 1}, {"o_totalprice", 3}
        }}
    };
    
    std::string path = dataset_path + table + ".tbl";
    
    // Find column indices
    auto tableIt = column_indices.find(table);
    if (tableIt == column_indices.end()) {
        std::cerr << "[GROUPBY] Unknown table: " << table << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    auto groupColIt = tableIt->second.find(groupByColumn);
    auto aggColIt = tableIt->second.find(aggColumn);
    if (groupColIt == tableIt->second.end() || aggColIt == tableIt->second.end()) {
        std::cerr << "[GROUPBY] Unknown column" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    auto uploadStart = std::chrono::high_resolution_clock::now();
    
    // Load data
    std::vector<uint32_t> groupKeys = loadUInt32Column(path, groupColIt->second);
    std::vector<float> aggValues = loadFloatColumn(path, aggColIt->second);
    
    if (groupKeys.empty() || aggValues.empty() || groupKeys.size() != aggValues.size()) {
        std::cerr << "[GROUPBY] Failed to load data" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    uint32_t rowCount = static_cast<uint32_t>(groupKeys.size());
    
    // Find unique key count for hash table sizing
    std::vector<uint32_t> uniqueKeys = groupKeys;
    std::sort(uniqueKeys.begin(), uniqueKeys.end());
    uniqueKeys.erase(std::unique(uniqueKeys.begin(), uniqueKeys.end()), uniqueKeys.end());
    uint32_t uniqueCount = static_cast<uint32_t>(uniqueKeys.size());
    
    // Hash table capacity (2x unique keys)
    uint32_t capacity = uniqueCount * 2;
    
    auto& store = ColumnStoreGPU::instance();
    store.initialize();
    if (!store.device() || !store.library()) {
        std::cerr << "[GROUPBY] GPU not available" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    // Create GPU buffers
    auto keysBuffer = store.device()->newBuffer(groupKeys.data(),
                                                rowCount * sizeof(uint32_t),
                                                MTL::ResourceStorageModeShared);
    auto valuesBuffer = store.device()->newBuffer(aggValues.data(),
                                                  rowCount * sizeof(float),
                                                  MTL::ResourceStorageModeShared);
    
    // Hash table buffers (initialized to zero)
    auto htKeysBuffer = store.device()->newBuffer(capacity * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
    auto htCountsBuffer = store.device()->newBuffer(capacity * sizeof(uint32_t),
                                                    MTL::ResourceStorageModeShared);
    auto htSumsBuffer = store.device()->newBuffer(capacity * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
    std::memset(htKeysBuffer->contents(), 0, capacity * sizeof(uint32_t));
    std::memset(htCountsBuffer->contents(), 0, capacity * sizeof(uint32_t));
    std::memset(htSumsBuffer->contents(), 0, capacity * sizeof(uint32_t));
    
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    // Get kernel
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string(KERNEL_GROUPBY, NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            fnName->release();
            fnName = NS::String::string("groupby_agg_single_key", NS::UTF8StringEncoding);
            fn = store.library()->newFunction(fnName);
        }
        if (!fn) {
            std::cerr << "[GROUPBY] Kernel not found" << std::endl;
            fnName->release();
            keysBuffer->release();
            valuesBuffer->release();
            htKeysBuffer->release();
            htCountsBuffer->release();
            htSumsBuffer->release();
            return {{}, 0.0, upload_ms};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release();
        fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[GROUPBY] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            keysBuffer->release();
            valuesBuffer->release();
            htKeysBuffer->release();
            htCountsBuffer->release();
            htSumsBuffer->release();
            return {{}, 0.0, upload_ms};
        }
    }
    
    // Execute kernel
    auto kernelStart = std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    
    enc->setBuffer(keysBuffer, 0, 0);
    enc->setBuffer(valuesBuffer, 0, 1);
    enc->setBuffer(htKeysBuffer, 0, 2);
    enc->setBuffer(htCountsBuffer, 0, 3);
    enc->setBuffer(htSumsBuffer, 0, 4);
    enc->setBytes(&capacity, sizeof(capacity), 5);
    enc->setBytes(&rowCount, sizeof(rowCount), 6);
    
    NS::UInteger maxTG = pipeline->maxTotalThreadsPerThreadgroup();
    if (maxTG > rowCount) maxTG = rowCount;
    MTL::Size gridSize = MTL::Size::Make(rowCount, 1, 1);
    MTL::Size tgSize = MTL::Size::Make(maxTG, 1, 1);
    
    enc->dispatchThreads(gridSize, tgSize);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    
    // Read back results
    uint32_t* htKeys = reinterpret_cast<uint32_t*>(htKeysBuffer->contents());
    uint32_t* htSums = reinterpret_cast<uint32_t*>(htSumsBuffer->contents());
    
    std::map<uint32_t, double> results;
    for (uint32_t i = 0; i < capacity; ++i) {
        if (htKeys[i] != 0) {  // Non-empty slot
            union { uint32_t u; float f; } conv;
            conv.u = htSums[i];
            results[htKeys[i]] = static_cast<double>(conv.f);
        }
    }
    
    // Release buffers
    keysBuffer->release();
    valuesBuffer->release();
    htKeysBuffer->release();
    htCountsBuffer->release();
    htSumsBuffer->release();
    
    return {results, gpu_ms, upload_ms};
}

} // namespace engine
