#include "SortExecutor.hpp"
#include "ColumnStoreGPU.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <iostream>
#include <cmath>
#include <Metal/Metal.hpp>

static const char* KERNEL_BITONIC_SORT = "ops::bitonic_sort_step";

namespace engine {

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

bool SortExecutor::isEligible(const std::string& orderByColumn, uint32_t rowCount) {
    // GPU bitonic sort requires power-of-2 size, we'll pad if needed
    // Limit to reasonable size for GPU memory
    if (rowCount > 16 * 1024 * 1024) return false;  // 16M rows max
    if (orderByColumn.empty()) return false;
    return true;
}

SortResult SortExecutor::runSort(const std::string& dataset_path,
                                 const std::string& table,
                                 const std::string& orderByColumn,
                                 bool ascending) {
    static const std::map<std::string, std::map<std::string, int>> column_indices = {
        {"lineitem", {
            {"l_orderkey", 0}, {"l_partkey", 1}, {"l_suppkey", 2}, {"l_linenumber", 3},
            {"l_quantity", 4}, {"l_extendedprice", 5}, {"l_discount", 6}, {"l_tax", 7},
            {"l_shipdate", 10}, {"l_commitdate", 11}, {"l_receiptdate", 12}
        }},
        {"orders", {
            {"o_orderkey", 0}, {"o_custkey", 1}, {"o_orderstatus", 2}, {"o_totalprice", 3},
            {"o_orderdate", 4}
        }}
    };
    
    std::string path = dataset_path + table + ".tbl";
    
    // Find column index
    auto tableIt = column_indices.find(table);
    if (tableIt == column_indices.end()) {
        std::cerr << "[SORT] Unknown table: " << table << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    auto colIt = tableIt->second.find(orderByColumn);
    if (colIt == tableIt->second.end()) {
        std::cerr << "[SORT] Unknown column: " << orderByColumn << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    auto uploadStart = std::chrono::high_resolution_clock::now();
    
    // Load column data
    std::vector<float> colData = loadFloatColumn(path, colIt->second);
    if (colData.empty()) {
        std::cerr << "[SORT] Failed to load column data" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    uint32_t actualCount = static_cast<uint32_t>(colData.size());
    
    // Bitonic sort requires power-of-2 size, pad with max/min values
    uint32_t paddedCount = 1;
    while (paddedCount < actualCount) paddedCount <<= 1;
    
    // Pad data with sentinel values (max float for ascending, min float for descending)
    float sentinelValue = ascending ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
    colData.resize(paddedCount, sentinelValue);
    
    // Initialize indices array
    std::vector<uint32_t> indices(paddedCount);
    for (uint32_t i = 0; i < paddedCount; ++i) {
        indices[i] = i;
    }
    
    auto& store = ColumnStoreGPU::instance();
    store.initialize();  // Ensure Metal is initialized
    if (!store.device() || !store.library()) {
        std::cerr << "[SORT] GPU not available" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    // Create GPU buffers
    auto dataBuffer = store.device()->newBuffer(colData.data(), 
                                               paddedCount * sizeof(float),
                                               MTL::ResourceStorageModeShared);
    auto indicesBuffer = store.device()->newBuffer(indices.data(),
                                                   paddedCount * sizeof(uint32_t),
                                                   MTL::ResourceStorageModeShared);
    
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    // Get kernel function
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string(KERNEL_BITONIC_SORT, NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            fnName->release();
            fnName = NS::String::string("bitonic_sort_step", NS::UTF8StringEncoding);
            fn = store.library()->newFunction(fnName);
        }
        if (!fn) {
            std::cerr << "[SORT] Kernel not found: bitonic_sort_step" << std::endl;
            fnName->release();
            dataBuffer->release();
            indicesBuffer->release();
            return {{}, 0.0, upload_ms};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release();
        fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[SORT] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            dataBuffer->release();
            indicesBuffer->release();
            return {{}, 0.0, upload_ms};
        }
    }
    
    // Execute bitonic sort passes
    auto kernelStart = std::chrono::high_resolution_clock::now();
    
    uint32_t numStages = static_cast<uint32_t>(std::log2(paddedCount));
    
    for (uint32_t stage = 1; stage <= numStages; ++stage) {
        for (uint32_t pass = 0; pass < stage; ++pass) {
            MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pipeline);
            
            // Set buffers and parameters
            enc->setBuffer(dataBuffer, 0, 0);
            enc->setBuffer(indicesBuffer, 0, 1);
            enc->setBytes(&stage, sizeof(stage), 2);
            enc->setBytes(&pass, sizeof(pass), 3);
            enc->setBytes(&paddedCount, sizeof(paddedCount), 4);
            
            // Dispatch threads (one thread per comparison pair)
            uint32_t numThreads = paddedCount / 2;
            NS::UInteger maxTG = pipeline->maxTotalThreadsPerThreadgroup();
            if (maxTG > numThreads) maxTG = numThreads;
            MTL::Size gridSize = MTL::Size::Make(numThreads, 1, 1);
            MTL::Size tgSize = MTL::Size::Make(maxTG, 1, 1);
            
            enc->dispatchThreads(gridSize, tgSize);
            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
        }
    }
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    
    // Read back sorted indices (only the actual data, not padding)
    uint32_t* indicesPtr = reinterpret_cast<uint32_t*>(indicesBuffer->contents());
    std::vector<uint32_t> sortedIndices;
    
    if (!ascending) {
        // For descending, reverse the order
        for (int i = actualCount - 1; i >= 0; --i) {
            if (indicesPtr[i] < actualCount) {
                sortedIndices.push_back(indicesPtr[i]);
            }
        }
    } else {
        // For ascending, take first actualCount indices
        for (uint32_t i = 0; i < paddedCount && sortedIndices.size() < actualCount; ++i) {
            if (indicesPtr[i] < actualCount) {
                sortedIndices.push_back(indicesPtr[i]);
            }
        }
    }
    
    // Release buffers
    dataBuffer->release();
    indicesBuffer->release();
    
    return {sortedIndices, gpu_ms, upload_ms};
}

} // namespace engine
