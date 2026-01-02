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

[[maybe_unused]] static const char* KERNEL_GROUPBY_SINGLE = "ops::groupby_agg_single_key";
static const char* KERNEL_GROUPBY_MULTI = "ops::groupby_agg_multi_key";

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
                // Handle string columns by hashing (simple FNV-1a)
                if (!token.empty() && !std::isdigit(token[0])) {
                    uint32_t hash = 2166136261u;
                    for (char c : token) {
                        hash ^= static_cast<uint8_t>(c);
                        hash *= 16777619u;
                    }
                    data.push_back(hash);
                } else {
                    data.push_back(token.empty() ? 0 : std::stoul(token));
                }
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

bool GroupByExecutor::isEligible(const std::vector<std::string>& groupByColumns, 
                                 const std::vector<std::string>& aggColumns) {
    if (groupByColumns.empty() || aggColumns.empty()) return false;
    if (groupByColumns.size() > 4 || aggColumns.size() > 4) return false;  // Hardware limit
    return true;
}

GroupByResult GroupByExecutor::runGroupBy(const std::string& dataset_path,
                                          const std::string& table,
                                          const std::vector<std::string>& groupByColumns,
                                          const std::vector<std::string>& aggColumns,
                                          const std::vector<std::string>& aggFuncs) {
    (void)aggFuncs;
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
    
    std::vector<int> groupIdxs, aggIdxs;
    for (const auto& col : groupByColumns) {
        auto it = tableIt->second.find(col);
        if (it == tableIt->second.end()) {
            std::cerr << "[GROUPBY] Unknown group column: " << col << std::endl;
            return {{}, 0.0, 0.0};
        }
        groupIdxs.push_back(it->second);
    }
    for (const auto& col : aggColumns) {
        auto it = tableIt->second.find(col);
        if (it == tableIt->second.end()) {
            std::cerr << "[GROUPBY] Unknown agg column: " << col << std::endl;
            return {{}, 0.0, 0.0};
        }
        aggIdxs.push_back(it->second);
    }
    
    auto uploadStart = std::chrono::high_resolution_clock::now();
    
    // Load group key columns
    std::vector<std::vector<uint32_t>> groupKeyCols;
    for (int idx : groupIdxs) {
        groupKeyCols.push_back(loadUInt32Column(path, idx));
    }
    
    // Load aggregate value columns
    std::vector<std::vector<float>> aggValCols;
    for (int idx : aggIdxs) {
        aggValCols.push_back(loadFloatColumn(path, idx));
    }
    
    if (groupKeyCols.empty() || aggValCols.empty() || groupKeyCols[0].empty()) {
        std::cerr << "[GROUPBY] Failed to load data" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    uint32_t rowCount = static_cast<uint32_t>(groupKeyCols[0].size());

    
    // Estimate unique key combinations for hash table sizing
    // For multi-column, use heuristic: min(product of cardinalities, row_count / 2)
    uint32_t estimatedUnique = std::min(rowCount / 2, rowCount / 10);
    uint32_t capacity = estimatedUnique * 3;  // 3x for hash table load factor
    
    auto& store = ColumnStoreGPU::instance();
    store.initialize();
    if (!store.device() || !store.library()) {
        std::cerr << "[GROUPBY] GPU not available" << std::endl;
        return {{}, 0.0, 0.0};
    }
    
    uint32_t numKeys = static_cast<uint32_t>(groupKeyCols.size());
    uint32_t numAggs = static_cast<uint32_t>(aggValCols.size());
    
    // Create GPU buffers for key columns (up to 4)
    std::vector<MTL::Buffer*> keyBuffers(4, nullptr);
    for (uint32_t i = 0; i < numKeys && i < 4; ++i) {
        keyBuffers[i] = store.device()->newBuffer(groupKeyCols[i].data(),
                                                  rowCount * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
    }
    // Dummy buffers for unused key slots
    std::vector<uint32_t> dummy(rowCount, 0);
    for (uint32_t i = numKeys; i < 4; ++i) {
        keyBuffers[i] = store.device()->newBuffer(dummy.data(),
                                                  rowCount * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
    }
    
    // Create GPU buffers for aggregate columns (up to 4)
    std::vector<MTL::Buffer*> aggBuffers(4, nullptr);
    for (uint32_t i = 0; i < numAggs && i < 4; ++i) {
        aggBuffers[i] = store.device()->newBuffer(aggValCols[i].data(),
                                                  rowCount * sizeof(float),
                                                  MTL::ResourceStorageModeShared);
    }
    // Dummy buffers for unused agg slots
    std::vector<float> dummyF(rowCount, 0.0f);
    for (uint32_t i = numAggs; i < 4; ++i) {
        aggBuffers[i] = store.device()->newBuffer(dummyF.data(),
                                                  rowCount * sizeof(float),
                                                  MTL::ResourceStorageModeShared);
    }
    
    // Hash table buffers: capacity * 4 slots for keys and aggregates
    auto htKeysBuffer = store.device()->newBuffer(capacity * 4 * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
    auto htAggBuffer = store.device()->newBuffer(capacity * 4 * sizeof(uint32_t),
                                                 MTL::ResourceStorageModeShared);
    std::memset(htKeysBuffer->contents(), 0, capacity * 4 * sizeof(uint32_t));
    std::memset(htAggBuffer->contents(), 0, capacity * 4 * sizeof(uint32_t));
    
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    // Get kernel
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string(KERNEL_GROUPBY_MULTI, NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            std::cerr << "[GROUPBY] Kernel not found: " << KERNEL_GROUPBY_MULTI << std::endl;
            fnName->release();
            // Cleanup
            for (auto* buf : keyBuffers) buf->release();
            for (auto* buf : aggBuffers) buf->release();
            htKeysBuffer->release();
            htAggBuffer->release();
            return {{}, 0.0, upload_ms};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release();
        fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[GROUPBY] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            for (auto* buf : keyBuffers) buf->release();
            for (auto* buf : aggBuffers) buf->release();
            htKeysBuffer->release();
            htAggBuffer->release();
            return {{}, 0.0, upload_ms};
        }
    }
    
    // Execute kernel
    auto kernelStart = std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    
    // Bind key columns (buffers 0-3)
    for (uint32_t i = 0; i < 4; ++i) {
        enc->setBuffer(keyBuffers[i], 0, i);
    }
    // Bind agg columns (buffers 4-7)
    for (uint32_t i = 0; i < 4; ++i) {
        enc->setBuffer(aggBuffers[i], 0, 4 + i);
    }
    // Bind hash table buffers
    enc->setBuffer(htKeysBuffer, 0, 8);
    enc->setBuffer(htAggBuffer, 0, 9);
    // Bind constants
    enc->setBytes(&capacity, sizeof(capacity), 10);
    enc->setBytes(&rowCount, sizeof(rowCount), 11);
    enc->setBytes(&numKeys, sizeof(numKeys), 12);
    enc->setBytes(&numAggs, sizeof(numAggs), 13);
    
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
    uint32_t* htAggs = reinterpret_cast<uint32_t*>(htAggBuffer->contents());
    
    std::map<std::vector<uint32_t>, std::vector<double>> results;
    for (uint32_t i = 0; i < capacity; ++i) {
        uint32_t baseKeyIdx = i * 4;
        // Check if slot is occupied (any key non-zero)
        bool occupied = false;
        for (uint32_t k = 0; k < numKeys; ++k) {
            if (htKeys[baseKeyIdx + k] != 0) {
                occupied = true;
                break;
            }
        }
        
        if (occupied) {
            std::vector<uint32_t> key(numKeys);
            for (uint32_t k = 0; k < numKeys; ++k) {
                key[k] = htKeys[baseKeyIdx + k];
            }
            
            std::vector<double> aggs(numAggs);
            uint32_t baseAggIdx = i * 4;
            for (uint32_t a = 0; a < numAggs; ++a) {
                union { uint32_t u; float f; } conv;
                conv.u = htAggs[baseAggIdx + a];
                aggs[a] = static_cast<double>(conv.f);
            }

            auto it = results.find(key);
            if (it == results.end()) {
                results.emplace(std::move(key), std::move(aggs));
            } else {
                for (uint32_t a = 0; a < numAggs; ++a) {
                    it->second[a] += aggs[a];
                }
            }
        }
    }
    
    // Release buffers
    for (auto* buf : keyBuffers) buf->release();
    for (auto* buf : aggBuffers) buf->release();
    htKeysBuffer->release();
    htAggBuffer->release();
    
    return {results, gpu_ms, upload_ms};
}

// Legacy single-column interface for backwards compatibility
GroupByResult GroupByExecutor::runGroupBySum(const std::string& dataset_path,
                                             const std::string& table,
                                             const std::string& groupByColumn,
                                             const std::string& aggColumn) {
    // Delegate to multi-column version
    return runGroupBy(dataset_path, table, 
                     {groupByColumn}, {aggColumn}, {"sum"});
}

} // namespace engine
