#include "GpuExecutor.hpp"
#include "ColumnStoreGPU.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <iostream>
#include <Metal/Metal.hpp>

// Kernel function name (namespaced in Operators.metal under 'ops')
static const char* KERNEL_SCAN_FILTER_SUM = "ops::scan_filter_sum_f32";

namespace engine {

static std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data; std::ifstream file(filePath); if (!file.is_open()) return data; std::string line;
    while (std::getline(file, line)) { std::string token; int col=0; size_t s=0,e=line.find('|');
        while (e!=std::string::npos) { if (col==columnIndex) { token=line.substr(s,e-s); data.push_back(std::stof(token)); break; }
            s=e+1; e=line.find('|', s); ++col; }
    } return data;
}

static std::vector<float> loadDateColumnAsFloat(const std::string& filePath, int columnIndex) {
    // Load date column and store YYYYMMDD integer as float bits for GPU
    std::vector<float> data; std::ifstream file(filePath); if (!file.is_open()) return data; std::string line;
    while (std::getline(file, line)) { std::string token; int col=0; size_t s=0,e=line.find('|');
        while (e!=std::string::npos) { if (col==columnIndex) {
            token=line.substr(s,e-s);
            token.erase(std::remove(token.begin(), token.end(), '-'), token.end());
            int date_int = std::stoi(token);
            union { int i; float f; } conv; conv.i = date_int;
            data.push_back(conv.f);  // Store int bits as float
            break;
        }
        s=e+1; e=line.find('|', s); ++col; }
    } return data;
}

bool GpuExecutor::isEligible(const std::string& aggFunc,
                             const std::vector<expr::Clause>& clauses,
                             const std::string& targetColumn) {
    std::string f = aggFunc; std::transform(f.begin(), f.end(), f.begin(), ::tolower);
    if (f != "sum") return false;
    if (targetColumn.empty()) return false;
    if (clauses.size() > 32) return false;
    return true;
}

// Packed predicate clause mirror of Metal struct (keep fields/order)
struct PredicateClausePacked {
    uint32_t colIndex; // always 0 for now (single column kernel)
    uint32_t op;       // 0..4
    uint32_t isDate;   // 0 numeric, 1 date
    int64_t value;     // lower 32 bits store float literal bits
};

GPUResult GpuExecutor::runSum(const std::string& dataset_path,
                              const std::string& targetColumn,
                              const std::vector<expr::Clause>& clauses) {
    static const std::map<std::string,int> float_idx{{"l_quantity",4},{"l_extendedprice",5},{"l_discount",6}};
    static const std::map<std::string,int> date_idx{{"l_shipdate",10}};
    std::string path = dataset_path + "lineitem.tbl";
    
    // Check if target is float or date column
    auto it = float_idx.find(targetColumn);
    auto date_it = date_idx.find(targetColumn);
    if (it==float_idx.end() && date_it==date_idx.end()) return {0.0,0.0,0.0};
    
    std::vector<float> colHost;
    if (it != float_idx.end()) {
        colHost = loadFloatColumn(path, it->second);
    } else {
        colHost = loadDateColumnAsFloat(path, date_it->second);
    }
    if (colHost.empty()) return {0.0,0.0,0.0};

    // Collect all unique columns referenced in predicates and target
    std::vector<std::string> neededCols;
    std::map<std::string,uint32_t> colIndexMap;
    neededCols.push_back(targetColumn);
    colIndexMap[targetColumn] = 0;
    for (const auto& c : clauses) {
        if (colIndexMap.find(c.ident)==colIndexMap.end()) {
            colIndexMap[c.ident] = static_cast<uint32_t>(neededCols.size());
            neededCols.push_back(c.ident);
        }
    }

    // Stage all columns on GPU
    auto uploadStart = std::chrono::high_resolution_clock::now();
    auto& store = ColumnStoreGPU::instance();
    std::vector<engine::GPUColumn*> gpuCols;
    for (const auto& col : neededCols) {
        std::vector<float> hostData;
        if (col==targetColumn) {
            hostData = colHost;
        } else {
            auto colIt = float_idx.find(col);
            auto dateIt = date_idx.find(col);
            if (colIt != float_idx.end()) {
                hostData = loadFloatColumn(path, colIt->second);
            } else if (dateIt != date_idx.end()) {
                hostData = loadDateColumnAsFloat(path, dateIt->second);
            } else {
                return {0.0,0.0,0.0};
            }
        }
        if (hostData.empty()) return {0.0,0.0,0.0};
        engine::GPUColumn* gc = store.stageFloatColumn(col, hostData);
        gpuCols.push_back(gc);
    }
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    if (gpuCols.empty() || !gpuCols[0] || !store.device() || !store.library()) {
        // Fallback CPU sum
        double sum=0.0; for (float v: colHost) sum += v; return {sum, 0.0, upload_ms};
    }

    // Build predicate clause buffer with proper column indices
    std::vector<PredicateClausePacked> packed;
    packed.reserve(clauses.size());
    for (const auto& c : clauses) {
        PredicateClausePacked pc{}; 
        pc.colIndex = colIndexMap[c.ident]; 
        pc.isDate = c.isDate ? 1u : 0u;
        switch (c.op) {
            case expr::CompOp::LT: pc.op = 0; break;
            case expr::CompOp::LE: pc.op = 1; break;
            case expr::CompOp::GT: pc.op = 2; break;
            case expr::CompOp::GE: pc.op = 3; break;
            case expr::CompOp::EQ: pc.op = 4; break;
        }
        if (c.isDate) {
            pc.value = static_cast<int64_t>(c.date);
        } else {
            union { float f; uint32_t u; } conv; conv.f = static_cast<float>(c.num);
            pc.value = static_cast<int64_t>(conv.u);
        }
        packed.push_back(pc);
    }
    auto predicateBuffer = store.device()->newBuffer(packed.data(), packed.size()*sizeof(PredicateClausePacked), MTL::ResourceStorageModeShared);

    // Prepare output atomic sum (uint32 bits for float)
    auto outSumBuffer = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outSumBuffer->contents(), 0, sizeof(uint32_t));

    // Acquire function & pipeline (cache naive for now)
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string(KERNEL_SCAN_FILTER_SUM, NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            // Try without namespace as fallback
            fnName->release(); fnName = NS::String::string("scan_filter_sum_f32", NS::UTF8StringEncoding);
            fn = store.library()->newFunction(fnName);
        }
        if (!fn) {
            std::cerr << "[GPU] Kernel not found: scan_filter_sum_f32" << std::endl;
            fnName->release();
            double sum=0.0; for(float v: colHost) sum += v; return {sum,0.0,upload_ms};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release(); fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[GPU] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            double sum=0.0; for(float v: colHost) sum += v; return {sum,0.0,upload_ms};
        }
    }

    // Encode and dispatch
    auto kernelStart = std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    // Set column buffers at indices 0-7 (kernel expects up to 8 columns)
    for (size_t i=0; i<gpuCols.size() && i<8; ++i) {
        enc->setBuffer(gpuCols[i]->buffer, 0, i);
    }
    // Fill unused slots with first buffer (dummy, won't be accessed if col_count is correct)
    for (size_t i=gpuCols.size(); i<8; ++i) {
        enc->setBuffer(gpuCols[0]->buffer, 0, i);
    }
    // Predicates at buffer(8), parameters at 9-11, output at 12
    enc->setBuffer(predicateBuffer, 0, 8);
    uint32_t colCount = static_cast<uint32_t>(gpuCols.size());
    uint32_t clauseCount = static_cast<uint32_t>(packed.size());
    uint32_t rowCount = static_cast<uint32_t>(gpuCols[0]->count);
    enc->setBytes(&colCount, sizeof(colCount), 9);
    enc->setBytes(&clauseCount, sizeof(clauseCount), 10);
    enc->setBytes(&rowCount, sizeof(rowCount), 11);
    enc->setBuffer(outSumBuffer, 0, 12);
    // Threads
    NS::UInteger maxTG = pipeline->maxTotalThreadsPerThreadgroup();
    if (maxTG > rowCount) maxTG = rowCount;
    MTL::Size gridSize = MTL::Size::Make(rowCount, 1, 1);
    MTL::Size tgSize = MTL::Size::Make(maxTG, 1, 1);
    enc->dispatchThreads(gridSize, tgSize);
    enc->endEncoding();
    cmd->commit(); cmd->waitUntilCompleted();
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    double gpu_ms = (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    // Fallback to wall if timestamps unavailable
    if (gpu_ms <= 0.0) gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();

    // Read back
    uint32_t sumBits = *reinterpret_cast<uint32_t*>(outSumBuffer->contents());
    union { uint32_t u; float f; } conv; conv.u = sumBits;
    double revenue = static_cast<double>(conv.f);

    // Release temp buffers (keep pipeline & staged column cached)
    predicateBuffer->release();
    outSumBuffer->release();

    return {revenue, gpu_ms, upload_ms};
}

} // namespace engine
