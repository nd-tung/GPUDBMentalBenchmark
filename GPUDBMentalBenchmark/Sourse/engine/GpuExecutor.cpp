#include "GpuExecutor.hpp"
#include "ColumnStoreGPU.hpp"
#include "Expression.hpp"
#include "Predicate.hpp"
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
    if (f != "sum" && f != "count" && f != "avg" && f != "min" && f != "max") return false;
    if (targetColumn.empty() && f != "count") return false;  // COUNT(*) is ok
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

// Multi-column predicate clause for scan_filter_aggregate kernel
struct PredicateClause {
    uint32_t colIndex;
    uint32_t op;
    uint32_t isDate;
    int64_t value;
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

    return {revenue, gpu_ms, upload_ms, 0};
}

// New kernel for expression evaluation
static const char* KERNEL_SCAN_FILTER_EVAL_SUM = "ops::scan_filter_eval_sum";

GPUResult GpuExecutor::runSumWithExpression(const std::string& dataset_path,
                                            const std::string& expression,
                                            const std::vector<expr::Clause>& clauses) {
    // Parse expression
    expr::ParsedExpression parsed;
    try {
        parsed = expr::ParsedExpression::parse(expression);
    } catch (const std::exception& e) {
        std::cerr << "[GPU] Expression parse error: " << e.what() << std::endl;
        return {0.0, 0.0, 0.0};
    }
    
    static const std::map<std::string,int> float_idx{{"l_quantity",4},{"l_extendedprice",5},{"l_discount",6},{"l_tax",7}};
    static const std::map<std::string,int> date_idx{{"l_shipdate",10}};
    std::string path = dataset_path + "lineitem.tbl";
    
    // Collect all unique columns referenced (expression + predicates)
    std::vector<std::string> neededCols = parsed.columns;
    std::map<std::string,uint32_t> colIndexMap;
    for (size_t i=0; i<neededCols.size(); ++i) {
        colIndexMap[neededCols[i]] = static_cast<uint32_t>(i);
    }
    
    // Add predicate columns
    for (const auto& c : clauses) {
        if (colIndexMap.find(c.ident)==colIndexMap.end()) {
            colIndexMap[c.ident] = static_cast<uint32_t>(neededCols.size());
            neededCols.push_back(c.ident);
        }
    }
    
    // Load all columns
    auto uploadStart = std::chrono::high_resolution_clock::now();
    auto& store = ColumnStoreGPU::instance();
    std::vector<engine::GPUColumn*> gpuCols;
    uint32_t rowCount = 0;
    
    for (const auto& col : neededCols) {
        std::vector<float> hostData;
        auto colIt = float_idx.find(col);
        auto dateIt = date_idx.find(col);
        if (colIt != float_idx.end()) {
            hostData = loadFloatColumn(path, colIt->second);
        } else if (dateIt != date_idx.end()) {
            hostData = loadDateColumnAsFloat(path, dateIt->second);
        } else {
            std::cerr << "[GPU] Unknown column: " << col << std::endl;
            return {0.0,0.0,0.0};
        }
        if (hostData.empty()) return {0.0,0.0,0.0};
        if (rowCount == 0) rowCount = static_cast<uint32_t>(hostData.size());
        engine::GPUColumn* gc = store.stageFloatColumn(col, hostData);
        gpuCols.push_back(gc);
    }
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    if (gpuCols.empty() || !store.device() || !store.library()) {
        return {0.0, 0.0, upload_ms};
    }
    
    // Build predicate clause buffer
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
    MTL::Buffer* predicateBuffer;
    if (packed.empty()) {
        // Create empty buffer
        predicateBuffer = store.device()->newBuffer(sizeof(PredicateClausePacked), MTL::ResourceStorageModeShared);
    } else {
        predicateBuffer = store.device()->newBuffer(packed.data(), 
                                                    packed.size()*sizeof(PredicateClausePacked), 
                                                    MTL::ResourceStorageModeShared);
    }
    
    // Build expression RPN buffer
    MTL::Buffer* exprBuffer;
    if (parsed.rpn.empty()) {
        exprBuffer = store.device()->newBuffer(sizeof(expr::ExprToken), MTL::ResourceStorageModeShared);
    } else {
        exprBuffer = store.device()->newBuffer(parsed.rpn.data(),
                                               parsed.rpn.size()*sizeof(expr::ExprToken),
                                               MTL::ResourceStorageModeShared);
    }
    
    // Output atomic sum
    auto outSumBuffer = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    std::memset(outSumBuffer->contents(), 0, sizeof(uint32_t));
    
    // Acquire function & pipeline
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string(KERNEL_SCAN_FILTER_EVAL_SUM, NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            fnName->release(); fnName = NS::String::string("scan_filter_eval_sum", NS::UTF8StringEncoding);
            fn = store.library()->newFunction(fnName);
        }
        if (!fn) {
            std::cerr << "[GPU] Kernel not found: scan_filter_eval_sum" << std::endl;
            fnName->release();
            predicateBuffer->release();
            exprBuffer->release();
            outSumBuffer->release();
            return {0.0,0.0,upload_ms};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release(); fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[GPU] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            predicateBuffer->release();
            exprBuffer->release();
            outSumBuffer->release();
            return {0.0,0.0,upload_ms};
        }
    }
    
    // Encode and dispatch
    auto kernelStart = std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    
    // Set column buffers at indices 0-7
    for (size_t i=0; i<gpuCols.size() && i<8; ++i) {
        enc->setBuffer(gpuCols[i]->buffer, 0, i);
    }
    for (size_t i=gpuCols.size(); i<8; ++i) {
        enc->setBuffer(gpuCols[0]->buffer, 0, i);  // dummy fill
    }
    
    // Predicates at buffer(8), expr_rpn at buffer(9), parameters at 10-13, output at 14
    enc->setBuffer(predicateBuffer, 0, 8);
    enc->setBuffer(exprBuffer, 0, 9);
    uint32_t colCount = static_cast<uint32_t>(gpuCols.size());
    uint32_t clauseCount = static_cast<uint32_t>(packed.size());
    uint32_t exprLength = static_cast<uint32_t>(parsed.rpn.size());
    enc->setBytes(&colCount, sizeof(colCount), 10);
    enc->setBytes(&clauseCount, sizeof(clauseCount), 11);
    enc->setBytes(&exprLength, sizeof(exprLength), 12);
    enc->setBytes(&rowCount, sizeof(rowCount), 13);
    enc->setBuffer(outSumBuffer, 0, 14);
    
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
    if (gpu_ms <= 0.0) gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    
    // Read back
    uint32_t sumBits = *reinterpret_cast<uint32_t*>(outSumBuffer->contents());
    union { uint32_t u; float f; } conv; conv.u = sumBits;
    double revenue = static_cast<double>(conv.f);
    
    // Release temp buffers
    predicateBuffer->release();
    exprBuffer->release();
    outSumBuffer->release();
    
    return {revenue, gpu_ms, upload_ms, 0};
}

GPUResult GpuExecutor::runAggregate(const std::string& dataset_path,
                                    const std::string& aggFunc,
                                    const std::string& targetColumn,
                                    const std::vector<expr::Clause>& clauses) {
    // Map aggregation function to type: COUNT=0, SUM=1, AVG=2, MIN=3, MAX=4
    uint32_t aggType;
    std::string lowerFunc = aggFunc;
    std::transform(lowerFunc.begin(), lowerFunc.end(), lowerFunc.begin(), ::tolower);
    
    if (lowerFunc == "count") aggType = 0;
    else if (lowerFunc == "sum") aggType = 1;
    else if (lowerFunc == "avg") aggType = 2;
    else if (lowerFunc == "min") aggType = 3;
    else if (lowerFunc == "max") aggType = 4;
    else return {0.0, 0.0, 0.0, 0};
    
    // Column schema
    static const std::map<std::string, int> lineitem_schema = {
        {"l_orderkey", 0}, {"l_partkey", 1}, {"l_suppkey", 2}, {"l_linenumber", 3},
        {"l_quantity", 4}, {"l_extendedprice", 5}, {"l_discount", 6}, {"l_tax", 7},
        {"l_returnflag", 8}, {"l_linestatus", 9}, {"l_shipdate", 10}, {"l_commitdate", 11},
        {"l_receiptdate", 12}, {"l_shipinstruct", 13}, {"l_shipmode", 14}, {"l_comment", 15}
    };
    
    auto uploadStart = std::chrono::high_resolution_clock::now();
    
    // Load target column (for COUNT, any column works, use first one)
    std::string filePath = dataset_path + "lineitem.tbl";
    int targetIdx = (aggType == 0 && targetColumn == "*") ? 4 : lineitem_schema.at(targetColumn);
    std::vector<float> targetData = loadFloatColumn(filePath, targetIdx);
    if (targetData.empty()) return {0.0, 0.0, 0.0, 0};
    
    uint32_t rowCount = static_cast<uint32_t>(targetData.size());
    
    // Load columns referenced in predicates
    std::vector<std::string> predicateColumns;
    for (const auto& cl : clauses) {
        if (std::find(predicateColumns.begin(), predicateColumns.end(), cl.ident) == predicateColumns.end()) {
            predicateColumns.push_back(cl.ident);
        }
    }
    
    // GPU setup
    auto& store = ColumnStoreGPU::instance();
    auto* targetCol = store.stageFloatColumn("__target", targetData);
    if (!targetCol) return {0.0, 0.0, 0.0, 0};
    
    std::vector<GPUColumn*> predColumns;
    for (const auto& col : predicateColumns) {
        auto it = lineitem_schema.find(col);
        if (it == lineitem_schema.end()) continue;
        
        std::vector<float> colData;
        if (col == "l_shipdate" || col == "l_commitdate" || col == "l_receiptdate") {
            colData = loadDateColumnAsFloat(filePath, it->second);
        } else {
            colData = loadFloatColumn(filePath, it->second);
        }
        auto* gpuCol = store.stageFloatColumn("__pred" + std::to_string(predColumns.size()), colData);
        if (gpuCol) predColumns.push_back(gpuCol);
    }
    
    // Build predicate buffer
    std::vector<PredicateClause> packed;
    std::map<std::string, uint32_t> colMap;
    colMap["__target"] = 0;
    for (size_t i = 0; i < predicateColumns.size(); ++i) {
        colMap[predicateColumns[i]] = static_cast<uint32_t>(i + 1);
    }
    
    for (const auto& cl : clauses) {
        PredicateClause pc;
        pc.colIndex = colMap[cl.ident];
        pc.op = static_cast<uint32_t>(cl.op);
        pc.isDate = cl.isDate ? 1u : 0u;
        
        if (pc.isDate) {
            pc.value = cl.date;
        } else {
            union { uint32_t u; float f; } conv;
            conv.f = static_cast<float>(cl.num);
            pc.value = conv.u;
        }
        packed.push_back(pc);
    }
    
    uint32_t colCount = static_cast<uint32_t>(predColumns.size() + 1);
    uint32_t clauseCount = static_cast<uint32_t>(packed.size());
    
    auto uploadEnd = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    
    // Create buffers
    MTL::Buffer* predicateBuffer = packed.empty() 
        ? store.device()->newBuffer(sizeof(PredicateClause), MTL::ResourceStorageModeShared)
        : store.device()->newBuffer(packed.data(), packed.size() * sizeof(PredicateClause), MTL::ResourceStorageModeShared);
    
    MTL::Buffer* outResultBuffer = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* outCountBuffer = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    
    // Initialize output buffers
    if (aggType == 3) {
        // MIN: initialize to max float
        union { uint32_t u; float f; } conv;
        conv.f = FLT_MAX;
        *reinterpret_cast<uint32_t*>(outResultBuffer->contents()) = conv.u;
    } else if (aggType == 4) {
        // MAX: initialize to min float
        union { uint32_t u; float f; } conv;
        conv.f = -FLT_MAX;
        *reinterpret_cast<uint32_t*>(outResultBuffer->contents()) = conv.u;
    } else {
        *reinterpret_cast<uint32_t*>(outResultBuffer->contents()) = 0;
    }
    *reinterpret_cast<uint32_t*>(outCountBuffer->contents()) = 0;
    
    // Get kernel
    static MTL::ComputePipelineState* pipeline = nullptr;
    if (!pipeline) {
        NS::Error* error = nullptr;
        auto fnName = NS::String::string("ops::scan_filter_aggregate", NS::UTF8StringEncoding);
        MTL::Function* fn = store.library()->newFunction(fnName);
        if (!fn) {
            std::cerr << "[GPU] Kernel ops::scan_filter_aggregate not found" << std::endl;
            fnName->release();
            predicateBuffer->release();
            outResultBuffer->release();
            outCountBuffer->release();
            return {0.0, 0.0, upload_ms, 0};
        }
        pipeline = store.device()->newComputePipelineState(fn, &error);
        fn->release();
        fnName->release();
        if (!pipeline) {
            if (error) std::cerr << "[GPU] Pipeline error: " << error->localizedDescription()->utf8String() << std::endl;
            predicateBuffer->release();
            outResultBuffer->release();
            outCountBuffer->release();
            return {0.0, 0.0, upload_ms, 0};
        }
    }
    
    // Execute
    auto kernelStart = std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cmd = store.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    
    // Bind column buffers
    enc->setBuffer(targetCol->buffer, 0, 0);
    for (size_t i = 0; i < predColumns.size(); ++i) {
        enc->setBuffer(predColumns[i]->buffer, 0, static_cast<NS::UInteger>(i + 1));
    }
    
    // Bind parameters
    enc->setBuffer(predicateBuffer, 0, 8);
    enc->setBytes(&colCount, sizeof(colCount), 9);
    enc->setBytes(&clauseCount, sizeof(clauseCount), 10);
    enc->setBytes(&rowCount, sizeof(rowCount), 11);
    enc->setBytes(&aggType, sizeof(aggType), 12);
    enc->setBuffer(outResultBuffer, 0, 13);
    enc->setBuffer(outCountBuffer, 0, 14);
    
    NS::UInteger maxTG = pipeline->maxTotalThreadsPerThreadgroup();
    if (maxTG > rowCount) maxTG = rowCount;
    MTL::Size gridSize = MTL::Size::Make(rowCount, 1, 1);
    MTL::Size tgSize = MTL::Size::Make(maxTG, 1, 1);
    enc->dispatchThreads(gridSize, tgSize);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    double gpu_ms = (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    if (gpu_ms <= 0.0) gpu_ms = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    
    // Read results
    uint32_t resultBits = *reinterpret_cast<uint32_t*>(outResultBuffer->contents());
    uint32_t count = *reinterpret_cast<uint32_t*>(outCountBuffer->contents());
    
    union { uint32_t u; float f; } conv;
    conv.u = resultBits;
    double result = static_cast<double>(conv.f);
    
    // For AVG, divide sum by count
    if (aggType == 2 && count > 0) {
        result = result / count;
    }
    
    // Release buffers
    predicateBuffer->release();
    outResultBuffer->release();
    outCountBuffer->release();
    
    return {result, gpu_ms, upload_ms, count};
}

} // namespace engine
