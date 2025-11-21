#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "Operators.hpp"
#include "IR.hpp"
#include "Planner.hpp"
#include "PipelineBuilder.hpp"
#include "Executor.hpp"
#include "GpuExecutor.hpp"
#include "JoinExecutor.hpp"
#include "ExprEval.hpp"

static std::string g_dataset_path = "Data/SF-1/";

static std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
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

static std::vector<int> loadDateColumn(const std::string& filePath, int columnIndex) {
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

static void runEngineSQL(const std::string& sql) {
    using namespace engine;
    std::cout << "--- Running (Engine Host) ---" << std::endl;
    Plan plan = Planner::fromSQL(sql);
    
    // Check for JOIN queries
    bool hasJoin = false;
    std::string leftTable, rightTable;
    for (const auto& n : plan.nodes) {
        if (n.type == IRNode::Type::Join) {
            hasJoin = true;
            rightTable = n.join.rightTable;
        }
        if (n.type == IRNode::Type::Scan) {
            if (leftTable.empty()) leftTable = n.scan.table;
        }
    }
    
    bool want_gpu = (std::getenv("GPUDB_USE_GPU")!=nullptr);
    
    // Handle JOIN queries
    if (hasJoin && want_gpu) {
        // Extract join key columns and aggregation target
        std::string leftKey = "l_orderkey";   // Default for lineitem-orders join
        std::string rightKey = "o_orderkey";
        std::string aggCol = "l_extendedprice";
        
        // Parse from plan if available
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Aggregate) {
                aggCol = n.aggregate.expr;
            }
        }
        
        if (JoinExecutor::isEligible(leftTable, rightTable)) {
            auto joinRes = JoinExecutor::runHashJoin(g_dataset_path, leftTable, rightTable, 
                                                     leftKey, rightKey, aggCol);
            std::cout << "Result:" << std::endl;
            std::cout << "JOIN SUM (GPU): " << std::fixed << std::setprecision(2) << joinRes.revenue << std::endl;
            printf("Matched rows: %u\n", joinRes.match_count);
            printf("Upload time: %0.2f ms\n", joinRes.upload_ms);
            printf("GPU kernel time: %0.2f ms\n", joinRes.gpu_ms);
            return;
        }
    }
    
    // Non-join path
    bool use_q6 = false;
    std::string pred; std::string aggexpr; std::string table="lineitem";
    for (const auto& n : plan.nodes) {
        if (n.type==IRNode::Type::Scan) table = n.scan.table;
        else if (n.type==IRNode::Type::Filter) pred = n.filter.predicate;
        else if (n.type==IRNode::Type::Aggregate) aggexpr = n.aggregate.expr;
    }
    std::string lowPred = pred; std::transform(lowPred.begin(), lowPred.end(), lowPred.begin(), ::tolower);
    std::string lowAgg = aggexpr; std::transform(lowAgg.begin(), lowAgg.end(), lowAgg.begin(), ::tolower);
    if (table=="lineitem" &&
        lowPred.find("shipdate")!=std::string::npos &&
        lowPred.find("discount")!=std::string::npos &&
        lowPred.find("quantity")!=std::string::npos &&
        lowAgg.find("extendedprice")!=std::string::npos &&
        lowAgg.find("discount")!=std::string::npos) {
        use_q6 = true;
    }
    if (use_q6) {
        auto spec = PipelineBuilder::buildQ6(plan);
        auto result = Executor::runQ6(spec, g_dataset_path);
        std::cout << "Result:" << std::endl;
        std::cout << "Total Revenue: $" << std::fixed << std::setprecision(2) << result.revenue << std::endl;
        printf("Total TPC-H Q6 GPU time: %0.2f ms\n", 0.0);
        printf("Q6 CPU time: %0.2f ms\n", result.cpu_ms);
        printf("Total TPC-H Q6 wall-clock: %0.2f ms\n", result.cpu_ms);
    } else {
        // Extract aggregate info
        std::string aggFunc; std::string aggExpr; for (auto& n: plan.nodes) if(n.type==engine::IRNode::Type::Aggregate){ aggFunc=n.aggregate.func; aggExpr=n.aggregate.expr; }
        // Very simple target ident (fast path) if single ident RPN
        std::string targetIdent;
        {
            auto tokens = engine::expr::tokenize_arith(aggExpr);
            auto rpn = engine::expr::to_rpn(tokens);
            if (rpn.size()==1 && rpn[0].type==engine::expr::Token::Type::Ident) targetIdent = rpn[0].text;
        }
        // Parsed predicate clauses (CPU version reused)
        std::string predicate; for (auto& n: plan.nodes) if(n.type==engine::IRNode::Type::Filter) predicate=n.filter.predicate;
        auto exists = [](const std::string&){ return true; };
        auto clauses = engine::expr::parse_predicate(predicate, exists);
        bool gpuEligible = want_gpu && !targetIdent.empty() && engine::GpuExecutor::isEligible(aggFunc, clauses, targetIdent);
        if (gpuEligible) {
            auto gpuRes = engine::GpuExecutor::runSum(g_dataset_path, targetIdent, clauses);
            std::cout << "Result:" << std::endl;
            std::cout << "Scalar SUM (GPU): " << std::fixed << std::setprecision(2) << gpuRes.revenue << std::endl;
            printf("Upload time: %0.2f ms\n", gpuRes.upload_ms);
            printf("GPU kernel time: %0.2f ms\n", gpuRes.gpu_ms);
            // CPU reference for validation
            auto cpuRes = engine::Executor::runGeneric(plan, g_dataset_path);
            std::cout << "Scalar SUM (CPU): " << std::fixed << std::setprecision(2) << cpuRes.revenue << std::endl;
            printf("CPU reference time: %0.2f ms\n", cpuRes.cpu_ms);
            double diff = std::abs(cpuRes.revenue - gpuRes.revenue);
            double rel = diff / (std::abs(cpuRes.revenue)+1e-9);
            printf("Rel diff: %.6f\n", rel);
        } else {
            auto result = engine::Executor::runGeneric(plan, g_dataset_path);
            std::cout << "Result:" << std::endl;
            std::cout << "Scalar SUM: " << std::fixed << std::setprecision(2) << result.revenue << std::endl;
            printf("CPU time: %0.2f ms\n", result.cpu_ms);
        }
    }
}

int main(int argc, const char* argv[]) {
    std::string sql =
        "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue\n"
        "FROM lineitem\n"
        "WHERE l_shipdate >= DATE '1994-01-01'\n"
        "  AND l_shipdate <  DATE '1995-01-01'\n"
        "  AND l_discount >= 0.05 AND l_discount <= 0.07\n"
        "  AND l_quantity < 24";

    // Args: sf1|sf10 and optional --sql "..."
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "sf1") g_dataset_path = "Data/SF-1/";
        else if (arg == "sf10") g_dataset_path = "Data/SF-10/";
        else if (arg == "--sql" && i+1 < argc) { sql = argv[++i]; }
        else if (arg == "help" || arg == "--help" || arg == "-h") {
            std::cout << "GPUDBEngineHost" << std::endl;
            std::cout << "Usage: GPUDBEngineHost [sf1|sf10] [--sql 'QUERY']" << std::endl;
            return 0;
        }
    }
    runEngineSQL(sql);
    return 0;
}
