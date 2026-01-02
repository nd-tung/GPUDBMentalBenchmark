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
#include "SortExecutor.hpp"
#include "GroupByExecutor.hpp"
#include "ExprEval.hpp"

static std::string g_dataset_path = "GPUDBMetalBenchmark/Data/SF-1/";

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
    
    // Check for GROUP BY
    bool hasGroupBy = false;
    std::vector<std::string> groupByCols;
    std::vector<std::string> aggCols;
    std::vector<std::string> aggFuncs;
    
    for (const auto& n : plan.nodes) {
        if (n.type == IRNode::Type::GroupBy) {
            hasGroupBy = true;
            groupByCols = n.groupBy.keys;
            aggFuncs = n.groupBy.aggFuncs;
            
            // If aggFuncs is empty, default to "sum"
            if (aggFuncs.empty() && !n.groupBy.aggs.empty()) {
                aggFuncs.resize(n.groupBy.aggs.size(), "sum");
            }
            
            // Extract column names from aggregate expressions
            for (const auto& agg : n.groupBy.aggs) {
                // Parse out the column name from AGG(column)
                auto start = agg.find('(');
                auto end = agg.rfind(')');
                if (start != std::string::npos && end != std::string::npos && end > start) {
                    std::string col = agg.substr(start + 1, end - start - 1);
                    // Trim spaces
                    col.erase(0, col.find_first_not_of(" \t\n\r"));
                    col.erase(col.find_last_not_of(" \t\n\r") + 1);
                    aggCols.push_back(col);
                } else {
                    aggCols.push_back(agg);
                }
            }
            break;
        }
    }
    
    if (want_gpu && hasGroupBy && !groupByCols.empty() && !aggCols.empty()) {
        // Extract table name
        std::string table = "lineitem";
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Scan) {
                table = n.scan.table;
                break;
            }
        }
        
        // Run GPU GROUP BY
        auto groupByRes = engine::GroupByExecutor::runGroupBy(g_dataset_path, table, groupByCols, aggCols, aggFuncs);
        
        if (!groupByRes.groups.empty()) {
            std::cout << "Result:" << std::endl;
            std::cout << "GROUP BY ";
            for (size_t i = 0; i < groupByCols.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << groupByCols[i];
            }
            std::cout << " with ";
            for (size_t i = 0; i < aggCols.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << (i < aggFuncs.size() ? aggFuncs[i] : "SUM") << "(" << aggCols[i] << ")";
            }
            std::cout << std::endl;
            printf("Upload time: %0.2f ms\n", groupByRes.upload_ms);
            printf("GPU kernel time: %0.2f ms\n", groupByRes.gpu_ms);
            printf("Number of groups: %zu\n", groupByRes.groups.size());
            
            // Show results (limit to first 10 for display)
            size_t count = 0;
            for (const auto& [keys, sums] : groupByRes.groups) {
                // Print key (composite key shown as tuple)
                if (keys.size() == 1) {
                    std::cout << groupByCols[0] << "=" << keys[0];
                } else {
                    std::cout << "(";
                    for (size_t i = 0; i < keys.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << (i < groupByCols.size() ? groupByCols[i] : "key") << "=" << keys[i];
                    }
                    std::cout << ")";
                }
                // Print aggregate value(s)
                std::cout << " -> ";
                if (sums.size() == 1) {
                    std::cout << (aggFuncs.empty() ? "AGG" : aggFuncs[0]) << "=" << sums[0];
                } else {
                    std::cout << "(";
                    for (size_t i = 0; i < sums.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << (i < aggFuncs.size() ? aggFuncs[i] : "AGG") << "=" << sums[i];
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
                if (++count >= 10) {
                    if (groupByRes.groups.size() > 10) {
                        std::cout << "... (" << (groupByRes.groups.size() - 10) << " more groups)" << std::endl;
                    }
                    break;
                }
            }
            return;
        }
    }
    
    // Check for ORDER BY (without aggregation for now - just sorting and display)
    bool hasOrderBy = false;
    std::string orderByCol;
    bool orderAscending = true;
    for (const auto& n : plan.nodes) {
        if (n.type == IRNode::Type::OrderBy) {
            hasOrderBy = true;
            if (!n.orderBy.columns.empty()) {
                orderByCol = n.orderBy.columns[0];
                if (!n.orderBy.ascending.empty()) {
                    orderAscending = n.orderBy.ascending[0];
                }
            }
            break;
        }
    }
    
    if (want_gpu && hasOrderBy && !orderByCol.empty()) {
        // Extract table name and LIMIT
        std::string table = "lineitem";
        int64_t limitCount = -1;
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Scan) {
                table = n.scan.table;
            } else if (n.type == IRNode::Type::Limit) {
                limitCount = n.limit.count;
            }
        }
        
        // Run GPU sort
        auto sortRes = engine::SortExecutor::runSort(g_dataset_path, table, orderByCol, orderAscending);
        
        if (!sortRes.indices.empty()) {
            // Apply LIMIT if specified
            size_t displayCount = sortRes.indices.size();
            if (limitCount > 0 && static_cast<size_t>(limitCount) < displayCount) {
                displayCount = static_cast<size_t>(limitCount);
            }
            
            std::cout << "Result:" << std::endl;
            std::cout << "Sorted by " << orderByCol << " (" << (orderAscending ? "ASC" : "DESC") << ")";
            if (limitCount > 0) std::cout << " LIMIT " << limitCount;
            std::cout << std::endl;
            printf("Upload time: %0.2f ms\n", sortRes.upload_ms);
            printf("GPU kernel time: %0.2f ms\n", sortRes.gpu_ms);
            printf("Total rows sorted: %zu\n", sortRes.indices.size());
            printf("Rows returned: %zu\n", displayCount);
            
            // Show indices
            std::cout << "Sorted indices: ";
            for (size_t i = 0; i < displayCount; ++i) {
                std::cout << sortRes.indices[i];
                if (i < displayCount - 1) std::cout << ", ";
            }
            std::cout << std::endl;
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
    // Check if aggexpr has arithmetic operators (if so, use generic path with expressions)
    bool hasArithmeticOps = (lowAgg.find('*') != std::string::npos || 
                             lowAgg.find('/') != std::string::npos ||
                             lowAgg.find('+') != std::string::npos ||
                             lowAgg.find('-') != std::string::npos);
    if (table=="lineitem" &&
        !hasArithmeticOps &&  // Disable Q6 if expressions present
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
        std::string aggFunc; std::string aggExpr; bool hasExpression = false;
        for (auto& n: plan.nodes) if(n.type==engine::IRNode::Type::Aggregate){ 
            aggFunc=n.aggregate.func; aggExpr=n.aggregate.expr; hasExpression=n.aggregate.hasExpression;
        }
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
        
        // Check if GPU can handle this query
        bool gpuEligible = false;
        std::string lowerAggFunc = aggFunc;
        std::transform(lowerAggFunc.begin(), lowerAggFunc.end(), lowerAggFunc.begin(), ::tolower);
        
        if (want_gpu) {
            if (lowerAggFunc == "sum" && hasExpression) {
                // Expression path (e.g., l_extendedprice * (1 - l_discount))
                gpuEligible = true;
            } else if (!targetIdent.empty() && (lowerAggFunc == "sum" || lowerAggFunc == "count" || 
                       lowerAggFunc == "avg" || lowerAggFunc == "min" || lowerAggFunc == "max")) {
                // Simple column aggregation
                gpuEligible = engine::GpuExecutor::isEligible(lowerAggFunc, clauses, targetIdent);
            }
        }
        
        if (gpuEligible) {
            engine::GPUResult gpuRes;
            if (lowerAggFunc == "sum" && hasExpression) {
                gpuRes = engine::GpuExecutor::runSumWithExpression(g_dataset_path, aggExpr, clauses);
            } else if (lowerAggFunc == "sum") {
                gpuRes = engine::GpuExecutor::runSum(g_dataset_path, targetIdent, clauses);
            } else {
                gpuRes = engine::GpuExecutor::runAggregate(g_dataset_path, lowerAggFunc, targetIdent, clauses);
            }
            
            std::cout << "Result:" << std::endl;
            std::string aggLabel = lowerAggFunc;
            std::transform(aggLabel.begin(), aggLabel.end(), aggLabel.begin(), ::toupper);
            std::cout << "Scalar " << aggLabel << ": " << std::fixed << std::setprecision(2) << gpuRes.revenue << std::endl;
            if (lowerAggFunc == "count" || lowerAggFunc == "avg") {
                std::cout << "Row count: " << gpuRes.count << std::endl;
            }
            printf("Upload time: %0.2f ms\n", gpuRes.upload_ms);
            printf("GPU kernel time: %0.2f ms\n", gpuRes.gpu_ms);
        } else {
            auto result = engine::Executor::runGeneric(plan, g_dataset_path);
            std::cout << "Result:" << std::endl;
            std::string aggLabel = lowerAggFunc;
            std::transform(aggLabel.begin(), aggLabel.end(), aggLabel.begin(), ::toupper);
            std::cout << "Scalar " << aggLabel << ": " << std::fixed << std::setprecision(2) << result.revenue << std::endl;
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

    // Args: sf1|sf10 and optional --sql "..." or just SQL as first arg
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "sf1") g_dataset_path = "GPUDBMetalBenchmark/Data/SF-1/";
        else if (arg == "sf10") g_dataset_path = "GPUDBMetalBenchmark/Data/SF-10/";
        else if (arg == "--sql" && i+1 < argc) { sql = argv[++i]; }
        else if (arg == "help" || arg == "--help" || arg == "-h") {
            std::cout << "GPUDBEngineHost" << std::endl;
            std::cout << "Usage: GPUDBEngineHost [sf1|sf10] [--sql 'QUERY' | 'QUERY']" << std::endl;
            return 0;
        }
        else if (i == 1 && arg.find("SELECT") != std::string::npos) {
            // First arg is a SQL query if it contains SELECT
            sql = arg;
        }
    }
    runEngineSQL(sql);
    return 0;
}
