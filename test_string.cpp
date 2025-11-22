#include "engine/GpuExecutor.hpp"
#include "engine/Predicate.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <lineitem.tbl> <shipmode>" << std::endl;
        return 1;
    }
    
    // Extract directory path from file path (remove "/lineitem.tbl")
    std::string file_path = argv[1];
    std::string dataset_path = file_path.substr(0, file_path.rfind('/') + 1);
    std::string shipmode = argv[2];
    
    // Build predicate: l_shipmode = 'shipmode'
    std::string where_clause = "l_shipmode = '" + shipmode + "'";
    std::cout << "Query: SELECT COUNT(*) FROM lineitem WHERE " << where_clause << std::endl;
    std::cout << "Dataset path: " << dataset_path << std::endl;
    
    // Parse the predicate
    auto exists_fn = [](const std::string& col) { return true; };  // Accept all columns
    auto clauses = engine::expr::parse_predicate(where_clause, exists_fn);
    
    if (clauses.empty()) {
        std::cerr << "Failed to parse WHERE clause" << std::endl;
        return 1;
    }
    
    std::cout << "Parsed " << clauses.size() << " clause(s)" << std::endl;
    for (const auto& c : clauses) {
        std::cout << "  Clause: " << c.ident << " isString=" << c.isString << " strValue='" << c.strValue << "'" << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    engine::GpuExecutor exec;
    auto result = exec.runAggregate(dataset_path, "COUNT", "", clauses);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "COUNT(*) = " << static_cast<uint64_t>(result.count) << std::endl;
    std::cout << "GPU execution time: " << result.gpu_ms << " ms" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    
    return 0;
}
