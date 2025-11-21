#pragma once
#include <string>
#include <vector>

namespace engine {

struct IRScan { std::string table; };
struct IRFilter { std::string predicate; };
struct IRAggregate { std::string func; std::string expr; };

struct IRNode {
    enum class Type { Scan, Filter, Aggregate } type;
    IRScan scan;
    IRFilter filter;
    IRAggregate aggregate;
};

struct Plan {
    std::vector<IRNode> nodes; // linear pipeline order: Scan -> Filter -> Aggregate
};

} // namespace engine
