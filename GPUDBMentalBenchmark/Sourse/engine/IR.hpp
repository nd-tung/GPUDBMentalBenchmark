#pragma once
#include <string>
#include <vector>

namespace engine {

struct IRScan { std::string table; };
struct IRFilter { std::string predicate; };
struct IRAggregate { 
    std::string func;       // e.g., "sum", "count", "avg"
    std::string expr;       // expression to aggregate (raw string)
    bool hasExpression = false;  // true if expr contains arithmetic operations
};
struct IRProject {
    std::vector<std::string> columns;  // SELECT list columns
};
struct IROrderBy {
    std::vector<std::string> columns;  // ORDER BY columns
    std::vector<bool> ascending;        // true for ASC, false for DESC
};
struct IRLimit {
    int64_t count;         // max rows to return
    int64_t offset = 0;    // rows to skip
};
struct IRGroupBy {
    std::vector<std::string> keys;     // GROUP BY columns
    std::vector<std::string> aggs;     // Aggregate expressions
};
struct IRJoin {
    std::string rightTable;            // Right side table name
    std::string condition;             // Join condition
    std::string joinType;              // "inner", "left", "right"
};

struct IRNode {
    enum class Type { Scan, Filter, Aggregate, Project, OrderBy, Limit, GroupBy, Join } type;
    IRScan scan;
    IRFilter filter;
    IRAggregate aggregate;
    IRProject project;
    IROrderBy orderBy;
    IRLimit limit;
    IRGroupBy groupBy;
    IRJoin join;
};

struct Plan {
    std::vector<IRNode> nodes; // linear pipeline order: Scan -> Filter -> ... -> Limit
};

} // namespace engine
