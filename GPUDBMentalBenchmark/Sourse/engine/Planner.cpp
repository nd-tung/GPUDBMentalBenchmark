#include "Planner.hpp"
#include "DuckDBAdapter.hpp"
#include <string>
#include <algorithm>
#include <regex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>

using nlohmann::json;

namespace engine {

static std::string tolower_copy(std::string s){ std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::tolower(c);}); return s; }

static void traverse(const json& node, Plan& p) {
    if (node.is_object()) {
        std::string name = node.contains("name") && node["name"].is_string() ? node["name"].get_string() : "";
        // extra_info can be string or object
        std::string extra;
        if (node.contains("extra_info")) {
            if (node["extra_info"].is_string()) {
                extra = node["extra_info"].get_string();
            }
        }
        if (name == "SEQ_SCAN" || name == "SEQ_SCAN " || name == "GET" || name == "TABLE_SCAN") {
            IRNode s; s.type = IRNode::Type::Scan;
            std::string t;
            // Try to extract table name from extra_info object
            if (node.contains("extra_info") && node["extra_info"].is_object()) {
                if (node["extra_info"].contains("Table") && node["extra_info"]["Table"].is_string()) {
                    t = node["extra_info"]["Table"].get_string();
                }
            }
            if (t.empty()) {
                t = extra;
                auto pos = t.find('['); if (pos != std::string::npos) t = t.substr(0,pos);
                auto sp = t.find('\n'); if (sp != std::string::npos) t = t.substr(0,sp);
            }
            if (t.empty()) t = "lineitem";
            s.scan.table = t;
            p.nodes.push_back(s);
        } else if (name == "FILTER") {
            IRNode f; f.type = IRNode::Type::Filter; f.filter.predicate = extra; p.nodes.push_back(f);
        } else if (name == "UNGROUPED_AGGREGATE" || name == "AGGREGATE") {
            IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = "sum"; std::string e = extra;
            auto s = e.find("sum("); if (s != std::string::npos) { auto r = e.find(')', s+4); if (r != std::string::npos) e = e.substr(s+4, r-(s+4)); }
            a.aggregate.expr = e; p.nodes.push_back(a);
        } else if (name == "HASH_GROUP_BY") {
            // Parse GROUP BY node
            IRNode g; g.type = IRNode::Type::GroupBy;
            
            // Extract group by keys from extra_info
            if (node.contains("extra_info") && node["extra_info"].is_object()) {
                const auto& ei = node["extra_info"];
                if (ei.contains("Groups") && ei["Groups"].is_string()) {
                    std::string groups = ei["Groups"].get_string();
                    // Parse group columns (e.g., "#0" or "l_orderkey")
                    g.groupBy.keys.push_back(groups);
                }
            }
            
            // Extract aggregate expressions from extra field
            if (!extra.empty()) {
                g.groupBy.aggs.push_back(extra);
            }
            
            p.nodes.push_back(g);
        } else if (name == "ORDER_BY" || name == "ORDER") {
            IRNode o; o.type = IRNode::Type::OrderBy;
            // Parse ORDER BY columns from extra (simplified)
            o.orderBy.ascending.push_back(true);  // Default ASC
            p.nodes.push_back(o);
        } else if (name == "LIMIT") {
            IRNode l; l.type = IRNode::Type::Limit;
            // Extract limit count from extra
            std::regex re("(\\d+)");
            std::smatch m;
            if (std::regex_search(extra, m, re)) {
                l.limit.count = std::stoll(m[1].str());
            } else {
                l.limit.count = 10;  // default
            }
            p.nodes.push_back(l);
        } else if (name == "PROJECTION") {
            IRNode pr; pr.type = IRNode::Type::Project;
            // Parse projection list (simplified)
            p.nodes.push_back(pr);
        } else if (name.find("JOIN") != std::string::npos) {
            IRNode j; j.type = IRNode::Type::Join;
            j.join.joinType = "inner";  // default
            
            // Extract join info from extra_info object
            if (node.contains("extra_info") && node["extra_info"].is_object()) {
                const auto& ei = node["extra_info"];
                if (ei.contains("Join Type") && ei["Join Type"].is_string()) {
                    std::string jtype = ei["Join Type"].get_string();
                    j.join.joinType = tolower_copy(jtype);
                }
                if (ei.contains("Conditions") && ei["Conditions"].is_string()) {
                    j.join.condition = ei["Conditions"].get_string();
                }
            }
            
            // Extract right table from children (second child is right table)
            if (node.contains("children") && node["children"].is_array() && node["children"].size() >= 2) {
                const auto& rightChild = node["children"][1];
                if (rightChild.contains("extra_info") && rightChild["extra_info"].is_object()) {
                    if (rightChild["extra_info"].contains("Table") && rightChild["extra_info"]["Table"].is_string()) {
                        j.join.rightTable = rightChild["extra_info"]["Table"].get_string();
                    }
                }
            }
            
            p.nodes.push_back(j);
        }
        if (node.contains("children")) {
            const auto& ch = node["children"]; if (ch.is_array()) {
                for (std::size_t i=0;i<ch.size();++i) traverse(ch[i], p);
            }
        }
    } else if (node.is_array()) {
        for (std::size_t i=0;i<node.size(); ++i) traverse(node[i], p);
    }
}

Plan Planner::fromSQL(const std::string& sql) {
    Plan p;
    std::string raw = DuckDBAdapter::explainJSON(sql);
    
    // Extract JSON array from DuckDB output (skip header, find balanced brackets)
    std::string jsonStr;
    auto start = raw.find('[');
    if (start != std::string::npos) {
        // Find matching closing bracket
        int depth = 0;
        size_t end = start;
        for (size_t i = start; i < raw.size(); i++) {
            if (raw[i] == '[') depth++;
            else if (raw[i] == ']') {
                depth--;
                if (depth == 0) {
                    end = i + 1;
                    break;
                }
            }
        }
        jsonStr = raw.substr(start, end - start);
        // Remove any trailing junk (e.g., shell prompt '%')
        while (!jsonStr.empty() && (jsonStr.back() == '%' || jsonStr.back() == '\n' || jsonStr.back() == '\r')) {
            jsonStr.pop_back();
        }
    } else {
        jsonStr = raw;
    }
    
    // Try to parse DuckDB JSON (bug in nlohmann::json has been fixed)
    bool ok = false;
    try {
        json j = json::parse(jsonStr);
        if (j.is_array() && j.size() > 0) {
            traverse(j[0], p);
            ok = !p.nodes.empty(); // Success if we got any nodes
        }
    } catch (...) {
        // Fall back to regex parser if JSON parsing fails
        ok = false;
    }
    
    if (!ok) {
        std::string low = tolower_copy(sql);
        std::string table; std::string predicate; std::string aggexpr;
        
        // Check for JOIN
        std::regex re_join(R"(from\s+([A-Za-z_][A-Za-z0-9_]*)\s+join\s+([A-Za-z_][A-Za-z0-9_]*)\s+on\s+([^\s]+)\s*=\s*([^\s]+))", std::regex::icase);
        std::smatch m_join;
        if (std::regex_search(sql, m_join, re_join) && m_join.size()>4) {
            // Found JOIN
            table = m_join[1].str();
            std::string rightTable = m_join[2].str();
            
            IRNode s; s.type = IRNode::Type::Scan; s.scan.table = table; p.nodes.push_back(s);
            IRNode s2; s2.type = IRNode::Type::Scan; s2.scan.table = rightTable; p.nodes.push_back(s2);
            IRNode j; j.type = IRNode::Type::Join;
            j.join.rightTable = rightTable;
            j.join.condition = m_join[3].str() + "=" + m_join[4].str();
            j.join.joinType = "inner";
            p.nodes.push_back(j);
        } else {
            // Simple query
            std::regex re_from(R"(from\s+([A-Za-z_][A-Za-z0-9_\.]*)\b)", std::regex::icase);
            std::smatch m;
            if (std::regex_search(sql, m, re_from) && m.size()>1) table = m[1].str();
            if (table.empty()) table = "lineitem";
            IRNode s; s.type = IRNode::Type::Scan; s.scan.table = table; p.nodes.push_back(s);
        }
        
        // Parse aggregation function: SUM, COUNT, AVG, MIN, MAX
        std::string aggFunc;
        std::regex re_agg(R"(select\s+(sum|count|avg|min|max)\s*\()", std::regex::icase);
        std::smatch m;
        if (std::regex_search(sql, m, re_agg)) {
            aggFunc = m[1].str();
            std::transform(aggFunc.begin(), aggFunc.end(), aggFunc.begin(), ::tolower);
            
            // Find matching closing parenthesis
            size_t start = m.position() + m.length();
            int depth = 1;
            size_t end = start;
            for (size_t i = start; i < sql.size() && depth > 0; ++i) {
                if (sql[i] == '(') depth++;
                else if (sql[i] == ')') {
                    depth--;
                    if (depth == 0) {
                        end = i;
                        break;
                    }
                }
            }
            aggexpr = sql.substr(start, end - start);
        }
        std::regex re_where(R"(where\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+limit|$))", std::regex::icase);
        if (std::regex_search(sql, m, re_where) && m.size()>1) predicate = m[1].str();
        
        // Parse ORDER BY
        std::regex re_orderby(R"(order\s+by\s+([A-Za-z_][A-Za-z0-9_]*)\s*(asc|desc)?)", std::regex::icase);
        if (std::regex_search(sql, m, re_orderby) && m.size()>1) {
            IRNode o; o.type = IRNode::Type::OrderBy;
            o.orderBy.columns.push_back(m[1].str());
            bool isAsc = true;
            if (m.size() > 2 && m[2].matched) {
                std::string dir = m[2].str();
                std::transform(dir.begin(), dir.end(), dir.begin(), ::tolower);
                isAsc = (dir != "desc");
            }
            o.orderBy.ascending.push_back(isAsc);
            p.nodes.push_back(o);
        }
        
        // Parse GROUP BY
        std::regex re_groupby(R"(group\s+by\s+([A-Za-z_][A-Za-z0-9_,\s]*?)(?:\s+order\s+by|\s+limit|$))", std::regex::icase);
        if (std::regex_search(sql, m, re_groupby) && m.size()>1) {
            IRNode g; g.type = IRNode::Type::GroupBy;
            std::string groupCols = m[1].str();
            // Split by comma for multiple columns
            size_t start = 0;
            while (start < groupCols.size()) {
                size_t comma = groupCols.find(',', start);
                if (comma == std::string::npos) comma = groupCols.size();
                std::string col = groupCols.substr(start, comma - start);
                // Trim spaces
                col.erase(0, col.find_first_not_of(" \t\n\r"));
                col.erase(col.find_last_not_of(" \t\n\r") + 1);
                if (!col.empty()) g.groupBy.keys.push_back(col);
                start = comma + 1;
            }
            if (!aggexpr.empty()) g.groupBy.aggs.push_back(aggexpr);
            p.nodes.push_back(g);
        }
        
        // Parse LIMIT
        std::regex re_limit(R"(limit\s+(\d+))", std::regex::icase);
        if (std::regex_search(sql, m, re_limit) && m.size()>1) {
            IRNode l; l.type = IRNode::Type::Limit;
            l.limit.count = std::stoll(m[1].str());
            l.limit.offset = 0;
            p.nodes.push_back(l);
        }
        
        // Only add aggregate if aggregation function is present AND no GROUP BY
        if (!aggexpr.empty() && !aggFunc.empty()) {
            bool hasGroupBy = false;
            for (auto& n : p.nodes) if (n.type == IRNode::Type::GroupBy) hasGroupBy = true;
            
            if (!hasGroupBy) {
                if (!predicate.empty()) { IRNode f; f.type = IRNode::Type::Filter; f.filter.predicate = predicate; p.nodes.push_back(f); }
                IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = aggFunc; a.aggregate.expr = aggexpr;
                // Check if expression contains arithmetic operators
                a.aggregate.hasExpression = (aggexpr.find('*') != std::string::npos || 
                                             aggexpr.find('/') != std::string::npos ||
                                             aggexpr.find('+') != std::string::npos ||
                                             aggexpr.find('-') != std::string::npos);
                p.nodes.push_back(a);
            }
        }
    }
    // Ensure order Scan -> Join/Filter -> GroupBy -> OrderBy -> Limit -> Aggregate
    std::vector<IRNode> ordered;
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Scan) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Join) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Filter) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::GroupBy) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::OrderBy) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Limit) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Aggregate) ordered.push_back(n);
    p.nodes.swap(ordered);
    return p;
}

} // namespace engine
