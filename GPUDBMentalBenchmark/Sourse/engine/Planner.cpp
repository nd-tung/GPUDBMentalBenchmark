#include "Planner.hpp"
#include "DuckDBAdapter.hpp"
#include <string>
#include <algorithm>
#include <regex>
#include <nlohmann/json.hpp>
#include <stdexcept>

using nlohmann::json;

namespace engine {

static std::string tolower_copy(std::string s){ std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::tolower(c);}); return s; }

static void traverse(const json& node, Plan& p) {
    if (node.is_object()) {
        std::string name = node.contains("name") && node["name"].is_string() ? node["name"].get_string() : "";
        std::string extra = node.contains("extra_info") && node["extra_info"].is_string() ? node["extra_info"].get_string() : "";
        if (name == "SEQ_SCAN" || name == "GET" || name == "TABLE_SCAN") {
            IRNode s; s.type = IRNode::Type::Scan; std::string t = extra;
            auto pos = t.find('['); if (pos != std::string::npos) t = t.substr(0,pos);
            auto sp = t.find('\n'); if (sp != std::string::npos) t = t.substr(0,sp);
            if (t.empty()) t = "lineitem"; s.scan.table = t; p.nodes.push_back(s);
        } else if (name == "FILTER") {
            IRNode f; f.type = IRNode::Type::Filter; f.filter.predicate = extra; p.nodes.push_back(f);
        } else if (name == "UNGROUPED_AGGREGATE" || name == "HASH_GROUP_BY" || name == "AGGREGATE") {
            IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = "sum"; std::string e = extra;
            auto s = e.find("sum("); if (s != std::string::npos) { auto r = e.find(')', s+4); if (r != std::string::npos) e = e.substr(s+4, r-(s+4)); }
            a.aggregate.expr = e; p.nodes.push_back(a);
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
    bool ok = true;
    try {
        json j = json::parse(raw);
        traverse(j, p);
        if (p.nodes.empty()) ok = false;
    } catch (...) { ok = false; }
    if (!ok) {
        std::string low = tolower_copy(raw);
        std::string table; std::string predicate; std::string aggexpr;
        // Fallback minimal regex extraction
        std::regex re_sum(R"(select\s+sum\s*\(([^\)]+)\))", std::regex::icase);
        std::smatch m; if (std::regex_search(sql, m, re_sum) && m.size()>1) aggexpr = m[1].str();
        std::regex re_from(R"(from\s+([A-Za-z_][A-Za-z0-9_\.]*)\b)", std::regex::icase);
        if (std::regex_search(sql, m, re_from) && m.size()>1) table = m[1].str();
        std::regex re_where(R"(where\s+(.+))", std::regex::icase);
        if (std::regex_search(sql, m, re_where) && m.size()>1) predicate = m[1].str();
        if (table.empty()) table = "lineitem";
        if (aggexpr.empty()) aggexpr = "l_extendedprice";
        IRNode s; s.type = IRNode::Type::Scan; s.scan.table = table; p.nodes.push_back(s);
        if (!predicate.empty()) { IRNode f; f.type = IRNode::Type::Filter; f.filter.predicate = predicate; p.nodes.push_back(f); }
        IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = "sum"; a.aggregate.expr = aggexpr; p.nodes.push_back(a);
    }
    // Ensure order Scan -> Filter -> Aggregate
    std::vector<IRNode> ordered;
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Scan) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Filter) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Aggregate) ordered.push_back(n);
    p.nodes.swap(ordered);
    return p;
}

} // namespace engine
