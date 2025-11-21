#include "Executor.hpp"
#include "ExprEval.hpp"
#include "Predicate.hpp"
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <regex>

namespace engine {

static std::vector<int> loadDateColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data; std::ifstream file(filePath); if (!file.is_open()) { fprintf(stderr, "Failed to open %s\n", filePath.c_str()); return data; }
    std::string line; while (std::getline(file, line)) { std::string token; int col=0; size_t s=0,e=line.find('|');
        while (e!=std::string::npos) { if (col==columnIndex) { token=line.substr(s,e-s); token.erase(std::remove(token.begin(), token.end(), '-'), token.end()); data.push_back(std::stoi(token)); break; }
            s=e+1; e=line.find('|', s); ++col; }
    } return data;
}
static std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data; std::ifstream file(filePath); if (!file.is_open()) { fprintf(stderr, "Failed to open %s\n", filePath.c_str()); return data; }
    std::string line; while (std::getline(file, line)) { std::string token; int col=0; size_t s=0,e=line.find('|');
        while (e!=std::string::npos) { if (col==columnIndex) { token=line.substr(s,e-s); data.push_back(std::stof(token)); break; }
            s=e+1; e=line.find('|', s); ++col; }
    } return data;
}

struct ColumnStore {
    std::map<std::string, std::vector<float>> fcols;
    std::map<std::string, std::vector<int>> icols;
    std::size_t size = 0;
};

static bool load_lineitem_columns(const std::string& dataset_path,
                                  const std::set<std::string>& needed,
                                  ColumnStore& store) {
    std::string path = dataset_path + "lineitem.tbl";
    static const std::map<std::string, int> float_idx{{"l_quantity",4},{"l_extendedprice",5},{"l_discount",6}};
    static const std::map<std::string, int> date_idx{{"l_shipdate",10}};
    bool ok = true; std::size_t n=0; bool n_set=false;
    for (const auto& col : needed) {
        if (auto it=float_idx.find(col); it!=float_idx.end()) {
            auto v = loadFloatColumn(path, it->second); if (!n_set) { n=v.size(); n_set=true; } if (v.size()!=n) ok=false; store.fcols[col]=std::move(v);
        } else if (auto it2=date_idx.find(col); it2!=date_idx.end()) {
            auto v = loadDateColumn(path, it2->second); if (!n_set) { n=v.size(); n_set=true; } if (v.size()!=n) ok=false; store.icols[col]=std::move(v);
        } else {
            ok=false;
        }
    }
    store.size = n_set? n : 0; return ok;
}

Executor::Result Executor::runQ6(const PipelineSpecQ6& spec, const std::string& dataset_path) {
    // Load needed columns from lineitem
    auto l_shipdate = loadDateColumn(dataset_path + "lineitem.tbl", 10);
    auto l_discount = loadFloatColumn(dataset_path + "lineitem.tbl", 6);
    auto l_quantity = loadFloatColumn(dataset_path + "lineitem.tbl", 4);
    auto l_extendedprice = loadFloatColumn(dataset_path + "lineitem.tbl", 5);

    BufferView shipdate{l_shipdate.data(), l_shipdate.size(), sizeof(int)};
    BufferView discount{l_discount.data(), l_discount.size(), sizeof(float)};
    BufferView quantity{l_quantity.data(), l_quantity.size(), sizeof(float)};
    BufferView extendedprice{l_extendedprice.data(), l_extendedprice.size(), sizeof(float)};

    FilterQ6 op; KernelConfig cfg{"filter_q6_cpu", 0, 0}; op.init(cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    double revenue = op.computeRevenue(shipdate, discount, quantity, extendedprice, spec.params);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return {revenue, cpu_ms};
}

Executor::Result Executor::runGeneric(const Plan& plan, const std::string& dataset_path) {
    using namespace engine::expr;
    std::string table = "lineitem";
    std::string predicate;
    std::string aggfunc;
    std::string aggexpr;
    for (const auto& n : plan.nodes) {
        if (n.type==IRNode::Type::Scan) table = n.scan.table;
        else if (n.type==IRNode::Type::Filter) predicate = n.filter.predicate;
        else if (n.type==IRNode::Type::Aggregate) { aggfunc = n.aggregate.func; aggexpr = n.aggregate.expr; }
    }
    if (table != "lineitem") return {0.0, 0.0};

    std::set<std::string> cols = collect_idents(aggexpr);
    // Extract only LHS identifiers from predicate comparisons
    if (!predicate.empty()) {
        try {
            std::regex re_lhs(R"(([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:<=|>=|=|<|>))", std::regex::icase);
            auto it = std::sregex_iterator(predicate.begin(), predicate.end(), re_lhs);
            auto end = std::sregex_iterator();
            for (; it != end; ++it) {
                if (it->size()>1) cols.insert((*it)[1].str());
            }
        } catch (...) {
            // regex failure fallback: scan chars
            std::string tmp = predicate;
            for (std::size_t i=0;i+1<tmp.size();++i) {
                if (std::isalpha(static_cast<unsigned char>(tmp[i])) || tmp[i]=='_') {
                    std::size_t j=i+1; while (j<tmp.size() && (std::isalnum(static_cast<unsigned char>(tmp[j])) || tmp[j]=='_' || tmp[j]=='.')) ++j;
                    cols.insert(tmp.substr(i,j-i)); i=j;
                }
            }
        }
    }
    ColumnStore store;
    auto ok = load_lineitem_columns(dataset_path, cols, store);
    (void)ok;

    auto getExists = [&](const std::string& name)->bool{
        return store.fcols.count(name) || store.icols.count(name);
    };
    auto getInt = [&](std::size_t i, const std::string& name)->long long{
        if (auto it=store.icols.find(name); it!=store.icols.end()) return static_cast<long long>(it->second[i]);
        if (auto itf=store.fcols.find(name); itf!=store.fcols.end()) return static_cast<long long>(itf->second[i]);
        return 0;
    };
    auto getFloat = [&](std::size_t i, const std::string& name)->double{
        if (auto it=store.fcols.find(name); it!=store.fcols.end()) return static_cast<double>(it->second[i]);
        if (auto iti=store.icols.find(name); iti!=store.icols.end()) return static_cast<double>(iti->second[i]);
        return 0.0;
    };

    auto debug_env = std::getenv("GPUDB_DEBUG");
    auto rpn = to_rpn(tokenize_arith(aggexpr));
    if (debug_env) {
        fprintf(stderr, "[GenericExec] table=%s aggexpr=%s predicate='%s' cols_loaded=%zu\n", table.c_str(), aggexpr.c_str(), predicate.c_str(), cols.size());
        for (auto& c : cols) {
            bool f = store.fcols.count(c); bool i = store.icols.count(c);
            fprintf(stderr, "[GenericExec] col=%s f=%d i=%d size_f=%zu size_i=%zu\n", c.c_str(), f?1:0, i?1:0, f?store.fcols[c].size():0, i?store.icols[c].size():0);
        }
        fprintf(stderr, "[GenericExec] store.size=%zu\n", store.size);
    }

    // Pre-parse predicate clauses for faster evaluation.
    auto parsedClauses = parse_predicate(predicate, getExists);

    // Fast path: SUM over single identifier (no arithmetic ops)
    bool fast_sum = false; std::string fast_ident;
    if (!aggfunc.empty()) {
        std::string lf = aggfunc; for(char& ch: lf) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (lf=="sum" && rpn.size()==1 && rpn[0].type==expr::Token::Type::Ident) {
            fast_sum = true; fast_ident = rpn[0].text;
        }
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    double acc = 0.0;
    const std::size_t n = store.size;
    if (fast_sum) {
        // Direct accumulation without RPN evaluation
        const bool hasFloat = store.fcols.count(fast_ident);
        const bool hasInt = store.icols.count(fast_ident);
        if (hasFloat) {
            const auto& col = store.fcols[fast_ident];
            for (std::size_t i=0;i<n;++i) {
                if (!parsedClauses.empty() && !eval_predicate(parsedClauses, i, getFloat, getInt)) continue;
                acc += col[i];
            }
        } else if (hasInt) {
            const auto& col = store.icols[fast_ident];
            for (std::size_t i=0;i<n;++i) {
                if (!parsedClauses.empty() && !eval_predicate(parsedClauses, i, getFloat, getInt)) continue;
                acc += static_cast<double>(col[i]);
            }
        } else {
            // Fallback to generic path if identifier not found
            for (std::size_t i=0;i<n;++i) {
                if (!parsedClauses.empty() && !eval_predicate(parsedClauses, i, getFloat, getInt)) continue;
                acc += eval_rpn(rpn, i, getFloat);
            }
        }
    } else {
        for (std::size_t i=0;i<n;++i) {
            if (!parsedClauses.empty()) {
                if (!eval_predicate(parsedClauses, i, getFloat, getInt)) continue;
            } else if (!eval_predicate_conjunction(predicate, i, getFloat, getInt, getExists)) {
                continue; // fallback slow path
            }
            double v = eval_rpn(rpn, i, getFloat);
            acc += v;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {acc, cpu_ms};
}

} // namespace engine