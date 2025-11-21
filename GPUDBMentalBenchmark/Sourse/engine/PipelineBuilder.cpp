#include "PipelineBuilder.hpp"
#include <regex>

namespace engine {

static int parse_date_to_int(const std::string& s) {
    std::string t; t.reserve(8);
    for(char c: s) if (std::isdigit(static_cast<unsigned char>(c))) t.push_back(c);
    if (t.size()==8) return std::stoi(t);
    return 0;
}

PipelineSpecQ6 PipelineBuilder::buildQ6(const Plan& plan) {
    PipelineSpecQ6 spec{};
    // Defaults matching classic Q6
    spec.params.start_date = 19940101;
    spec.params.end_date = 19950101;
    spec.params.min_discount = 0.05f;
    spec.params.max_discount = 0.07f;
    spec.params.max_quantity = 24.0f;

    // Try to refine from Filter predicate if present
    for (const auto& n : plan.nodes) {
        if (n.type == IRNode::Type::Filter) {
            const std::string& p = n.filter.predicate;
            // shipdate >= DATE 'YYYY-MM-DD'
            std::regex re_start("shipdate\\s*>=\\s*date\\s*'([0-9-]+)'", std::regex::icase);
            std::smatch m;
            if (std::regex_search(p, m, re_start) && m.size()>1) {
                spec.params.start_date = parse_date_to_int(m[1].str());
            }
            // shipdate < DATE 'YYYY-MM-DD'
            std::regex re_end("shipdate\\s*<\\s*date\\s*'([0-9-]+)'", std::regex::icase);
            if (std::regex_search(p, m, re_end) && m.size()>1) {
                spec.params.end_date = parse_date_to_int(m[1].str());
            }
            // discount >= x, discount <= y
            std::regex re_dmin("discount\\s*>=\\s*([0-9]*\\.?[0-9]+)", std::regex::icase);
            if (std::regex_search(p, m, re_dmin) && m.size()>1) {
                spec.params.min_discount = std::stof(m[1].str());
            }
            std::regex re_dmax("discount\\s*<=\\s*([0-9]*\\.?[0-9]+)", std::regex::icase);
            if (std::regex_search(p, m, re_dmax) && m.size()>1) {
                spec.params.max_discount = std::stof(m[1].str());
            }
            // quantity < z
            std::regex re_qmax("quantity\\s*<\\s*([0-9]*\\.?[0-9]+)", std::regex::icase);
            if (std::regex_search(p, m, re_qmax) && m.size()>1) {
                spec.params.max_quantity = std::stof(m[1].str());
            }
            break;
        }
    }
    return spec;
}

} // namespace engine