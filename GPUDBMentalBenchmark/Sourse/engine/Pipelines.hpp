#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "Operators.hpp"

namespace engine {

class Pipeline {
public:
    void addFilterEq(engine::FilterProject::Params p) { filterParams_ = p; }
    void setConfig(const KernelConfig& cfg) { cfg_ = cfg; }

    // Minimal API placeholder
    void run(const BufferView& in, MutableBufferView maskOut) {
        FilterProject op;
        op.init(cfg_);
        op.dispatch(in, maskOut, filterParams_);
    }

private:
    KernelConfig cfg_{};
    FilterProject::Params filterParams_{};
};

} // namespace engine
