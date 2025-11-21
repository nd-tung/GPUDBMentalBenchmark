#ifdef ENGINE_DEMO
#include "Pipelines.hpp"
#include <vector>
#include <cstdint>
#include <cstdio>

using namespace engine;

int main() {
    // Demo: filter eq on small array
    std::vector<uint32_t> data = {1,2,3,2,2,5};
    std::vector<uint8_t> mask(data.size());

    BufferView in{data.data(), data.size(), sizeof(uint32_t)};
    MutableBufferView out{mask.data(), mask.size(), sizeof(uint8_t)};

    Pipeline p;
    KernelConfig cfg{"filter_eq_u32", 256, 1};
    p.setConfig(cfg);
    p.addFilterEq(FilterProject::Params{2});
    p.run(in, out);

    for (size_t i = 0; i < mask.size(); ++i) {
        std::printf("%zu:%u\n", i, (unsigned)mask[i]);
    }
    return 0;
}
#endif
