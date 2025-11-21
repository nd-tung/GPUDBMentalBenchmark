#include "Operators.hpp"
#include <cstring>

namespace engine {

void FilterProject::init(const KernelConfig& cfg) { cfg_ = cfg; }

void FilterProject::dispatch(const BufferView& in, MutableBufferView maskOut, const Params& p) {
    // CPU placeholder implementation for correctness, not performance.
    const auto* inU32 = static_cast<const uint32_t*>(in.data);
    auto* out = static_cast<uint8_t*>(maskOut.data);
    const std::size_t n = in.count;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (inU32[i] == p.eqValue) ? 1 : 0;
    }
}

void HashJoinU32::init(const KernelConfig& buildCfg, const KernelConfig& probeCfg, std::size_t capacity) {
    buildCfg_ = buildCfg;
    probeCfg_ = probeCfg;
    capacity_ = capacity;
}

void HashJoinU32::build(const BuildInput& in) {
    // CPU placeholder: no-op for now.
    (void)in; // suppress unused
}

void HashJoinU32::probe(const ProbeInput& in, Output out) {
    // CPU placeholder: write zeros
    auto* payloads = static_cast<uint32_t*>(out.payloads.data);
    std::memset(payloads, 0, out.payloads.count * sizeof(uint32_t));
    (void)in;
}

void GroupBySumF32::init(const KernelConfig& cfg, std::size_t bucketsPow2) {
    cfg_ = cfg;
    // assume bucketsPow2 is power-of-two size; mask = size-1
    bucketMask_ = (bucketsPow2 > 0) ? (bucketsPow2 - 1) : 0;
}

void GroupBySumF32::aggregate(const Input& in, Output out) {
    // CPU placeholder: naive aggregation by slot = key & mask
    const auto* keys = static_cast<const uint32_t*>(in.keys.data);
    const auto* vals = static_cast<const float*>(in.vals.data);
    auto* outKeys = static_cast<uint32_t*>(out.bucketKeys.data);
    auto* outCounts = static_cast<uint32_t*>(out.bucketCounts.data);
    auto* outSumBits = static_cast<uint32_t*>(out.bucketSumsBits.data);

    const std::size_t n = in.keys.count;
    const std::size_t buckets = out.bucketKeys.count;

    // zero init outputs
    std::memset(outKeys, 0, buckets * sizeof(uint32_t));
    std::memset(outCounts, 0, buckets * sizeof(uint32_t));
    std::memset(outSumBits, 0, buckets * sizeof(uint32_t));

    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t k = keys[i];
        const float v = vals[i];
        const std::size_t slot = static_cast<std::size_t>(k & static_cast<uint32_t>(bucketMask_));
        outKeys[slot] = k;
        outCounts[slot] += 1;
        float cur = std::bit_cast<float>(outSumBits[slot]);
        cur += v;
        outSumBits[slot] = std::bit_cast<uint32_t>(cur);
    }
}

double FilterQ6::computeRevenue(const BufferView& shipdate,
                                const BufferView& discount,
                                const BufferView& quantity,
                                const BufferView& extendedprice,
                                const Params& p) {
    const auto n = shipdate.count;
    const auto* sd = static_cast<const int*>(shipdate.data);
    const auto* d = static_cast<const float*>(discount.data);
    const auto* q = static_cast<const float*>(quantity.data);
    const auto* ep = static_cast<const float*>(extendedprice.data);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const int s = sd[i];
        const float disc = d[i];
        const float qty = q[i];
        if (s >= p.start_date && s < p.end_date &&
            disc >= p.min_discount && disc <= p.max_discount &&
            qty < p.max_quantity) {
            sum += static_cast<double>(ep[i]) * static_cast<double>(disc);
        }
    }
    return sum;
}

} // namespace engine
