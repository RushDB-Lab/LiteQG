#pragma once

#include <immintrin.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace symqg {

class QueryQuantizer {
   private:
    size_t padded_dim_ = 0;

   public:
    QueryQuantizer() = default;

    explicit QueryQuantizer(size_t padded_dim) : padded_dim_(padded_dim) {}

    ~QueryQuantizer() = default;

    void query_range(const float* __restrict__ query, float& lo, float& hi) const {
#if defined(__AVX512F__)
        __m512 max_q = _mm512_setzero_ps();
        __m512 min_q = _mm512_setzero_ps();
        for (size_t i = 0; i < padded_dim_; i += 16) {
            __m512 y1 = _mm512_load_ps(&query[i]);
            max_q = _mm512_max_ps(y1, max_q);
            min_q = _mm512_min_ps(y1, min_q);
        }
        hi = _mm512_reduce_max_ps(max_q);
        lo = _mm512_reduce_min_ps(min_q);
#else
        lo = FLT_MAX;
        hi = FLT_MIN;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float tmp = query[i];
            lo = tmp < lo ? tmp : lo;
            hi = tmp > hi ? tmp : hi;
        }
#endif
    }

    void quantize(
        uint8_t* __restrict__ result,
        const float* __restrict__ query,
        float lo,
        float width,
        int32_t& sum_q
    ) const {
        float one_over_width = 1.0F / width;
        int32_t sum = 0;
        uint8_t cur;
        for (size_t i = 0; i < padded_dim_; ++i) {
            cur = static_cast<uint8_t>(std::lround((query[i] - lo) * one_over_width + 0.5));
            result[i] = cur;
            sum += cur;
        }
        sum_q = sum;
    }
};
}  // namespace symqg