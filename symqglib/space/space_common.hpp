#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

namespace symqg::space {
inline float reduce_add_m256(__m256 x) {
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
}

}  // namespace symqg::space