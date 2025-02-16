#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

namespace symqg::space {

inline auto l2_sqr(
    const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim
) -> float {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < num16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        auto yyy = _mm512_loadu_ps(&vec1[i]);
        auto ttt = _mm512_sub_ps(xxx, yyy);
        sum = _mm512_fmadd_ps(ttt, ttt, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

#elif defined(__AVX2__)
    size_t num8 = dim - (dim & 0b111);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < num8; i += 8) {
        __m256 xx = _mm256_loadu_ps(&vec0[i]);
        __m256 yy = _mm256_loadu_ps(&vec1[i]);
        __m256 t = _mm256_sub_ps(xx, yy);
        sum = _mm256_fmadd_ps(t, t, sum);
    }
    result = reduce_add_m256(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

inline float l2_spr_uint8(const uint8_t* __restrict__ vec0, const uint8_t* __restrict__ vec1, size_t dim) {
    const uint8_t *pVect1 = (const uint8_t *) vec0;
    const uint8_t *pVect2 = (const uint8_t *) vec1;

    size_t qty = dim;

    const uint8_t *pEnd1 = pVect1 + qty;

    __m512i v1, v2, v1_512_low, v1_512_high, v2_512_low, v2_512_high;
    __m512i diff_low, diff_high;
    __m512i sum512 = _mm512_setzero_si512(); // 初始和设置为零

    while (pVect1 < pEnd1) {
        // 加载 64 个 uint8_t
        v1 = _mm512_loadu_si512((const __m512i *) pVect1);
        v2 = _mm512_loadu_si512((const __m512i *) pVect2);

        // 转换 uint8_t 到 uint16_t，降到 32 个 nhp
        v1_512_low = _mm512_unpacklo_epi8(v1, _mm512_setzero_si512());
        v1_512_high = _mm512_unpackhi_epi8(v1, _mm512_setzero_si512());
        v2_512_low = _mm512_unpacklo_epi8(v2, _mm512_setzero_si512());
        v2_512_high = _mm512_unpackhi_epi8(v2, _mm512_setzero_si512());

        // 计算差值
        diff_low = _mm512_sub_epi16(v1_512_low, v2_512_low);
        diff_high = _mm512_sub_epi16(v1_512_high, v2_512_high);

        // 使用 madd_epi16 进行平方
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(diff_low, diff_low));
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(diff_high, diff_high));

        pVect1 += 64; // 加载下一个块
        pVect2 += 64; // 加载下一个块
    }

    // 提取结果并计算总和
    int32_t cTmp[16];
    _mm512_store_si512((__m512i *) cTmp, sum512);

    // 累加
    int32_t res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }

    return static_cast<float>(res);
}

}  // namespace symqg::space