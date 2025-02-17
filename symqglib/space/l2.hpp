#pragma once

#include <immintrin.h>

#include <cstddef>

namespace symqg::space {

inline float reduce_add_m256(__m256 x) {
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
}

inline float l2_sqr(
    const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim
) {
    float result = 0;
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < mul16; i += 16) {
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
    size_t mul8 = dim - (dim & 0b111);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < mul8; i += 8) {
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
    for (size_t i = 0; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

inline float l2_sqr_single(const float* __restrict__ vec0, size_t dim) {
    float result = 0;
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < mul16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        sum = _mm512_fmadd_ps(xxx, xxx, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i];
        result += tmp * tmp;
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        float tmp = vec0[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

inline float l2_sqr_uint8(const uint8_t* __restrict__ vec0, const uint8_t* __restrict__ vec1, size_t dim) {
    const uint8_t *pVect1 = vec0;
    const uint8_t *pVect2 = vec1;

    size_t qty = dim;

    const uint8_t *pEnd1 = pVect1 + qty;

    __m256i v1, v2, v1_256_low, v1_256_high, v2_256_low, v2_256_high;
    __m256i diff_low, diff_high;
    __m256i sum256 = _mm256_setzero_si256(); // 初始和设置为零

    while (pVect1 + 32 <= pEnd1) { // 每次处理 32 个 uint8_t
        // 加载 32 个 uint8_t
        v1 = _mm256_loadu_si256((const __m256i *) pVect1);
        v2 = _mm256_loadu_si256((const __m256i *) pVect2);

        // 转换 uint8_t 到 uint16_t，降到 16 个元素
        v1_256_low = _mm256_unpacklo_epi8(v1, _mm256_setzero_si256());
        v1_256_high = _mm256_unpackhi_epi8(v1, _mm256_setzero_si256());
        v2_256_low = _mm256_unpacklo_epi8(v2, _mm256_setzero_si256());
        v2_256_high = _mm256_unpackhi_epi8(v2, _mm256_setzero_si256());

        // 计算差值
        diff_low = _mm256_sub_epi16(v1_256_low, v2_256_low);
        diff_high = _mm256_sub_epi16(v1_256_high, v2_256_high);

        // 使用 madd_epi16 进行平方
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(diff_low, diff_low));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(diff_high, diff_high));

        pVect1 += 32; // 加载下一个块
        pVect2 += 32; // 加载下一个块
    }

    // 提取结果并计算总和
    int32_t cTmp[8];
    _mm256_store_si256((__m256i *) cTmp, sum256);

    // 累加
    int32_t res = 0;
    for (int i = 0; i < 8; i++) {
        res += cTmp[i];
    }

    // 处理剩余的不足 32 字节的部分
    while (pVect1 < pEnd1) {
        int16_t diff = static_cast<int16_t>(*pVect1) - static_cast<int16_t>(*pVect2);
        res += diff * diff;
        pVect1++;
        pVect2++;
    }

    return static_cast<float>(res);
}

}  // namespace symqg::space