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
    int32_t res = 0;
#if defined(__AVX512BW__)
    __m512i sum512 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vec0[i]));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vec1[i]));
        __m512i v1_16 = _mm512_cvtepu8_epi16(v1);
        __m512i v2_16 = _mm512_cvtepu8_epi16(v2);
        __m512i diff = _mm512_sub_epi16(v1_16, v2_16);
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(diff, diff));
    }
    res = _mm512_reduce_add_epi32(sum512);
    for (; i < dim; ++i) {
        int16_t diff = static_cast<int16_t>(vec0[i]) - static_cast<int16_t>(vec1[i]);
        res += diff * diff;
    }

#elif defined(__AVX2__)
    __m256i sum256 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vec0[i]));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vec1[i]));
        __m256i v1_lo = _mm256_unpacklo_epi8(v1, _mm256_setzero_si256());
        __m256i v1_hi = _mm256_unpackhi_epi8(v1, _mm256_setzero_si256());
        __m256i v2_lo = _mm256_unpacklo_epi8(v2, _mm256_setzero_si256());
        __m256i v2_hi = _mm256_unpackhi_epi8(v2, _mm256_setzero_si256());
        __m256i diff_lo = _mm256_sub_epi16(v1_lo, v2_lo);
        __m256i diff_hi = _mm256_sub_epi16(v1_hi, v2_hi);
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(diff_lo, diff_lo));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(diff_hi, diff_hi));
    }
    // horizontal sum
    __m128i lo128 = _mm256_castsi256_si128(sum256);
    __m128i hi128 = _mm256_extracti128_si256(sum256, 1);
    __m128i s = _mm_add_epi32(lo128, hi128);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    res = _mm_cvtsi128_si32(s);
    for (; i < dim; ++i) {
        int16_t diff = static_cast<int16_t>(vec0[i]) - static_cast<int16_t>(vec1[i]);
        res += diff * diff;
    }

#else
    for (size_t i = 0; i < dim; ++i) {
        int16_t diff = static_cast<int16_t>(vec0[i]) - static_cast<int16_t>(vec1[i]);
        res += diff * diff;
    }
#endif
    return static_cast<float>(res);
}

}  // namespace symqg::space