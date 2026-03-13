#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace symqg {

// SoA layout: separate min[] and scale[] arrays for SIMD-friendly access
inline void sq8_quantize(
    const float* data,
    size_t dim,
    size_t num,
    uint8_t* quantized,
    float* min_vals,
    float* scale_vals
) {
    for (size_t d = 0; d < dim; ++d) {
        float lo = data[d], hi = data[d];
        for (size_t i = 1; i < num; ++i) {
            float val = data[i * dim + d];
            lo = std::min(lo, val);
            hi = std::max(hi, val);
        }
        min_vals[d] = lo;
        scale_vals[d] = (hi == lo) ? 1.0f : (hi - lo) / 255.0f;
        for (size_t i = 0; i < num; ++i) {
            float val = data[i * dim + d];
            quantized[i * dim + d] =
                static_cast<uint8_t>(std::round((val - lo) / scale_vals[d]));
        }
    }
}

inline void sq8_dequantize(
    const uint8_t* quantized,
    size_t dim,
    float* data,
    const float* min_vals,
    const float* scale_vals
) {
    for (size_t i = 0; i < dim; ++i) {
        data[i] = min_vals[i] + quantized[i] * scale_vals[i];
    }
}

// Fused SQ8 dequantize + L2 squared distance in a single pass.
// Computes sum((query[i] - (min[i] + quantized[i] * scale[i]))^2)
// without materializing the decoded vector.
inline float sq8_l2_sqr(
    const float* __restrict__ query,
    const uint8_t* __restrict__ quantized,
    const float* __restrict__ min_vals,
    const float* __restrict__ scale_vals,
    size_t dim
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // uint8 x16 → int32 x16 → float x16
        __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&quantized[i]));
        __m512 fval = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8));

        // decoded = min + val * scale  (FMA)
        __m512 decoded = _mm512_fmadd_ps(
            fval, _mm512_loadu_ps(&scale_vals[i]), _mm512_loadu_ps(&min_vals[i])
        );

        // diff = query - decoded;  sum += diff * diff
        __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(&query[i]), decoded);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float diff = query[i] - (min_vals[i] + quantized[i] * scale_vals[i]);
        result += diff * diff;
    }

#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        // uint8 x8 → int32 x8 → float x8
        __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&quantized[i]));
        __m256 fval = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8));

        __m256 decoded = _mm256_fmadd_ps(
            fval, _mm256_loadu_ps(&scale_vals[i]), _mm256_loadu_ps(&min_vals[i])
        );

        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(&query[i]), decoded);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    // horizontal reduction
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    result = _mm_cvtss_f32(s);
    for (; i < dim; ++i) {
        float diff = query[i] - (min_vals[i] + quantized[i] * scale_vals[i]);
        result += diff * diff;
    }

#else
    for (size_t i = 0; i < dim; ++i) {
        float diff = query[i] - (min_vals[i] + quantized[i] * scale_vals[i]);
        result += diff * diff;
    }
#endif
    return result;
}

// Quantize a query vector to uint8 using the same SQ8 parameters as the database.
// Enables direct uint8-vs-uint8 L2 distance computation.
inline void sq8_quantize_query(
    const float* __restrict__ query,
    size_t dim,
    const float* __restrict__ min_vals,
    const float* __restrict__ scale_vals,
    uint8_t* __restrict__ out
) {
    for (size_t i = 0; i < dim; ++i) {
        float val = (query[i] - min_vals[i]) / scale_vals[i];
        val = std::max(0.0f, std::min(255.0f, std::round(val)));
        out[i] = static_cast<uint8_t>(val);
    }
}

}  // namespace symqg
