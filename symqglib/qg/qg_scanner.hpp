#pragma once

#include <immintrin.h>

#include <cstdint>
#include <vector>

#include "../quantization/fastscan_impl.hpp"

namespace symqg {

// 2-factor distance kernel: dist = sqr_y + adjusted_a + fac_b * width * result
static inline void appro_dist_2fac(
    size_t num_points,
    float sqr_y,
    float neg2_width,
    const float* __restrict__ result,
    const float* __restrict__ adjusted_a,
    const float* __restrict__ fac_b,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 sy = _mm512_set1_ps(sqr_y);
    const __m512 nw = _mm512_set1_ps(neg2_width);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 r = _mm512_loadu_ps(&result[i]);
        __m512 a = _mm512_loadu_ps(&adjusted_a[i]);
        __m512 b = _mm512_loadu_ps(&fac_b[i]);

        __m512 dist = _mm512_add_ps(sy, a);
        dist = _mm512_fmadd_ps(_mm512_mul_ps(b, nw), r, dist);
        _mm512_storeu_ps(&appro_dist[i], dist);
    }
#elif defined(__AVX2__)
    const __m256 sy = _mm256_set1_ps(sqr_y);
    const __m256 nw = _mm256_set1_ps(neg2_width);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 r = _mm256_loadu_ps(&result[i]);
        __m256 a = _mm256_loadu_ps(&adjusted_a[i]);
        __m256 b = _mm256_loadu_ps(&fac_b[i]);

        __m256 dist = _mm256_add_ps(sy, a);
        dist = _mm256_fmadd_ps(_mm256_mul_ps(b, nw), r, dist);
        _mm256_storeu_ps(&appro_dist[i], dist);
    }
#else
    for (size_t i = 0; i < num_points; ++i) {
        appro_dist[i] = sqr_y + adjusted_a[i] + fac_b[i] * neg2_width * result[i];
    }
#endif
}

class QGScanner {
   private:
    size_t padded_dim_;
    size_t fascscan_dim_;
    size_t degree_bound_;
    // Pre-allocated buffers
    mutable std::vector<uint16_t> result_;
    mutable std::vector<float> result_float_;
    mutable std::vector<float> adjusted_a_;

   public:
    QGScanner() = default;

    explicit QGScanner(size_t padded_dim, size_t degree_bound)
        : padded_dim_(padded_dim)
        , fascscan_dim_(padded_dim * 4)
        , degree_bound_(degree_bound)
        , result_(degree_bound)
        , result_float_(degree_bound)
        , adjusted_a_(degree_bound) {}

    void pack_lut(const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT) const {
        pack_lut_4bit(padded_dim_, byte_query, LUT);
    }

    void scan_neighbors(
        float* __restrict__ appro_dist,
        const uint8_t* __restrict__ LUT,
        float sqr_y,
        float width,
        float vl_half,
        float sum_q_float,
        const uint8_t* packed_code,
        const float* factor
    ) const {
        // Fastscan accumulate (4-bit codes)
        for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
            accumulate_impl(fascscan_dim_, packed_code, LUT, &result_[i]);
            packed_code = &packed_code[fascscan_dim_ << 2];
        }

        // uint16 → float (raw dot product)
#if defined(__AVX512F__)
        for (size_t i = 0; i < degree_bound_; i += 32) {
            __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_[i]));
            __m256i i16b =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_[i + 16]));
            _mm512_storeu_ps(&result_float_[i],
                _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(i16a)));
            _mm512_storeu_ps(&result_float_[i + 16],
                _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(i16b)));
        }
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            result_float_[i] = static_cast<float>(result_[i]);
        }
#endif

        // Pre-pass: fold fac_c * vl_half + fac_d * sum_q into adjusted_a
        const float* fac_a = factor;
        const float* fac_b = &fac_a[degree_bound_];
        const float* fac_c = &fac_b[degree_bound_];
        const float* fac_d = &fac_c[degree_bound_];

#if defined(__AVX512F__)
        const __m512 vh = _mm512_set1_ps(vl_half);
        const __m512 sq = _mm512_set1_ps(sum_q_float);
        for (size_t i = 0; i < degree_bound_; i += 16) {
            __m512 a = _mm512_loadu_ps(&fac_a[i]);
            __m512 c = _mm512_loadu_ps(&fac_c[i]);
            __m512 d = _mm512_loadu_ps(&fac_d[i]);
            a = _mm512_fmadd_ps(c, vh, a);
            a = _mm512_fmadd_ps(d, sq, a);
            _mm512_storeu_ps(&adjusted_a_[i], a);
        }
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            adjusted_a_[i] = fac_a[i] + fac_c[i] * vl_half + fac_d[i] * sum_q_float;
        }
#endif

        // 2-factor distance kernel
        appro_dist_2fac(
            degree_bound_, sqr_y, width,
            result_float_.data(), adjusted_a_.data(), fac_b, appro_dist
        );
    }
};
}  // namespace symqg
