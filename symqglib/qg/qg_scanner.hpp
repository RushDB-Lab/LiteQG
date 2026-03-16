#pragma once

#include <immintrin.h>

#include <cstdint>
#include <vector>

#include "../quantization/fastscan_impl.hpp"

namespace symqg {

// 3-factor kernel: dist = sqr_y + adj_a + b0 * w * r0 + b1 * w * r1
static inline void appro_dist_2seg(
    size_t num_points,
    float sqr_y,
    float width,
    const float* __restrict__ result0,
    const float* __restrict__ result1,
    const float* __restrict__ adjusted_a,
    const float* __restrict__ fac_b0,
    const float* __restrict__ fac_b1,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 sy = _mm512_set1_ps(sqr_y);
    const __m512 w = _mm512_set1_ps(width);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 r0 = _mm512_loadu_ps(&result0[i]);
        __m512 r1 = _mm512_loadu_ps(&result1[i]);
        __m512 a = _mm512_loadu_ps(&adjusted_a[i]);
        __m512 b0 = _mm512_loadu_ps(&fac_b0[i]);
        __m512 b1 = _mm512_loadu_ps(&fac_b1[i]);

        __m512 dist = _mm512_add_ps(sy, a);
        dist = _mm512_fmadd_ps(_mm512_mul_ps(b0, w), r0, dist);
        dist = _mm512_fmadd_ps(_mm512_mul_ps(b1, w), r1, dist);
        _mm512_storeu_ps(&appro_dist[i], dist);
    }
#elif defined(__AVX2__)
    const __m256 sy = _mm256_set1_ps(sqr_y);
    const __m256 w = _mm256_set1_ps(width);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 r0 = _mm256_loadu_ps(&result0[i]);
        __m256 r1 = _mm256_loadu_ps(&result1[i]);
        __m256 a = _mm256_loadu_ps(&adjusted_a[i]);
        __m256 b0 = _mm256_loadu_ps(&fac_b0[i]);
        __m256 b1 = _mm256_loadu_ps(&fac_b1[i]);

        __m256 dist = _mm256_add_ps(sy, a);
        dist = _mm256_fmadd_ps(_mm256_mul_ps(b0, w), r0, dist);
        dist = _mm256_fmadd_ps(_mm256_mul_ps(b1, w), r1, dist);
        _mm256_storeu_ps(&appro_dist[i], dist);
    }
#else
    for (size_t i = 0; i < num_points; ++i) {
        appro_dist[i] = sqr_y + adjusted_a[i]
            + fac_b0[i] * width * result0[i]
            + fac_b1[i] * width * result1[i];
    }
#endif
}

class QGScanner {
   private:
    size_t split_dim_;
    size_t seg0_fascscan_dim_;  // split_dim * 4 (4-bit)
    size_t seg1_dim_;           // padded_dim - split_dim
    size_t degree_bound_;

    mutable std::vector<uint16_t> result0_;
    mutable std::vector<uint16_t> result1_;
    mutable std::vector<float> result0_float_;
    mutable std::vector<float> result1_float_;
    mutable std::vector<float> adjusted_a_;

   public:
    QGScanner() = default;

    explicit QGScanner(size_t padded_dim, size_t split_dim, size_t degree_bound)
        : split_dim_(split_dim)
        , seg0_fascscan_dim_(split_dim * 4)
        , seg1_dim_(padded_dim - split_dim)
        , degree_bound_(degree_bound)
        , result0_(degree_bound)
        , result1_(degree_bound)
        , result0_float_(degree_bound)
        , result1_float_(degree_bound)
        , adjusted_a_(degree_bound) {}

    // seg0_code_bytes: bytes per batch for seg0 4-bit codes
    [[nodiscard]] size_t seg0_code_bytes() const {
        return seg0_fascscan_dim_ * 4;  // code_length for accumulate_impl
    }
    // seg1_code_bytes: bytes per batch for seg1 1-bit codes
    [[nodiscard]] size_t seg1_code_bytes() const {
        return seg1_dim_ * 4;
    }

    void scan_neighbors(
        float* __restrict__ appro_dist,
        const uint8_t* __restrict__ lut_seg0,
        const uint8_t* __restrict__ lut_seg1,
        float sqr_y,
        float width,
        float vl_half,
        float sum_q_seg0,
        float sum_q_seg1,
        const uint8_t* seg0_code,
        const uint8_t* seg1_code,
        const float* factor
    ) const {
        // Seg0: 4-bit fascscan
        for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
            accumulate_impl(seg0_fascscan_dim_, seg0_code, lut_seg0, &result0_[i]);
            seg0_code += seg0_fascscan_dim_ << 2;
        }

        // Seg1: 1-bit fascscan
        for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
            accumulate_impl(seg1_dim_, seg1_code, lut_seg1, &result1_[i]);
            seg1_code += seg1_dim_ << 2;
        }

        // uint16 → float (raw dot products, no 2*-sumq)
#if defined(__AVX512F__)
        for (size_t i = 0; i < degree_bound_; i += 32) {
            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result0_[i]));
            __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result0_[i+16]));
            _mm512_storeu_ps(&result0_float_[i], _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a0)));
            _mm512_storeu_ps(&result0_float_[i+16], _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b0)));

            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result1_[i]));
            __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result1_[i+16]));
            _mm512_storeu_ps(&result1_float_[i], _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a1)));
            _mm512_storeu_ps(&result1_float_[i+16], _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b1)));
        }
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            result0_float_[i] = static_cast<float>(result0_[i]);
            result1_float_[i] = static_cast<float>(result1_[i]);
        }
#endif

        // Factor layout: [fac_a | fac_b0 | fac_b1 | fac_c | fac_d0 | fac_d1]
        const float* fac_a  = factor;
        const float* fac_b0 = &fac_a[degree_bound_];
        const float* fac_b1 = &fac_b0[degree_bound_];
        const float* fac_c  = &fac_b1[degree_bound_];
        const float* fac_d0 = &fac_c[degree_bound_];
        const float* fac_d1 = &fac_d0[degree_bound_];

        // Pre-fold: adjusted_a = fac_a + fac_c*vl_half + fac_d0*sum_q_0 + fac_d1*sum_q_1
#if defined(__AVX512F__)
        const __m512 vh = _mm512_set1_ps(vl_half);
        const __m512 sq0 = _mm512_set1_ps(sum_q_seg0);
        const __m512 sq1 = _mm512_set1_ps(sum_q_seg1);
        for (size_t i = 0; i < degree_bound_; i += 16) {
            __m512 a = _mm512_loadu_ps(&fac_a[i]);
            a = _mm512_fmadd_ps(_mm512_loadu_ps(&fac_c[i]), vh, a);
            a = _mm512_fmadd_ps(_mm512_loadu_ps(&fac_d0[i]), sq0, a);
            a = _mm512_fmadd_ps(_mm512_loadu_ps(&fac_d1[i]), sq1, a);
            _mm512_storeu_ps(&adjusted_a_[i], a);
        }
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            adjusted_a_[i] = fac_a[i] + fac_c[i] * vl_half
                + fac_d0[i] * sum_q_seg0 + fac_d1[i] * sum_q_seg1;
        }
#endif

        appro_dist_2seg(
            degree_bound_, sqr_y, width,
            result0_float_.data(), result1_float_.data(),
            adjusted_a_.data(), fac_b0, fac_b1, appro_dist
        );
    }
};
}  // namespace symqg
