#pragma once

#include <immintrin.h>

#include <cstdint>
#include <vector>

#include "../quantization/fastscan_impl.hpp"

namespace symqg {

// RaBitQ distance formula (baseline):
// dist = sqr_y + triple_x + fac_dq * width * result + fac_vq * vl
static inline void appro_dist_rabitq(
    size_t num_points,
    float sqr_y,
    float width,
    float vl,
    const float* __restrict__ result,
    const float* __restrict__ triple_x,
    const float* __restrict__ fac_dq,
    const float* __restrict__ fac_vq,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 sqr_y_simd = _mm512_set1_ps(sqr_y);
    const __m512 width_simd = _mm512_set1_ps(width);
    const __m512 vl_simd = _mm512_set1_ps(vl);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 r = _mm512_loadu_ps(&result[i]);
        __m512 tx = _mm512_loadu_ps(&triple_x[i]);
        __m512 fd = _mm512_loadu_ps(&fac_dq[i]);
        __m512 fv = _mm512_loadu_ps(&fac_vq[i]);

        tx = _mm512_add_ps(tx, sqr_y_simd);
        fd = _mm512_mul_ps(fd, width_simd);
        fd = _mm512_mul_ps(fd, r);
        fv = _mm512_fmadd_ps(fv, vl_simd, tx);
        _mm512_storeu_ps(&appro_dist[i], _mm512_add_ps(fd, fv));
    }
#elif defined(__AVX2__)
    const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
    const __m256 width_simd = _mm256_set1_ps(width);
    const __m256 vl_simd = _mm256_set1_ps(vl);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 r = _mm256_loadu_ps(&result[i]);
        __m256 tx = _mm256_loadu_ps(&triple_x[i]);
        __m256 fd = _mm256_loadu_ps(&fac_dq[i]);
        __m256 fv = _mm256_loadu_ps(&fac_vq[i]);

        tx = _mm256_add_ps(tx, sqr_y_simd);
        fd = _mm256_mul_ps(fd, width_simd);
        fd = _mm256_mul_ps(fd, r);
        fv = _mm256_mul_ps(fv, vl_simd);
        _mm256_storeu_ps(&appro_dist[i],
            _mm256_add_ps(_mm256_add_ps(tx, fd), fv));
    }
#else
    std::cerr << "SIMD (AVX512 or AVX2) REQUIRED!\n";
    abort();
#endif
}

// SAQ distance formula:
// dist = sqr_y + fac_a + fac_b * width * result + fac_c * vl_half + fac_d * sum_q
static inline void appro_dist_saq(
    size_t num_points,
    float sqr_y,
    float width,
    float vl_half,
    float sum_q_float,
    const float* __restrict__ result,
    const float* __restrict__ fac_a,
    const float* __restrict__ fac_b,
    const float* __restrict__ fac_c,
    const float* __restrict__ fac_d,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 sy = _mm512_set1_ps(sqr_y);
    const __m512 w = _mm512_set1_ps(width);
    const __m512 vh = _mm512_set1_ps(vl_half);
    const __m512 sq = _mm512_set1_ps(sum_q_float);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 r = _mm512_loadu_ps(&result[i]);
        __m512 a = _mm512_loadu_ps(&fac_a[i]);
        __m512 b = _mm512_loadu_ps(&fac_b[i]);
        __m512 c = _mm512_loadu_ps(&fac_c[i]);
        __m512 d = _mm512_loadu_ps(&fac_d[i]);

        __m512 dist = _mm512_add_ps(sy, a);
        dist = _mm512_fmadd_ps(_mm512_mul_ps(b, w), r, dist);
        dist = _mm512_fmadd_ps(c, vh, dist);
        dist = _mm512_fmadd_ps(d, sq, dist);
        _mm512_storeu_ps(&appro_dist[i], dist);
    }
#elif defined(__AVX2__)
    const __m256 sy = _mm256_set1_ps(sqr_y);
    const __m256 w = _mm256_set1_ps(width);
    const __m256 vh = _mm256_set1_ps(vl_half);
    const __m256 sq = _mm256_set1_ps(sum_q_float);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 r = _mm256_loadu_ps(&result[i]);
        __m256 a = _mm256_loadu_ps(&fac_a[i]);
        __m256 b = _mm256_loadu_ps(&fac_b[i]);
        __m256 c = _mm256_loadu_ps(&fac_c[i]);
        __m256 d = _mm256_loadu_ps(&fac_d[i]);

        __m256 dist = _mm256_add_ps(sy, a);
        dist = _mm256_fmadd_ps(_mm256_mul_ps(b, w), r, dist);
        dist = _mm256_fmadd_ps(c, vh, dist);
        dist = _mm256_fmadd_ps(d, sq, dist);
        _mm256_storeu_ps(&appro_dist[i], dist);
    }
#else
    std::cerr << "SIMD (AVX512 or AVX2) REQUIRED!\n";
    abort();
#endif
}

class QGScanner {
   private:
    size_t padded_dim_;
    size_t fascscan_dim_;
    size_t degree_bound_;
    mutable std::vector<uint16_t> result_;
    mutable std::vector<float> result_float_;

   public:
    QGScanner() = default;

    explicit QGScanner(size_t padded_dim, size_t degree_bound)
        : padded_dim_(padded_dim)
        , fascscan_dim_(padded_dim * 4)
        , degree_bound_(degree_bound)
        , result_(degree_bound)
        , result_float_(degree_bound) {}

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
        // Fastscan accumulate with 4-bit codes (fascscan_dim = padded_dim * 4)
        for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
            accumulate_impl(fascscan_dim_, packed_code, LUT, &result_[i]);
            packed_code = &packed_code[fascscan_dim_ << 2];
        }

        // Convert uint16 result to float (raw dot product, no 2*-sumq conversion)
#if defined(__AVX512F__)
        for (size_t i = 0; i < degree_bound_; i += 32) {
            __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_[i]));
            __m256i i16b =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result_[i + 16]));
            __m512 f32a = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(i16a));
            __m512 f32b = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(i16b));
            _mm512_storeu_ps(&result_float_[i], f32a);
            _mm512_storeu_ps(&result_float_[i + 16], f32b);
        }
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            result_float_[i] = static_cast<float>(result_[i]);
        }
#endif

        // SAQ distance formula with 4 per-neighbor factors
        const float* fac_a = factor;
        const float* fac_b = &fac_a[degree_bound_];
        const float* fac_c = &fac_b[degree_bound_];
        const float* fac_d = &fac_c[degree_bound_];
        appro_dist_saq(
            degree_bound_, sqr_y, width, vl_half, sum_q_float,
            result_float_.data(), fac_a, fac_b, fac_c, fac_d, appro_dist
        );
    }
};
}  // namespace symqg
