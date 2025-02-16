// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>

#include <functional>
#include <iostream>

#include "../../utils/memory.hpp"

namespace symqg {

#define LOWBIT(x) ((x) & (-(x)))

constexpr static size_t kBatchSize = 32;
constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};

template <size_t B>
static inline void pack_lut_impl(
    const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT
);

template <size_t B>
static inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result
);

template <size_t B, size_t D>
static inline void fastscan_impl(
    float* __restrict__ appro_dist,
    const uint8_t* __restrict__ LUT,
    float sqr_y,
    float vl,
    float width,
    int32_t sumq,
    const uint8_t* __restrict__ packed_code,
    const float* __restrict__ factor
);

template <size_t L>
static inline void fast_appro_dist(
    float sqr_y,
    float width,
    float vl,
    const float* __restrict__ result,
    const float* __restrict__ triple_x,
    const float* __restrict__ fac_dq,
    const float* __restrict__ fac_vq,
    float* __restrict__ appro_dist
);

class Scanner {
   private:
    // func for packing lookup tables
    std::function<void(const uint8_t* __restrict__, uint8_t* __restrict__)> pack_lut_func_ =
        pack_lut_impl<128>;
    // func for fastscan
    std::function<
        void(float* __restrict__, const uint8_t* __restrict__, float, float, float, int32_t, const uint8_t*, const float*)>
        fastscan_func_ = fastscan_impl<128, 32>;

   public:
    Scanner() = default;

    explicit Scanner(size_t padded_dim, size_t degree_bound) {
        // currently, due to ffht, the padded dimension is power of 2
        switch (padded_dim) {
            case 64:
                this->pack_lut_func_ = pack_lut_impl<64>;
                init_fast_func<64>(degree_bound);
                break;
            case 128:
                this->pack_lut_func_ = pack_lut_impl<128>;
                init_fast_func<128>(degree_bound);
                break;
            case 256:
                this->pack_lut_func_ = pack_lut_impl<256>;
                init_fast_func<256>(degree_bound);
                break;
            case 512:
                this->pack_lut_func_ = pack_lut_impl<512>;
                init_fast_func<512>(degree_bound);
                break;
            case 1024:
                this->pack_lut_func_ = pack_lut_impl<1024>;
                init_fast_func<1024>(degree_bound);
                break;
            case 2048:
                this->pack_lut_func_ = pack_lut_impl<2048>;
                init_fast_func<2048>(degree_bound);
                break;
            default:
                std::cerr << "Invalid padded dimension\n";
                abort();
        };
    }

    template <size_t PaddedDim>
    void init_fast_func(size_t degree_bound) {
        switch (degree_bound) {
            case 32:
                this->fastscan_func_ = fastscan_impl<PaddedDim, 32>;
                break;
            case 64:
                this->fastscan_func_ = fastscan_impl<PaddedDim, 64>;
                break;
            case 96:
                this->fastscan_func_ = fastscan_impl<PaddedDim, 96>;
                break;
            case 128:
                this->fastscan_func_ = fastscan_impl<PaddedDim, 128>;
                break;
            default:
                std::cerr << "Invalid degree bound\n";
                abort();
        }
    }

    void pack_lut(const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT) const {
        this->pack_lut_func_(byte_query, LUT);
    }

    void fastscan(
        float* __restrict__ appro_dist,
        const uint8_t* __restrict__ LUT,
        float sqr_y,
        float vl,
        float width,
        int32_t sumq,
        const uint8_t* packed_code,
        const float* factor
    ) const {
        this->fastscan_func_(appro_dist, LUT, sqr_y, vl, width, sumq, packed_code, factor);
    }
};

template <size_t PaddedDim>
static inline void pack_lut_impl(
    const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT
) {
    constexpr size_t kNumCodebook = PaddedDim >> 2;  // num of codebook
    for (size_t i = 0; i < kNumCodebook; ++i) {
        LUT[0] = 0;
        for (int j = 1; j < 16; ++j) {
            LUT[j] = LUT[j - LOWBIT(j)] + byte_query[kPos[j]];
        }
        LUT += 16;
        byte_query += 4;
    }
}

template <size_t PaddedDim>
static inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result
) {
    constexpr size_t kCodeLength = PaddedDim << 2;  // kBatchSize(32) * PaddedDim / 8
#if defined(__AVX512F__)
    __m512i c;
    __m512i lo;
    __m512i hi;
    __m512i lut;
    __m512i res_lo;
    __m512i res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < kCodeLength; i += 128) {
        c = _mm512_load_si512(&codes[i]);
        lut = _mm512_load_si512(&LUT[i]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

        c = _mm512_load_si512(&codes[i + 64]);
        lut = _mm512_load_si512(&LUT[i + 64]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }
    // v2test
    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

    __m512i ret1 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
        _mm512_shuffle_i64x2(accu0, accu1, 0b01001110)
    );
    __m512i ret2 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
        _mm512_shuffle_i64x2(accu2, accu3, 0b01001110)
    );
    __m512i ret = _mm512_setzero_si512();

    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

    _mm512_store_si512(result, ret);

    // __m256i res0 = _mm256_add_epi16(_mm512_castsi512_si256(accu0),
    // _mm512_extracti64x4_epi64(accu0, 1));
    // __m256i res1 = _mm256_add_epi16(_mm512_castsi512_si256(accu1),
    // _mm512_extracti64x4_epi64(accu1, 1));

    // res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    // __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(res0, res1,
    // 0x21),_mm256_blend_epi32(res0, res1, 0xF0)); _mm256_store_si256((__m256i*)result,
    // dis0);

    // __m256i res2 = _mm256_add_epi16(_mm512_castsi512_si256(accu2),
    // _mm512_extracti64x4_epi64(accu2, 1));
    // __m256i res3 = _mm256_add_epi16(_mm512_castsi512_si256(accu3),
    // _mm512_extracti64x4_epi64(accu3, 1));

    // res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));
    // __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(res2, res3,
    // 0x21),_mm256_blend_epi32(res2, res3, 0xF0));
    // _mm256_store_si256((__m256i*)&result[16], dis1);

#elif defined(__AVX2__)
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < kCodeLength; i += 64) {
        c = _mm256_load_si256((__m256i*)&codes[i]);
        lut = _mm256_load_si256((__m256i*)&LUT[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_load_si256((__m256i*)&codes[i + 32]);
        lut = _mm256_load_si256((__m256i*)&LUT[i + 32]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_store_si256((__m256i*)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_store_si256((__m256i*)&result[16], dis1);
#else
    std::cerr << "NO AVX SIMD SUPPORTED!\n";
    abort();
#endif
}

/**
 * @brief implementation of using fast scan to estimate distances for a vertex's neighbors
 *
 * @tparam PaddedDim    padded dimension
 * @tparam Degree       degree of current vertex, must be a multiple of 32 (batch size)
 * @param appro_dist    arr to store computed approximate distance
 * @param LUT           lookup tables
 * @param sqr_y         sqr of L2 distance between query and current vertex
 * @param vl            lower bound for rotated query vector
 * @param width         delta for quatizing query vector
 * @param sumq          sum of quantized query vector
 * @param packed_code   quantization codes for current vertex's neighbors, must be aligned
 * to 64 bytes
 * @param factor        pre-computed factors, must be aligned to 64 bytes
 */
template <size_t PaddedDim, size_t Degree>
static inline void fastscan_impl(
    float* __restrict__ appro_dist,
    const uint8_t* __restrict__ LUT,
    float sqr_y,
    float vl,
    float width,
    int32_t sumq,
    const uint8_t* __restrict__ packed_code,
    const float* __restrict__ factor
) {
    constexpr size_t kCodeLength = PaddedDim << 2;  // kBatchSize * B / 8
    std::array<uint16_t, Degree> result PORTABLE_ALIGN64;

    /* Compute block by block */
    for (size_t i = 0; i < Degree; i += kBatchSize) {
        accumulate<PaddedDim>(packed_code, LUT, &result[i]);
        packed_code = &packed_code[kCodeLength];
    }

    /* Cast to float and multiple by 2 then minus sumq */
    std::array<float, Degree> result_float PORTABLE_ALIGN32;
#if defined(__AVX512F__)
    const __m512i qq = _mm512_set1_epi32(sumq);
    for (size_t i = 0; i < Degree; i += 32) {
        __m256i i16a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&result[i]));
        __m256i i16b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&result[i + 16]));
        __m512i i32a = _mm512_cvtepi16_epi32(i16a);
        __m512i i32b = _mm512_cvtepi16_epi32(i16b);

        i32a = _mm512_sub_epi32(_mm512_slli_epi32(i32a, 1), qq);
        i32b = _mm512_sub_epi32(_mm512_slli_epi32(i32b, 1), qq);
        __m512 f32a = _mm512_cvtepi32_ps(i32a);
        __m512 f32b = _mm512_cvtepi32_ps(i32b);

        _mm512_store_ps(&result_float[i], f32a);
        _mm512_store_ps(&result_float[i + 16], f32b);
    }
#else
    for (size_t i = 0; i < D; ++i) {
        result_float[i] = static_cast<float>((static_cast<int>(result[i]) << 1) - sumq);
    }
#endif
    const float* triple_x = factor;
    const float* fac_dq = &triple_x[Degree];
    const float* fac_vq = &fac_dq[Degree];
    fast_appro_dist<Degree>(
        sqr_y, width, vl, result_float.data(), triple_x, fac_dq, fac_vq, appro_dist
    );
}

/**
 * @brief compute approximate distances by SIMD
 *
 * @tparam Degree       degree of current vertex, must be a multiple of 32 (batch size)
 * @param sqr_y         sqr L2 between query and centroid
 * @param width         quantization constant
 * @param vl            lowest value of rotated query
 * @param result        result of 2xfast_scan - sumq, must be aligned to 64 bytes
 * @param triple_x      precomputed constant, must be aligned to 64 bytes
 * @param fac_dq        precomputed constant, must be aligned to 64 bytes
 * @param fac_vq        precomputed constant, must be aligned to 64 bytes
 * @param appro_dist    approximated distances results, must be aligned to 64 bytes
 */
template <size_t Degree>
static inline void fast_appro_dist(
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

    __m512 result_simd;
    __m512 triple_x_simd;
    __m512 fac_dq_simd;
    __m512 fac_vq_simd;

    for (size_t i = 0; i < Degree; i += 16) {
        result_simd = _mm512_load_ps(&result[i]);
        triple_x_simd = _mm512_load_ps(&triple_x[i]);
        fac_dq_simd = _mm512_load_ps(&fac_dq[i]);
        fac_vq_simd = _mm512_load_ps(&fac_vq[i]);

        triple_x_simd = _mm512_add_ps(triple_x_simd, sqr_y_simd);

        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, width_simd);
        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, result_simd);

        fac_vq_simd = _mm512_fmadd_ps(fac_vq_simd, vl_simd, triple_x_simd);

        triple_x_simd = _mm512_add_ps(fac_dq_simd, fac_vq_simd);
        _mm512_store_ps(&appro_dist[i], triple_x_simd);
    }
#elif defined(__AVX2__)
    const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
    const __m256 width_simd = _mm256_set1_ps(width);
    const __m256 vl_simd = _mm256_set1_ps(vl);

    __m256 result_simd;
    __m256 triple_x_simd;
    __m256 fac_dq_simd;
    __m256 fac_vq_simd;

    for (size_t i = 0; i < L; i += 8) {
        result_simd = _mm256_load_ps(&result[i]);
        triple_x_simd = _mm256_load_ps(&triple_x[i]);
        fac_dq_simd = _mm256_load_ps(&fac_dq[i]);
        fac_vq_simd = _mm256_load_ps(&fac_vq[i]);

        triple_x_simd = _mm256_add_ps(triple_x_simd, sqr_y_simd);

        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, width_simd);
        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, result_simd);

        fac_vq_simd = _mm256_mul_ps(fac_vq_simd, vl_simd);

        triple_x_simd =
            _mm256_add_ps(_mm256_add_ps(triple_x_simd, fac_dq_simd), fac_vq_simd);
        _mm256_store_ps(&appro_dist[i], triple_x_simd);
    }
    return;
#else
    std::cerr << "SIMD (AVX512 or AVX2) REQUIRED!\n";
    abort();
#endif
}
}  // namespace symqg
