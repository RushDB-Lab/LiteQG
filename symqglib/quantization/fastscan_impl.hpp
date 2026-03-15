// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

constexpr static std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};

template <typename T, class TA>
static inline void get_column(
    const T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest
) {
    size_t k = 0;
    size_t max_k = std::min(rows - row, dest.size());
    for (; k < max_k; ++k) {
        dest[k] = src[((k + row) * cols) + col];
    }
    if (k < dest.size()) {
        std::fill(dest.begin() + k, dest.end(), 0);
    }
}

/**
 * @brief ack 32 quantization codes in a batch from the quantization codes represented by a
 * sequence of uint8_t variables
 *
 * @param padded_dim    dim of vector
 * @param codes         input quantization codes
 * @param ncode         number of vectors (codes)
 * @param blocks        padded results (block of codes)
 */
static inline void pack_codes_helper(
    size_t padded_dim, const uint8_t* codes, size_t ncode, uint8_t* blocks
) {
    size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
    size_t num_codebook = padded_dim / 4;
    std::memset(blocks, 0, ncode_pad * num_codebook / 2);

    uint8_t* codes2 = blocks;
    for (size_t blk = 0; blk < ncode_pad; blk += kBatchSize) {
        // enumerate i
        for (size_t i = 0; i < num_codebook; i += 2) {
            std::array<uint8_t, 32> col;
            std::array<uint8_t, 32> col_lo;
            std::array<uint8_t, 32> col_hi;
            get_column(codes, ncode, num_codebook / 2, blk, i / 2, col);
            for (int j = 0; j < 32; j++) {
                col_lo[j] = col[j] & 15;
                col_hi[j] = col[j] >> 4;
            }
            for (int j = 0; j < 16; j++) {
                auto val0 = col_lo[kPerm0[j]] | (col_lo[kPerm0[j] + 16] << 4);
                auto val1 = col_hi[kPerm0[j]] | (col_hi[kPerm0[j] + 16] << 4);
                codes2[j] = val0;
                codes2[j + 16] = val1;
            }
            codes2 += 32;
        }
    }
}

inline void pack_codes(
    size_t padded_dim, const uint64_t* binary_code, size_t ncode, uint8_t* blocks
) {
    size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
    std::vector<uint8_t> binary_code_8bit(ncode_pad * padded_dim / 8);
    std::memcpy(
        binary_code_8bit.data(), binary_code, ncode * padded_dim / 64 * sizeof(uint64_t)
    );

    for (size_t i = 0; i < ncode; ++i) {
        for (size_t j = 0; j < padded_dim / 64; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                std::swap(
                    binary_code_8bit[(i * padded_dim / 8) + (8 * j) + k],
                    binary_code_8bit[(i * padded_dim / 8) + (8 * j) + 8 - k - 1]
                );
            }
        }
    }

    for (size_t i = 0; i < ncode * padded_dim / 8; ++i) {
        uint8_t val = binary_code_8bit[i];
        uint8_t val_hi = (val >> 4);
        uint8_t val_lo = (val & 15);
        binary_code_8bit[i] = (val_lo << 4) | val_hi;
    }
    pack_codes_helper(padded_dim, binary_code_8bit.data(), ncode, blocks);
}

// use fast scan to accumulate one block, dim % 16 == 0
inline void accumulate_impl(
    size_t dim,
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result
) {
    size_t code_length = dim << 2;
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

    for (size_t i = 0; i < code_length; i += 64) {
        c = _mm512_loadu_si512(&codes[i]);
        lut = _mm512_loadu_si512(&LUT[i]);
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

    _mm512_storeu_si512(result, ret);

    // __m256i res0 = _mm256_add_epi16(_mm512_castsi512_si256(accu0),
    // _mm512_extracti64x4_epi64(accu0, 1));
    // __m256i res1 = _mm256_add_epi16(_mm512_castsi512_si256(accu1),
    // _mm512_extracti64x4_epi64(accu1, 1));

    // res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    // __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(res0, res1,
    // 0x21),_mm256_blend_epi32(res0, res1, 0xF0)); _mm256_storeu_si256((__m256i*)result,
    // dis0);

    // __m256i res2 = _mm256_add_epi16(_mm512_castsi512_si256(accu2),
    // _mm512_extracti64x4_epi64(accu2, 1));
    // __m256i res3 = _mm256_add_epi16(_mm512_castsi512_si256(accu3),
    // _mm512_extracti64x4_epi64(accu3, 1));

    // res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));
    // __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(res2, res3,
    // 0x21),_mm256_blend_epi32(res2, res3, 0xF0));
    // _mm256_storeu_si256((__m256i*)&result[16], dis1);

#elif defined(__AVX2__)
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < code_length; i += 64) {
        c = _mm256_loadu_si256((__m256i*)&codes[i]);
        lut = _mm256_loadu_si256((__m256i*)&LUT[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_loadu_si256((__m256i*)&codes[i + 32]);
        lut = _mm256_loadu_si256((__m256i*)&LUT[i + 32]);
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
    _mm256_storeu_si256((__m256i*)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_storeu_si256((__m256i*)&result[16], dis1);
#else
    std::cerr << "no avx simd supported!\n";
    abort();
#endif
}

// pack lut for fast scan, dim % 4 == 0
inline void pack_lut_impl(
    size_t dim, const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT
) {
    size_t num_codebook = dim >> 2;
    for (size_t i = 0; i < num_codebook; ++i) {
        LUT[0] = 0;
        for (int j = 1; j < 16; ++j) {
            LUT[j] = LUT[j - LOWBIT(j)] + byte_query[kPos[j]];
        }
        LUT += 16;
        byte_query += 4;
    }
}

// --- SAQ 4-bit fastscan support ---

// Build LUT for 4-bit data codes × 4-bit query values.
// Each dimension is its own codebook with 16 entries.
// LUT[d*16 + c] = c * byte_query[d], max = 15*16 = 240, fits uint8.
// Total LUT size: dim * 16 bytes.
inline void pack_lut_4bit(
    size_t dim, const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT
) {
    for (size_t d = 0; d < dim; ++d) {
        uint8_t q = byte_query[d];
        for (int c = 0; c < 16; ++c) {
            LUT[c] = static_cast<uint8_t>(c * q);
        }
        LUT += 16;
    }
}

// Pack 4-bit codes into fastscan format.
// Input: codes_per_dim[ncode * dim], each byte is one 4-bit code value (0-15).
// Uses pack_codes_helper with fascscan_dim = dim * 4 (1 codebook per dim).
inline void pack_codes_4bit(
    size_t dim, const uint8_t* codes_per_dim, size_t ncode, uint8_t* blocks
) {
    size_t bytes_per_vec = dim / 2;
    size_t ncode_pad = (ncode + 31) & ~31;
    std::vector<uint8_t> packed(ncode_pad * bytes_per_vec, 0);
    for (size_t v = 0; v < ncode; ++v) {
        for (size_t d = 0; d < dim; d += 2) {
            uint8_t lo = codes_per_dim[v * dim + d] & 0x0F;
            uint8_t hi = codes_per_dim[v * dim + d + 1] & 0x0F;
            packed[v * bytes_per_vec + d / 2] = (hi << 4) | lo;
        }
    }
    pack_codes_helper(dim * 4, packed.data(), ncode, blocks);
}

}  // namespace symqg