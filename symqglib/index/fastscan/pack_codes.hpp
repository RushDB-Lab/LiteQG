// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace symqg {
template <typename T, class TA>
static inline void get_column(
    T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest
) {
    for (size_t k = 0; k < dest.size(); k++) {
        if (k + row < rows) {
            dest[k] = src[(k + row) * cols + col];
        } else {
            dest[k] = 0;
        }
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
static void pack_codes(
    size_t padded_dim, const uint8_t* codes, size_t ncode, uint8_t* blocks
) {
    size_t ncode_pad = (ncode + 31) / 32 * 32;
    size_t num_codebook = padded_dim / 4;
    const uint8_t bbs = 32;
    std::memset(blocks, 0, ncode_pad * num_codebook / 2);

    constexpr std::array<int, 16> kPerm0 = {
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
    };
    uint8_t* codes2 = blocks;
    for (size_t blk = 0; blk < ncode_pad; blk += bbs) {
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
    size_t ncode_pad = (ncode + 31) / 32 * 32;
    std::vector<uint8_t> binary_code_8bit(ncode_pad * padded_dim / 8);
    std::memcpy(
        binary_code_8bit.data(), binary_code, ncode * padded_dim / 64 * sizeof(uint64_t)
    );

    for (size_t i = 0; i < ncode; ++i) {
        for (size_t j = 0; j < padded_dim / 64; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                std::swap(
                    binary_code_8bit[i * padded_dim / 8 + 8 * j + k],
                    binary_code_8bit[i * padded_dim / 8 + 8 * j + 8 - k - 1]
                );
            }
        }
    }

    for (size_t i = 0; i < ncode * padded_dim / 8; ++i) {
        uint8_t val = binary_code_8bit[i];
        uint8_t val_hi = (val >> 4);
        uint8_t val_lo = (val & 15);
        binary_code_8bit[i] = (val_lo << 4 | val_hi);
    }
    pack_codes(padded_dim, binary_code_8bit.data(), ncode, blocks);
}
}  // namespace symqg