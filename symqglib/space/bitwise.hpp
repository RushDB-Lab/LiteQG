#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

#include <cstdint>

#include "../common.hpp"

namespace symqg::space {

inline auto popcount(size_t dim, const uint64_t* __restrict__ data) -> size_t {
    size_t ret = 0;
    for (size_t i = 0; i < dim / 64; ++i) {
        ret += __builtin_popcountll((*data));
        ++data;
    }
    return ret;
}

/* Change 0/1 mat to uint64 */
inline void pack_binary(const RowMatrix<int>& bin_x_rotated, uint64_t* binary) {
    size_t rows = bin_x_rotated.rows();
    size_t cols = bin_x_rotated.cols();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; col += 64) {
            uint64_t cur = 0;
            for (size_t i = 0; i < 64; ++i) {
                cur |=
                    (static_cast<uint64_t>(
                         bin_x_rotated(static_cast<long>(row), static_cast<long>(col + i))
                     )
                     << (63 - i));
            }
            *binary = cur;
            ++binary;
        }
    }
}

}  // namespace symqg::space