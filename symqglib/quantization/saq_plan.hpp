#pragma once

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace symqg {

// A segment in the quantization plan: [dim_offset, dim_offset + dim_len) with `bits` per dim.
struct QuantSegment {
    size_t dim_offset;
    size_t dim_len;
    uint8_t bits;  // 1, 2, 4, or 8
};

using QuantPlan = std::vector<QuantSegment>;

// Allowed bit widths for SAQ segmentation (must evenly divide into 4-bit fastscan).
static constexpr uint8_t kAllowedBits[] = {1, 2, 4, 8};
static constexpr size_t kNumBitChoices = 4;

// Minimum segment length in dimensions (for SIMD alignment).
// fastscan operates on groups of 4 dims, so segments must be multiples of 4.
static constexpr size_t kDimPaddingSize = 4;

// Dynamic programming for optimal dimension segmentation.
// Allocates variable bits per segment to minimize total quantization error
// under a total bit budget constraint.
//
// Error model: for a segment [d_start, d_start+len) with b bits/dim,
//   error = sum(eigenvalues[d_start..d_start+len)) / 2^b
//
// Input:
//   eigenvalues: per-dim variances (descending), length = dim
//   dim: number of dimensions
//   avg_bits: average bits per dimension (total budget = avg_bits * dim)
//   max_segments: maximum number of segments
//
// Output: QuantPlan with segments covering all dimensions
inline QuantPlan segment_dp(
    const float* eigenvalues,
    size_t dim,
    float avg_bits,
    size_t max_segments = 16
) {
    size_t total_budget = static_cast<size_t>(avg_bits * dim);
    size_t num_positions = dim / kDimPaddingSize;  // number of possible split points

    // dp[s][p] = minimum error using s segments to cover dimensions [0, p*kDimPaddingSize)
    // choice[s][p] = {start_position, bits} for backtracking
    struct DPChoice {
        size_t start_pos;
        uint8_t bits;
        size_t bits_used;
    };

    size_t S = max_segments + 1;
    size_t P = num_positions + 1;

    std::vector<double> dp(S * P, 1e30);
    std::vector<DPChoice> choice(S * P);
    std::vector<size_t> bits_used(S * P, 0);

    dp[0 * P + 0] = 0;
    bits_used[0 * P + 0] = 0;

    // Precompute prefix sums of eigenvalues at kDimPaddingSize boundaries
    std::vector<double> prefix(P, 0);
    for (size_t p = 0; p < num_positions; ++p) {
        prefix[p + 1] = prefix[p];
        for (size_t d = 0; d < kDimPaddingSize; ++d) {
            prefix[p + 1] += eigenvalues[p * kDimPaddingSize + d];
        }
    }

    for (size_t s = 1; s <= max_segments; ++s) {
        for (size_t p = 1; p <= num_positions; ++p) {
            // Try all possible segment starts and bit widths
            for (size_t q = (s == 1 ? 0 : 1); q < p; ++q) {
                size_t seg_dims = (p - q) * kDimPaddingSize;
                double seg_var_sum = prefix[p] - prefix[q];

                for (size_t bi = 0; bi < kNumBitChoices; ++bi) {
                    uint8_t b = kAllowedBits[bi];
                    size_t seg_bits = seg_dims * b;
                    size_t prev_bits = bits_used[(s - 1) * P + q];
                    size_t new_total_bits = prev_bits + seg_bits;

                    if (new_total_bits > total_budget) {
                        continue;
                    }

                    double error = seg_var_sum / static_cast<double>(1ULL << b);
                    double total_error = dp[(s - 1) * P + q] + error;

                    if (total_error < dp[s * P + p]) {
                        dp[s * P + p] = total_error;
                        bits_used[s * P + p] = new_total_bits;
                        choice[s * P + p] = {q, b, new_total_bits};
                    }
                }
            }
        }
    }

    // Find best number of segments covering all dimensions
    double best_error = 1e30;
    size_t best_s = 0;
    for (size_t s = 1; s <= max_segments; ++s) {
        if (dp[s * P + num_positions] < best_error) {
            best_error = dp[s * P + num_positions];
            best_s = s;
        }
    }

    if (best_s == 0) {
        // Fallback: single segment with max allowed bits
        QuantPlan plan;
        plan.push_back({0, dim, std::min(static_cast<uint8_t>(8),
                        static_cast<uint8_t>(total_budget / dim))});
        return plan;
    }

    // Backtrack to reconstruct plan
    QuantPlan plan;
    size_t cur_pos = num_positions;
    for (size_t s = best_s; s >= 1; --s) {
        auto& ch = choice[s * P + cur_pos];
        size_t seg_offset = ch.start_pos * kDimPaddingSize;
        size_t seg_len = (cur_pos - ch.start_pos) * kDimPaddingSize;
        plan.push_back({seg_offset, seg_len, ch.bits});
        cur_pos = ch.start_pos;
    }

    std::reverse(plan.begin(), plan.end());

    // Debug output
    std::cout << "\tSegmentation plan (" << plan.size() << " segments, "
              << total_budget << " total bits):\n";
    for (auto& seg : plan) {
        std::cout << "\t  dims [" << seg.dim_offset << ", "
                  << seg.dim_offset + seg.dim_len << ") → "
                  << static_cast<int>(seg.bits) << " bits\n";
    }

    return plan;
}

// Simple uniform plan: all dimensions get the same bit width.
inline QuantPlan uniform_plan(size_t dim, uint8_t bits) {
    QuantPlan plan;
    plan.push_back({0, dim, bits});
    return plan;
}

}  // namespace symqg
