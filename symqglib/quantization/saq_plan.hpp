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

// Mixed-bitwidth plan: 4-bit SAQ + 1-bit binary + 0-bit discarded dims.
// All dims are PCA-ordered (descending variance).
struct MixedPlan {
    QuantPlan segments;
    size_t saq_dim = 0;
    size_t bin_dim = 0;
    size_t bin_dim_actual = 0;
    size_t saq_codebooks = 0;
    size_t bin_codebooks = 0;
    size_t saq_fascscan_dim = 0;
    size_t bin_fascscan_dim = 0;
};

// Codebook-budget DP for mixed-bitwidth allocation.
//
// Allocates 4-bit, 1-bit, or 0-bit per segment to minimize total quantization error
// under a codebook budget constraint.
//
// Codebook cost:
//   bits=4: seg_dims codebooks (1 per dim)
//   bits=1: seg_dims/4 codebooks (4 binary dims → 1 codebook)
//   bits=0: 0 codebooks
//
// Error model: var_sum / 2^b  (var_sum for b=0)
//
// Alignment:
//   4-bit segments: multiples of 4 dims
//   1-bit segments: multiples of 16 dims (AVX512: 64 bytes = 16 codebooks per iteration)
//
// Input:
//   eigenvalues: per-dim variances (descending), length = dim
//   dim: number of dimensions
//   codebook_budget: max codebooks (e.g. 32)
//
// Output: MixedPlan
inline MixedPlan segment_dp_codebook(
    const float* eigenvalues,
    size_t dim,
    size_t codebook_budget,
    size_t max_segments = 16
) {
    // DP grid granularity: 4 dims per position (smallest segment unit)
    constexpr size_t kGranularity = 4;
    size_t num_positions = dim / kGranularity;

    // Allowed bit widths: {0, 1, 4}
    struct BitChoice {
        uint8_t bits;
        size_t min_segment_positions;  // minimum segment size in positions
    };
    // Only 4-bit and 0-bit: PCA rotation makes 1-bit binary unreliable
    // (PCA concentrates variance, making sign bits in low-variance dims noise)
    static constexpr BitChoice kChoices[] = {
        {4, 1},    // 4-bit: 4 dims minimum, codebook cost = seg_dims
        {0, 1},    // 0-bit: 4 dims minimum, codebook cost = 0
    };
    static constexpr size_t kNumChoices = 2;

    size_t S = max_segments + 1;
    size_t P = num_positions + 1;

    struct DPChoice {
        size_t start_pos;
        uint8_t bits;
    };

    std::vector<double> dp(S * P, 1e30);
    std::vector<size_t> cb_used(S * P, 0);
    std::vector<DPChoice> choice(S * P);

    dp[0 * P + 0] = 0;
    cb_used[0 * P + 0] = 0;

    // Prefix sums of eigenvalues at kGranularity boundaries
    std::vector<double> prefix(P, 0);
    for (size_t p = 0; p < num_positions; ++p) {
        prefix[p + 1] = prefix[p];
        for (size_t d = 0; d < kGranularity; ++d) {
            prefix[p + 1] += eigenvalues[p * kGranularity + d];
        }
    }

    for (size_t s = 1; s <= max_segments; ++s) {
        for (size_t p = 1; p <= num_positions; ++p) {
            for (size_t bi = 0; bi < kNumChoices; ++bi) {
                uint8_t b = kChoices[bi].bits;
                size_t min_pos = kChoices[bi].min_segment_positions;

                // For 1-bit: segment must be multiple of 4 positions (16 dims)
                // For all: segment must be at least min_pos positions
                size_t pos_step = (b == 1) ? 4 : 1;

                for (size_t q = (s == 1 ? 0 : 1); q < p; ++q) {
                    size_t seg_positions = p - q;
                    if (seg_positions < min_pos) continue;
                    // 1-bit requires multiple of 16 positions (64 dims)
                    if (b == 1 && seg_positions % 16 != 0) continue;

                    size_t seg_dims = seg_positions * kGranularity;
                    double seg_var_sum = prefix[p] - prefix[q];

                    // Codebook cost
                    size_t seg_cb;
                    if (b == 4) seg_cb = seg_dims;         // 1 codebook per dim
                    else if (b == 1) seg_cb = seg_dims / 4; // 4 binary dims → 1 codebook
                    else seg_cb = 0;                        // 0-bit: no codebooks

                    size_t prev_cb = cb_used[(s - 1) * P + q];
                    size_t new_total_cb = prev_cb + seg_cb;
                    if (new_total_cb > codebook_budget) continue;

                    // Error: var_sum / 2^b  (b=0 → var_sum, i.e. no information)
                    double error;
                    if (b == 0) error = seg_var_sum;
                    else error = seg_var_sum / static_cast<double>(1ULL << b);

                    double total_error = dp[(s - 1) * P + q] + error;

                    if (total_error < dp[s * P + p]) {
                        dp[s * P + p] = total_error;
                        cb_used[s * P + p] = new_total_cb;
                        choice[s * P + p] = {q, b};
                    }
                }
            }
        }
    }

    // Find best
    double best_error = 1e30;
    size_t best_s = 0;
    for (size_t s = 1; s <= max_segments; ++s) {
        if (dp[s * P + num_positions] < best_error) {
            best_error = dp[s * P + num_positions];
            best_s = s;
        }
    }

    MixedPlan plan;
    plan.saq_dim = 0;
    plan.bin_dim = 0;
    plan.bin_dim_actual = 0;

    if (best_s == 0) {
        // Fallback: all 1-bit
        size_t bin = std::min(dim, codebook_budget * 4);
        bin = (bin / 16) * 16;  // align to 16
        plan.segments.push_back({0, bin, 1});
        plan.bin_dim = bin;
        plan.bin_dim_actual = bin;
        if (bin < dim) {
            plan.segments.push_back({bin, dim - bin, 0});
        }
    } else {
        // Backtrack
        size_t cur_pos = num_positions;
        for (size_t s = best_s; s >= 1; --s) {
            auto& ch = choice[s * P + cur_pos];
            size_t seg_offset = ch.start_pos * kGranularity;
            size_t seg_len = (cur_pos - ch.start_pos) * kGranularity;
            plan.segments.push_back({seg_offset, seg_len, ch.bits});
            if (ch.bits == 4) plan.saq_dim += seg_len;
            else if (ch.bits == 1) plan.bin_dim += seg_len;
            cur_pos = ch.start_pos;
        }
        std::reverse(plan.segments.begin(), plan.segments.end());
    }

    // Pad bin_dim to multiple of 64 (required by pack_binary / pack_codes)
    // The extra padded dims have zero eigenvalue (zero residual) so no impact on accuracy.
    plan.bin_dim_actual = plan.bin_dim;
    if (plan.bin_dim > 0 && plan.bin_dim % 64 != 0) {
        plan.bin_dim = (plan.bin_dim + 63) & ~63UL;
    }

    // Derived fields
    plan.saq_codebooks = plan.saq_dim;
    plan.bin_codebooks = plan.bin_dim / 4;
    plan.saq_fascscan_dim = plan.saq_dim * 4;
    plan.bin_fascscan_dim = plan.bin_dim;

    // Debug output
    std::cout << "\tMixed-bitwidth plan (" << plan.segments.size() << " segments, "
              << plan.saq_codebooks + plan.bin_codebooks << "/" << codebook_budget
              << " codebooks):\n";
    for (auto& seg : plan.segments) {
        std::cout << "\t  dims [" << seg.dim_offset << ", "
                  << seg.dim_offset + seg.dim_len << ") → "
                  << static_cast<int>(seg.bits) << "-bit\n";
    }
    std::cout << "\t  SAQ: " << plan.saq_dim << " dims (" << plan.saq_codebooks << " CB)"
              << "  Binary: " << plan.bin_dim << " dims (" << plan.bin_codebooks << " CB)"
              << "\n";

    return plan;
}

}  // namespace symqg
