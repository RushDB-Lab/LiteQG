#pragma once

#include <cstdint>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./qg_scanner.hpp"

namespace symqg {

constexpr int kSaqQueryBits = 4;

class QGQuery {
   private:
    const float* query_data_ = nullptr;
    size_t padded_dim_ = 0;
    size_t split_dim_ = 0;
    float width_ = 0;
    float vl_half_ = 0;
    float sum_q_seg0_ = 0;
    float sum_q_seg1_ = 0;

    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_seg0_;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_seg1_;
    std::vector<float, memory::AlignedAllocator<float>> rd_query_;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query_;

   public:
    explicit QGQuery(size_t padded_dim, size_t split_dim)
        : padded_dim_(padded_dim)
        , split_dim_(split_dim)
        , lut_seg0_(split_dim * 16)
        , lut_seg1_((padded_dim - split_dim) / 4 * 16)
        , rd_query_(padded_dim)
        , byte_query_(padded_dim) {}

    void prepare(const float* query, const PCARotator& rotator, const QGScanner& scanner) {
        query_data_ = query;

        std::fill(rd_query_.begin(), rd_query_.end(), 0.0f);
        rotator.rotate(query, rd_query_.data());

        // Per-segment sum of rotated query (float)
        sum_q_seg0_ = 0;
        for (size_t d = 0; d < split_dim_; ++d) {
            sum_q_seg0_ += rd_query_[d];
        }
        sum_q_seg1_ = 0;
        for (size_t d = split_dim_; d < padded_dim_; ++d) {
            sum_q_seg1_ += rd_query_[d];
        }

        // 4-bit quantize full query
        float lo, hi;
        scalar::data_range(rd_query_.data(), padded_dim_, lo, hi);
        width_ = (hi - lo) / ((1 << kSaqQueryBits) - 1);
        if (width_ < 1e-10f) width_ = 1e-10f;
        int32_t sumq_unused;
        scalar::quantize(byte_query_.data(), rd_query_.data(), padded_dim_, lo, width_, sumq_unused);

        vl_half_ = lo - 0.5f * width_;

        // Seg0 LUT: 4-bit (1 dim per nibble)
        pack_lut_4bit(split_dim_, byte_query_.data(), lut_seg0_.data());
        // Seg1 LUT: 1-bit (4 dims per nibble, baseline-style)
        size_t seg1_dim = padded_dim_ - split_dim_;
        pack_lut_impl(seg1_dim, &byte_query_[split_dim_], lut_seg1_.data());
    }

    [[nodiscard]] float width() const { return width_; }
    [[nodiscard]] float vl_half() const { return vl_half_; }
    [[nodiscard]] float sum_q_seg0() const { return sum_q_seg0_; }
    [[nodiscard]] float sum_q_seg1() const { return sum_q_seg1_; }
    [[nodiscard]] const uint8_t* lut_seg0() const { return lut_seg0_.data(); }
    [[nodiscard]] const uint8_t* lut_seg1() const { return lut_seg1_.data(); }
    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg
