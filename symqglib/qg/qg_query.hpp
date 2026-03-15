#pragma once

#include <cstdint>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./qg_scanner.hpp"

namespace symqg {

// SAQ query bits: 4-bit query quantization (max LUT entry = 15*16 = 240, fits uint8)
constexpr int kSaqQueryBits = 4;

class QGQuery {
   private:
    const float* query_data_ = nullptr;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
    size_t padded_dim_ = 0;
    float width_ = 0;
    float lower_val_ = 0;
    float upper_val_ = 0;
    int32_t sumq_ = 0;
    float sum_q_float_ = 0;
    float vl_half_ = 0;

   public:
    explicit QGQuery(const float* q, size_t padded_dim)
        : query_data_(q)
        , lut_(padded_dim * 16)  // 16 entries per dim for 4-bit LUT
        , padded_dim_(padded_dim) {}

    void query_prepare(const PCARotator& rotator, const QGScanner& scanner) {
        // PCA rotate query (writes dimension_ elements, rest stays zero)
        std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_, 0.0f);
        rotator.rotate(query_data_, rd_query.data());

        // Compute sum of rotated query (float) for SAQ distance formula
        sum_q_float_ = 0;
        for (size_t d = 0; d < padded_dim_; ++d) {
            sum_q_float_ += rd_query[d];
        }

        // 4-bit quantize rotated query
        std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
        scalar::data_range(rd_query.data(), padded_dim_, lower_val_, upper_val_);
        width_ = (upper_val_ - lower_val_) / ((1 << kSaqQueryBits) - 1);
        if (width_ < 1e-10f) width_ = 1e-10f;
        scalar::quantize(
            byte_query.data(), rd_query.data(), padded_dim_, lower_val_, width_, sumq_
        );

        // vl_half = lower_val - 0.5 * width (reconstruction offset)
        vl_half_ = lower_val_ - 0.5f * width_;

        // Build 4-bit LUT
        scanner.pack_lut(byte_query.data(), lut_.data());
    }

    [[nodiscard]] const float& width() const { return width_; }
    [[nodiscard]] const float& vl_half() const { return vl_half_; }
    [[nodiscard]] float sum_q_float() const { return sum_q_float_; }

    [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>>& lut(
    ) const {
        return lut_;
    }

    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg
