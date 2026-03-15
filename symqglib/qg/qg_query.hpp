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
    float width_ = 0;
    float vl_half_ = 0;
    float sum_q_float_ = 0;

    // Pre-allocated buffers (avoid per-query malloc)
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
    std::vector<float, memory::AlignedAllocator<float>> rd_query_;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query_;

   public:
    explicit QGQuery(size_t padded_dim)
        : padded_dim_(padded_dim)
        , lut_(padded_dim * 16)
        , rd_query_(padded_dim)
        , byte_query_(padded_dim) {}

    void prepare(const float* query, const PCARotator& rotator, const QGScanner& scanner) {
        query_data_ = query;

        // PCA rotate
        std::fill(rd_query_.begin(), rd_query_.end(), 0.0f);
        rotator.rotate(query, rd_query_.data());

        // Sum of rotated query (float, exact)
        sum_q_float_ = 0;
        for (size_t d = 0; d < padded_dim_; ++d) {
            sum_q_float_ += rd_query_[d];
        }

        // 4-bit quantize
        float lo, hi;
        scalar::data_range(rd_query_.data(), padded_dim_, lo, hi);
        width_ = (hi - lo) / ((1 << kSaqQueryBits) - 1);
        if (width_ < 1e-10f) width_ = 1e-10f;
        int32_t sumq_unused;
        scalar::quantize(byte_query_.data(), rd_query_.data(), padded_dim_, lo, width_, sumq_unused);

        vl_half_ = lo - 0.5f * width_;

        // Build LUT
        scanner.pack_lut(byte_query_.data(), lut_.data());
    }

    [[nodiscard]] float width() const { return width_; }
    [[nodiscard]] float vl_half() const { return vl_half_; }
    [[nodiscard]] float sum_q_float() const { return sum_q_float_; }
    [[nodiscard]] const uint8_t* lut() const { return lut_.data(); }
    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg
