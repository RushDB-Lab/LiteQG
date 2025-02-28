#pragma once
#include <vector>
#include <cstdint>

namespace symqg {
struct SQ8Params {
    float min;
    float scale;
};

void sq8_quantize(const float* data, size_t dim, size_t num, uint8_t* quantized, SQ8Params* params) {
    for (size_t d = 0; d < dim; ++d) {
        float min = data[d], max = data[d];
        for (size_t i = 1; i < num; ++i) {
            float val = data[i * dim + d];
            min = std::min(min, val);
            max = std::max(max, val);
        }
        params[d].min = min;
        params[d].scale = (max - min) / 255.0f;
        for (size_t i = 0; i < num; ++i) {
            float val = data[i * dim + d];
            quantized[i * dim + d] = static_cast<uint8_t>(std::round((val - min) / params[d].scale));
        }
    }
}

void sq8_dequantize(const uint8_t* quantized, size_t dim, float* data, const SQ8Params* params) {
    for (size_t i = 0; i < dim; ++i) {
        data[i] = params[i].min + quantized[i] * params[i].scale;
    }
}
}