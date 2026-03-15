#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace symqg {

// Per-vector CAQ factors for asymmetric quantization.
// Distance: est_ip = fac_dot * dot(code, q) + fac_sum * sum_q
//           dist   = o_l2sqr + q_l2sqr - 2 * est_ip
struct CaqFactors {
    float o_l2sqr;   // ||o||²
    float fac_dot;   // rescale * delta
    float fac_sum;   // rescale * (0.5 * delta + vmin)
};

// Encode a single vector using CAQ (Code Adjustment Quantization).
// Codes are uint8 in [0, 255] with asymmetric [vmin, vmax] range per vector.
inline void caq_encode_single(
    const float* __restrict__ vec,
    size_t dim,
    uint8_t* __restrict__ code,
    CaqFactors* __restrict__ factors,
    int adj_rounds = 6
) {
    constexpr int code_max = 255;

    // Step 1: compute vmin, vmax and ||o||²
    float v_min = vec[0], v_max = vec[0];
    float o_l2sqr = 0;
    for (size_t i = 0; i < dim; ++i) {
        v_min = std::min(v_min, vec[i]);
        v_max = std::max(v_max, vec[i]);
        o_l2sqr += vec[i] * vec[i];
    }
    factors->o_l2sqr = o_l2sqr;

    float range = v_max - v_min;
    if (range == 0) {
        std::memset(code, 0, dim);
        factors->fac_dot = 0;
        factors->fac_sum = 0;
        return;
    }

    float delta = range / 256.0f;

    // Step 2: initial LVQ quantization (floor)
    double ip_o_oa = 0;
    double oa_l2sqr = 0;
    double vec_sum = 0;

    for (size_t i = 0; i < dim; ++i) {
        vec_sum += vec[i];
        int c = static_cast<int>((vec[i] - v_min) / delta);
        c = std::clamp(c, 0, code_max);
        code[i] = static_cast<uint8_t>(c);
    }

    // Compute initial <o, o_a> and ||o_a||²
    {
        double ip_o_code = 0;
        uint64_t code_l2sqr = 0;
        uint32_t code_sum = 0;
        for (size_t i = 0; i < dim; ++i) {
            ip_o_code += static_cast<double>(code[i]) * vec[i];
            code_l2sqr += static_cast<uint64_t>(code[i]) * code[i];
            code_sum += code[i];
        }
        double d = delta;
        double vm = v_min;
        ip_o_oa = d * ip_o_code + (vm + 0.5 * d) * vec_sum;
        oa_l2sqr = d * d * code_l2sqr + (d * d + 2 * d * vm) * code_sum
                   + (0.25 * d * d + d * vm + vm * vm) * dim;
    }

    // Step 3: code adjustment — coordinate descent maximizing cosine(o, o_a)
    if (adj_rounds > 0 && oa_l2sqr > 0) {
        constexpr double eps = 1e-8;

        for (int round = 0; round < adj_rounds; ++round) {
            double re_eps = eps * oa_l2sqr;
            int adj_count = 0;

            for (size_t j = 0; j < dim; ++j) {
                double o = vec[j];
                double oa = (code[j] + 0.5) * delta + v_min;
                uint8_t c = code[j];
                double oa_l2sqr_rest = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;

                // try increment
                while (c < code_max) {
                    double new_oa = oa + delta;
                    double new_length = oa_l2sqr_rest + new_oa * new_oa;
                    double new_ip = ip_o_oa + ip_delta;
                    // Accept if cos²(new) > cos²(old) + tolerance
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length
                        >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = new_length;
                    adj_count++;
                }
                // try decrement
                while (c > 0) {
                    double new_oa = oa - delta;
                    double new_length = oa_l2sqr_rest + new_oa * new_oa;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length
                        >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = new_length;
                    adj_count++;
                }
                code[j] = c;
            }

            if (adj_count == 0)
                break;

            // Correction pass to avoid floating-point drift
            ip_o_oa = 0;
            oa_l2sqr = 0;
            for (size_t j = 0; j < dim; ++j) {
                double oa = (code[j] + 0.5) * delta + v_min;
                ip_o_oa += oa * vec[j];
                oa_l2sqr += oa * oa;
            }
        }
    }

    // Step 4: compute asymmetric factors
    // rescale = ||o||² / <o, o_a>
    // est_ip = rescale * <o_a, q> = rescale * (delta * dot(code, q) + (0.5*delta + vmin) * sum_q)
    //        = fac_dot * dot(code, q) + fac_sum * sum_q
    if (ip_o_oa > 0) {
        double rescale = o_l2sqr / ip_o_oa;
        factors->fac_dot = static_cast<float>(rescale * delta);
        factors->fac_sum = static_cast<float>(rescale * (0.5 * delta + v_min));
    } else {
        factors->fac_dot = 0;
        factors->fac_sum = 0;
    }
}

// Batch quantize all vectors using CAQ (OpenMP parallel)
inline void caq_quantize(
    const float* data,
    size_t dim,
    size_t num,
    uint8_t* codes,
    CaqFactors* factors,
    int adj_rounds = 6
) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num; ++i) {
        caq_encode_single(
            data + i * dim, dim,
            codes + i * dim, &factors[i],
            adj_rounds
        );
    }
}

// SIMD dot product: Σ uint8_code[i] * float_query[i]
inline float caq_dot_u8f32(
    const uint8_t* __restrict__ code,
    const float* __restrict__ query,
    size_t dim
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&code[i]));
        __m512 fval = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8));
        sum = _mm512_fmadd_ps(fval, _mm512_loadu_ps(&query[i]), sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        result += code[i] * query[i];
    }

#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&code[i]));
        __m256 fval = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8));
        sum = _mm256_fmadd_ps(fval, _mm256_loadu_ps(&query[i]), sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    result = _mm_cvtss_f32(s);
    for (; i < dim; ++i) {
        result += code[i] * query[i];
    }

#else
    for (size_t i = 0; i < dim; ++i) {
        result += code[i] * query[i];
    }
#endif
    return result;
}

// Estimate L2² distance using asymmetric CAQ codes.
//   <o, q> ≈ fac_dot * dot(code, q) + fac_sum * sum_q
//   dist = o_l2sqr + q_l2sqr - 2 * est_ip
inline float caq_l2_estimate(
    const float* __restrict__ query,
    const uint8_t* __restrict__ code,
    const CaqFactors* __restrict__ factors,
    float q_l2sqr,
    float sum_q,
    size_t dim
) {
    float raw_dot = caq_dot_u8f32(code, query, dim);
    float est_ip = factors->fac_dot * raw_dot + factors->fac_sum * sum_q;
    float dist = factors->o_l2sqr + q_l2sqr - 2.0f * est_ip;
    return std::max(0.0f, dist);
}

// Per-neighbor SAQ factors for fastscan distance estimation.
// dist = sqr_y + fac_a + fac_b * width * fs_result + fac_c * vl_half + fac_d * sum_q
struct SaqFactors {
    float fac_a;    // ||residual||² + 2<centroid, residual>
    float fac_b;    // -2 * rescale * delta
    float fac_c;    // -2 * rescale * delta * sum_code
    float fac_d;    // -2 * rescale * (0.5*delta + vmin)
};

// Encode a single vector using 4-bit CAQ (16 quantization levels).
// Returns codes (0-15) and factors needed for SAQ distance estimation.
inline void caq_encode_4bit(
    const float* __restrict__ vec,
    size_t dim,
    uint8_t* __restrict__ code,
    float* __restrict__ o_l2sqr_out,
    float* __restrict__ delta_out,
    float* __restrict__ vmin_out,
    float* __restrict__ rescale_out,
    float* __restrict__ sum_code_out,
    int adj_rounds = 6
) {
    constexpr int code_max = 15;

    float v_min = vec[0], v_max = vec[0];
    float o_l2sqr = 0;
    for (size_t i = 0; i < dim; ++i) {
        v_min = std::min(v_min, vec[i]);
        v_max = std::max(v_max, vec[i]);
        o_l2sqr += vec[i] * vec[i];
    }
    *o_l2sqr_out = o_l2sqr;
    *vmin_out = v_min;

    float range = v_max - v_min;
    if (range == 0) {
        std::memset(code, 0, dim);
        *delta_out = 0;
        *rescale_out = 0;
        *sum_code_out = 0;
        return;
    }

    float delta = range / 16.0f;
    *delta_out = delta;

    // Initial quantization
    double vec_sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        vec_sum += vec[i];
        int c = static_cast<int>((vec[i] - v_min) / delta);
        c = std::clamp(c, 0, code_max);
        code[i] = static_cast<uint8_t>(c);
    }

    // Compute initial <o, o_a> and ||o_a||²
    double ip_o_oa = 0;
    double oa_l2sqr = 0;
    {
        double ip_o_code = 0;
        uint64_t code_l2sqr = 0;
        uint32_t code_sum = 0;
        for (size_t i = 0; i < dim; ++i) {
            ip_o_code += static_cast<double>(code[i]) * vec[i];
            code_l2sqr += static_cast<uint64_t>(code[i]) * code[i];
            code_sum += code[i];
        }
        double d = delta;
        double vm = v_min;
        ip_o_oa = d * ip_o_code + (vm + 0.5 * d) * vec_sum;
        oa_l2sqr = d * d * code_l2sqr + (d * d + 2 * d * vm) * code_sum
                   + (0.25 * d * d + d * vm + vm * vm) * dim;
    }

    // Code adjustment — coordinate descent maximizing cosine(o, o_a)
    if (adj_rounds > 0 && oa_l2sqr > 0) {
        constexpr double eps = 1e-8;
        for (int round = 0; round < adj_rounds; ++round) {
            double re_eps = eps * oa_l2sqr;
            int adj_count = 0;
            for (size_t j = 0; j < dim; ++j) {
                double o = vec[j];
                double oa = (code[j] + 0.5) * delta + v_min;
                uint8_t c = code[j];
                double oa_l2sqr_rest = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;
                while (c < code_max) {
                    double new_oa = oa + delta;
                    double new_length = oa_l2sqr_rest + new_oa * new_oa;
                    double new_ip = ip_o_oa + ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length
                        >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = new_length;
                    adj_count++;
                }
                while (c > 0) {
                    double new_oa = oa - delta;
                    double new_length = oa_l2sqr_rest + new_oa * new_oa;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length
                        >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = new_length;
                    adj_count++;
                }
                code[j] = c;
            }
            if (adj_count == 0)
                break;
            ip_o_oa = 0;
            oa_l2sqr = 0;
            for (size_t j = 0; j < dim; ++j) {
                double oa = (code[j] + 0.5) * delta + v_min;
                ip_o_oa += oa * vec[j];
                oa_l2sqr += oa * oa;
            }
        }
    }

    // Compute rescale and sum_code
    float sum_code = 0;
    for (size_t i = 0; i < dim; ++i) {
        sum_code += code[i];
    }
    *sum_code_out = sum_code;
    *rescale_out = (ip_o_oa > 0) ? static_cast<float>(o_l2sqr / ip_o_oa) : 0;
}

}  // namespace symqg
