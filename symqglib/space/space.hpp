#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

#include "../common.hpp"
#include "../utils/tools.hpp"

namespace symqg::space {
/**
 * @brief Use the data point with mininum distance to medioid as the entry point
 *
 * @param data          data vectors
 * @param medioid       medioid vector
 * @param N             num of data vectors
 * @param dim           dimension of data
 * @param dist_func_
 * @return PID
 */
inline auto compute_entrypoint(
    const float* data,
    const float* medioid,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    const DistFunc<float>& dist_func_
) {
    std::vector<Candidate<float>> best_entries(num_threads, Candidate(0, FLT_MAX));
    size_t batchsize = div_round_up(num_points, num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = std::min(i * batchsize, num_points);
        size_t stop = std::min((i + 1) * batchsize, num_points);
        Candidate<float>& cur_entry = best_entries[i];
        const float* batch_data = data + dim * start;

        for (size_t cur_id = start; cur_id < stop; ++cur_id) {
            float distance = dist_func_(batch_data, medioid, dim);
            if (distance < cur_entry.distance) {
                cur_entry.id = static_cast<PID>(cur_id);
                cur_entry.distance = distance;
            }
            batch_data += dim;
        }
    }

    PID entry = 0;
    float min_dist = FLT_MAX;
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            entry = candi.id;
            min_dist = candi.distance;
        }
    }
    return entry;
}

inline auto compute_medioid(
    const float* data, size_t num_points, size_t dim, size_t num_threads
) {
    std::vector<std::vector<double>> all_results(num_threads, std::vector<double>(dim, 0));
    size_t batchsize = div_round_up(num_points, num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = std::min(i * batchsize, num_points);
        size_t stop = std::min((i + 1) * batchsize, num_points);

        std::vector<double>& cur_results = all_results[i];
        const float* batch_data = data + dim * start;
        for (size_t j = start; j < stop; ++j) {
            for (size_t k = 0; k < dim; ++k) {
                cur_results[k] += batch_data[k];
            }
            batch_data += dim;
        }
    }

    std::vector<double> medioid_d(dim, 0);
    std::vector<float> medioid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            medioid_d[i] += one_res[i];
        }
    }
    for (size_t i = 0; i < dim; ++i) {
        medioid[i] = static_cast<float>(medioid_d[i] / static_cast<double>(num_points));
    }

    return medioid;
}
}  // namespace symqg::space