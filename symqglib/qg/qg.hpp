#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>

#include "../common.hpp"
#include "../quantization/caq.hpp"
#include "../quantization/saq_plan.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"

namespace symqg {

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    PID entry_point_ = 0;

    // PCA rotator (Phase 1)
    PCARotator pca_rotator_;

    // Segmentation plan from DP (Phase 2, stored for future use)
    QuantPlan quant_plan_;

    // Per-vector CAQ factors
    std::vector<CaqFactors> caq_factors_;

    // Build-time: raw vectors + neighbor IDs
    //   Layout: [raw_vec (dimension_ floats) | neighbor_IDs (degree_bound_ PIDs)]
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // Query-time: CAQ codes (in rotated space) + neighbor IDs colocated
    //   Layout: [caq_code (dimension_ bytes) | neighbor_IDs (degree_bound_ * 4 bytes)]
    data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>
        qdata_;

    size_t cur_ef_ = 0;
    HashBasedBooleanSet visited_;

    // Offsets for data_ (build-time, float units)
    size_t neighbor_offset_ = 0;
    size_t row_offset_ = 0;

    // Offsets for qdata_ (query-time, byte units)
    size_t q_neighbor_offset_ = 0;
    size_t q_row_offset_ = 0;
    size_t q_prefetch_lines_ = 0;

    void initialize();
    void copy_vectors(const float*);

    // Build-time accessors (data_)
    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }
    [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }
    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }
    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    // Query-time accessors (qdata_, colocated layout)
    [[nodiscard]] const uint8_t* get_qvector(PID data_id) const {
        return &qdata_.at(q_row_offset_ * data_id);
    }
    [[nodiscard]] const PID* get_qneighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &qdata_.at(q_row_offset_ * data_id + q_neighbor_offset_)
        );
    }

    void find_candidates(
        PID, size_t, std::vector<Candidate<float>>&,
        HashBasedBooleanSet&, const std::vector<uint32_t>&
    ) const;
    void update_qg(PID, const std::vector<Candidate<float>>&);
    void finalize_index();

   public:
    explicit QuantizedGraph(size_t, size_t, size_t);

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }
    [[nodiscard]] auto dimension() const { return this->dimension_; }
    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }
    [[nodiscard]] auto entry_point() const { return this->entry_point_; }
    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save_index(const char*) const;
    void load_index(const char*);
    void set_ef(size_t);
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , pca_rotator_(dim)
    , visited_(100) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
    // Copy raw vectors to data_
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }

    // Phase 1: Fit PCA rotation on raw data
    pca_rotator_.fit(data, num_points_, dimension_);

    // Phase 2: Compute segmentation plan (stored for future use)
    constexpr float kAvgBits = 4.0f;
    quant_plan_ = segment_dp(pca_rotator_.eigenvalues().data(), dimension_, kAvgBits);

    // Rotate all vectors and CAQ-encode into qdata_
    std::vector<uint8_t> tmp_codes(num_points_ * dimension_);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* raw_vec = data + i * dimension_;
        std::vector<float> rotated(dimension_);
        pca_rotator_.rotate(raw_vec, rotated.data());
        caq_encode_single(
            rotated.data(), dimension_,
            &tmp_codes[i * dimension_], &caq_factors_[i]
        );
    }

    // Scatter codes into qdata_ rows
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::memcpy(
            &qdata_.at(q_row_offset_ * i),
            &tmp_codes[dimension_ * i],
            dimension_
        );
    }

    std::cout << "\tVectors Copied, PCA Rotated, CAQ Encoded (asymmetric)\n";
}

inline void QuantizedGraph::finalize_index() {
    // Copy neighbor IDs from data_ to qdata_ for colocated query access
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const PID* src = get_neighbors(i);
        uint8_t* dst = &qdata_.at(q_row_offset_ * i + q_neighbor_offset_);
        std::memcpy(dst, src, degree_bound_ * sizeof(PID));
    }
    std::cout << "\tIndex finalized (neighbors colocated)\n";
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::ofstream output(filename, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    // Save PCA rotator
    pca_rotator_.save(output);

    // Save quant plan
    size_t plan_size = quant_plan_.size();
    output.write(reinterpret_cast<const char*>(&plan_size), sizeof(size_t));
    for (auto& seg : quant_plan_) {
        output.write(reinterpret_cast<const char*>(&seg), sizeof(QuantSegment));
    }

    data_.save(output);
    qdata_.save(output);
    output.write(
        reinterpret_cast<const char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );
    output.close();
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    // Load PCA rotator
    pca_rotator_.load(input);

    // Load quant plan
    size_t plan_size = 0;
    input.read(reinterpret_cast<char*>(&plan_size), sizeof(size_t));
    quant_plan_.resize(plan_size);
    for (auto& seg : quant_plan_) {
        input.read(reinterpret_cast<char*>(&seg), sizeof(QuantSegment));
    }

    data_.load(input);
    qdata_.load(input);
    input.read(
        reinterpret_cast<char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );
    input.close();
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    cur_ef_ = cur_ef;
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

struct CandidateComparator {
    bool operator()(const Candidate<float>& lhs, const Candidate<float>& rhs) const {
        return lhs.distance < rhs.distance;
    }
};

using MaxHeap = std::
    priority_queue<Candidate<float>, std::vector<Candidate<float>>, CandidateComparator>;

/*
 * Search using PCA-rotated asymmetric CAQ distance estimation.
 *
 * Phase 0: Asymmetric [vmin, vmax] quantization with rescale estimator.
 * Phase 1: PCA rotation concentrates variance in leading dimensions.
 *
 * All code data accessed from qdata_ (colocated CAQ codes + neighbor IDs).
 * Per-vector factors accessed from caq_factors_.
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();

    // Phase 1: PCA rotate query
    std::vector<float> rotated_q(dimension_);
    pca_rotator_.rotate(query, rotated_q.data());

    // Precompute per-query constants for CAQ distance
    float q_l2sqr = 0;
    float sum_q = 0;
    for (size_t d = 0; d < dimension_; ++d) {
        q_l2sqr += rotated_q[d] * rotated_q[d];
        sum_q += rotated_q[d];
    }

    MaxHeap search_pool, res_pool;
    constexpr size_t PREFETCH_AHEAD = 4;

    // Start from entry point
    PID cur_node = entry_point_;
    float sqr_y = caq_l2_estimate(
        rotated_q.data(), get_qvector(cur_node), &caq_factors_[cur_node],
        q_l2sqr, sum_q, dimension_
    );
    float lowerBound = sqr_y;
    search_pool.emplace(cur_node, -sqr_y);
    res_pool.emplace(cur_node, sqr_y);
    visited_.set(cur_node);

    while (!search_pool.empty()) {
        const auto& candidate = search_pool.top();
        cur_node = candidate.id;
        if (-candidate.distance > lowerBound) {
            break;
        }
        search_pool.pop();

        const PID* ptr_nb = get_qneighbors(cur_node);

        // Prefetch initial batch
        for (uint32_t j = 0; j < std::min(degree_bound_, PREFETCH_AHEAD); ++j) {
            memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(get_qvector(ptr_nb[j])),
                q_prefetch_lines_
            );
        }

        for (uint32_t i = 0; i < degree_bound_; ++i) {
            if (i + PREFETCH_AHEAD < degree_bound_) {
                memory::mem_prefetch_l2(
                    reinterpret_cast<const char*>(get_qvector(ptr_nb[i + PREFETCH_AHEAD])),
                    q_prefetch_lines_
                );
            }

            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                sqr_y = caq_l2_estimate(
                    rotated_q.data(), get_qvector(cur_neighbor),
                    &caq_factors_[cur_neighbor],
                    q_l2sqr, sum_q, dimension_
                );
                if (res_pool.size() < cur_ef_ || lowerBound > sqr_y) {
                    search_pool.emplace(cur_neighbor, -sqr_y);
                    res_pool.emplace(cur_neighbor, sqr_y);
                    if (res_pool.size() > cur_ef_) {
                        res_pool.pop();
                    }
                    lowerBound = res_pool.top().distance;
                }
            }
        }
    }

    while (res_pool.size() > knn) {
        res_pool.pop();
    }
    for (int i = knn - 1; i >= 0 && !res_pool.empty(); --i) {
        results[i] = res_pool.top().id;
        res_pool.pop();
    }
}

inline void QuantizedGraph::initialize() {
    this->neighbor_offset_ = dimension_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );

    // Query-time layout: caq_code + neighbor_IDs (colocated)
    this->q_neighbor_offset_ = dimension_;
    this->q_row_offset_ = dimension_ + degree_bound_ * sizeof(PID);
    this->q_prefetch_lines_ = (q_row_offset_ + 63) / 64;
    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{num_points_, q_row_offset_}
    );

    caq_factors_.resize(num_points_);
}

inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);

    buffer::SearchBuffer tmp_pool(search_ef);
    float entry_dist = space::l2_sqr(query, get_vector(this->entry_point_), dimension_);
    tmp_pool.insert(this->entry_point_, entry_dist);

    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);

        float sqr_y = space::l2_sqr(query, get_vector(cur_candi), dimension_);

        const PID* ptr_nb = get_neighbors(cur_candi);
        auto cur_degree = degrees[cur_candi];
        for (uint32_t i = 0; i < cur_degree; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (vis.get(cur_neighbor)) {
                continue;
            }
            float dist = space::l2_sqr(query, get_vector(cur_neighbor), dimension_);
            if (tmp_pool.is_full(dist)) {
                continue;
            }
            tmp_pool.insert(cur_neighbor, dist);
        }

        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, sqr_y);
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();
    if (cur_degree == 0) {
        return;
    }
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }
}
}  // namespace symqg
