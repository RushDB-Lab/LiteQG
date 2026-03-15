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
#include "../quantization/rabitq.hpp"
#include "../quantization/saq_plan.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./qg_scanner.hpp"

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

    // Per-vector CAQ factors (System B: accurate re-score)
    std::vector<CaqFactors> caq_factors_;

    // Per-vector CAQ codes in centered-rotated space [N * dim bytes]
    std::vector<uint8_t> vec_codes_;

    // Rotated-space centroid [dim floats]
    std::vector<float> centroid_;

    // Padded dimension (rounded up to 64 for binary packing)
    size_t padded_dim_ = 0;

    // Per-vector RaBitQ factors (System A: fastscan)
    std::vector<float> fast_triple_x_;
    std::vector<float> fast_fac_dq_;
    std::vector<float> fast_fac_vq_;

    // Per-vector packed binary codes [N][padded_dim/64 uint64s]
    std::vector<std::vector<uint64_t>> binary_packed_;

    // Fastscan scanner
    QGScanner scanner_;

    // Build-time: raw vectors + neighbor IDs
    //   Layout: [raw_vec (dimension_ floats) | neighbor_IDs (degree_bound_ PIDs)]
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // Query-time: fastscan packed codes + RaBitQ factors + neighbor IDs (colocated)
    //   Layout per node:
    //   [packed_fastscan_codes | triple_x[deg] | fac_dq[deg] | fac_vq[deg] | neighbor_IDs[deg]]
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
    size_t fastscan_bytes_ = 0;    // packed fastscan codes size per node
    size_t q_factors_offset_ = 0;  // offset to triple_x array
    size_t q_neighbor_offset_ = 0; // offset to neighbor IDs
    size_t q_row_offset_ = 0;      // total bytes per node
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
    [[nodiscard]] const uint8_t* get_qcodes(PID data_id) const {
        return &qdata_.at(q_row_offset_ * data_id);
    }
    [[nodiscard]] const float* get_qfactors(PID data_id) const {
        return reinterpret_cast<const float*>(
            &qdata_.at(q_row_offset_ * data_id + q_factors_offset_)
        );
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

    // Rotate all vectors and compute centroid in rotated space
    std::vector<float> all_rotated(num_points_ * dimension_);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* raw_vec = data + i * dimension_;
        pca_rotator_.rotate(raw_vec, &all_rotated[i * dimension_]);
    }

    // Compute centroid = mean of rotated vectors
    centroid_.assign(dimension_, 0.0f);
    for (size_t i = 0; i < num_points_; ++i) {
        for (size_t d = 0; d < dimension_; ++d) {
            centroid_[d] += all_rotated[i * dimension_ + d];
        }
    }
    float inv_n = 1.0f / static_cast<float>(num_points_);
    for (size_t d = 0; d < dimension_; ++d) {
        centroid_[d] *= inv_n;
    }

    // System B (accurate): CAQ encode centered vectors
    vec_codes_.resize(num_points_ * dimension_);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::vector<float> centered(dimension_);
        for (size_t d = 0; d < dimension_; ++d) {
            centered[d] = all_rotated[i * dimension_ + d] - centroid_[d];
        }
        caq_encode_single(
            centered.data(), dimension_,
            &vec_codes_[i * dimension_], &caq_factors_[i]
        );
    }

    // System A (fast): binarize centered vectors + compute RaBitQ factors
    size_t words_per_vec = padded_dim_ / 64;
    fast_triple_x_.resize(num_points_);
    fast_fac_dq_.resize(num_points_);
    fast_fac_vq_.resize(num_points_);
    binary_packed_.resize(num_points_);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::vector<float> centered(padded_dim_, 0.0f);
        std::vector<int> binary(padded_dim_, 0);
        for (size_t d = 0; d < dimension_; ++d) {
            centered[d] = all_rotated[i * dimension_ + d] - centroid_[d];
            binary[d] = (centered[d] > 0) ? 1 : 0;
        }

        rabitq_factors_single(
            centered.data(), binary.data(), centroid_.data(), dimension_,
            &fast_triple_x_[i], &fast_fac_dq_[i], &fast_fac_vq_[i]
        );

        binary_packed_[i].resize(words_per_vec);
        space::pack_binary(binary.data(), binary_packed_[i].data(), padded_dim_);
    }

    std::cout << "\tVectors Copied, PCA Rotated, Centered, CAQ + RaBitQ Encoded\n";
}

inline void QuantizedGraph::finalize_index() {
    // Build qdata_ with fastscan layout per node
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const PID* src_nb = get_neighbors(i);
        uint8_t* qrow = &qdata_.at(q_row_offset_ * i);

        // Gather binary codes of neighbors and pack for fastscan
        size_t words_per_vec = padded_dim_ / 64;
        std::vector<uint64_t> gathered(degree_bound_ * words_per_vec, 0);
        for (size_t nb = 0; nb < degree_bound_; ++nb) {
            PID nb_id = src_nb[nb];
            std::memcpy(
                &gathered[nb * words_per_vec],
                binary_packed_[nb_id].data(),
                words_per_vec * sizeof(uint64_t)
            );
        }
        pack_codes(padded_dim_, gathered.data(), degree_bound_, qrow);

        // Write RaBitQ factors: [triple_x | fac_dq | fac_vq]
        float* factors = reinterpret_cast<float*>(qrow + q_factors_offset_);
        for (size_t nb = 0; nb < degree_bound_; ++nb) {
            PID nb_id = src_nb[nb];
            factors[nb] = fast_triple_x_[nb_id];
            factors[degree_bound_ + nb] = fast_fac_dq_[nb_id];
            factors[2 * degree_bound_ + nb] = fast_fac_vq_[nb_id];
        }

        // Write neighbor IDs
        PID* dst_nb = reinterpret_cast<PID*>(qrow + q_neighbor_offset_);
        std::memcpy(dst_nb, src_nb, degree_bound_ * sizeof(PID));
    }

    // Free build-time binary data (no longer needed)
    binary_packed_.clear();
    binary_packed_.shrink_to_fit();
    fast_triple_x_.clear();
    fast_triple_x_.shrink_to_fit();
    fast_fac_dq_.clear();
    fast_fac_dq_.shrink_to_fit();
    fast_fac_vq_.clear();
    fast_fac_vq_.shrink_to_fit();

    std::cout << "\tIndex finalized (fastscan packed + factors + neighbors colocated)\n";
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

    // Save centroid
    output.write(
        reinterpret_cast<const char*>(centroid_.data()),
        static_cast<std::streamsize>(dimension_ * sizeof(float))
    );

    // Save padded_dim and layout offsets
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&fastscan_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_factors_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_neighbor_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_row_offset_), sizeof(size_t));

    data_.save(output);
    qdata_.save(output);

    // Save CAQ factors
    output.write(
        reinterpret_cast<const char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );

    // Save per-vector CAQ codes
    output.write(
        reinterpret_cast<const char*>(vec_codes_.data()),
        static_cast<std::streamsize>(num_points_ * dimension_)
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

    // Load centroid
    centroid_.resize(dimension_);
    input.read(
        reinterpret_cast<char*>(centroid_.data()),
        static_cast<std::streamsize>(dimension_ * sizeof(float))
    );

    // Load padded_dim and layout offsets
    input.read(reinterpret_cast<char*>(&padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&fastscan_bytes_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_factors_offset_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_neighbor_offset_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_row_offset_), sizeof(size_t));

    data_.load(input);
    qdata_.load(input);

    // Load CAQ factors
    input.read(
        reinterpret_cast<char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );

    // Load per-vector CAQ codes
    vec_codes_.resize(num_points_ * dimension_);
    input.read(
        reinterpret_cast<char*>(vec_codes_.data()),
        static_cast<std::streamsize>(num_points_ * dimension_)
    );

    // Reconstruct scanner
    scanner_ = QGScanner(padded_dim_, degree_bound_);

    q_prefetch_lines_ = (q_row_offset_ + 63) / 64;

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
 * Search using fastscan (RaBitQ) for fast filtering + CAQ for accurate re-scoring.
 *
 * System A (fast): binary fastscan with RaBitQ factors → approximate distance
 * System B (accurate): 8-bit CAQ codes → precise distance estimate
 *
 * Both systems operate in centered-rotated space (rotated - centroid).
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();

    // PCA rotate query
    std::vector<float> rotated_q(dimension_);
    pca_rotator_.rotate(query, rotated_q.data());

    // Center query (subtract centroid)
    std::vector<float> centered_q(dimension_);
    float q_l2sqr = 0;
    float sum_q = 0;
    for (size_t d = 0; d < dimension_; ++d) {
        centered_q[d] = rotated_q[d] - centroid_[d];
        q_l2sqr += centered_q[d] * centered_q[d];
        sum_q += centered_q[d];
    }

    // Fastscan query quantization: pad to padded_dim_, quantize 6-bit, build LUT
    std::vector<float, memory::AlignedAllocator<float>> padded_q(padded_dim_, 0.0f);
    std::copy(centered_q.begin(), centered_q.end(), padded_q.data());

    float lo, hi;
    scalar::data_range(padded_q.data(), padded_dim_, lo, hi);
    float width = (hi - lo) / ((1 << QG_BQUERY) - 1);
    if (width < 1e-10f) width = 1e-10f;
    float vl = lo;

    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
    int32_t sumq_int = 0;
    scalar::quantize(byte_query.data(), padded_q.data(), padded_dim_, lo, width, sumq_int);

    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut(padded_dim_ * 4);
    scanner_.pack_lut(byte_query.data(), lut.data());

    // Approx dist buffer for one node's neighbors
    std::vector<float> appro_dist(degree_bound_);

    MaxHeap search_pool, res_pool;

    // Start from entry point — use accurate CAQ distance
    PID cur_node = entry_point_;
    float sqr_y = caq_l2_estimate(
        centered_q.data(), &vec_codes_[cur_node * dimension_], &caq_factors_[cur_node],
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

        // Fastscan: batch compute approximate distances for all neighbors
        const uint8_t* packed_codes_ptr = get_qcodes(cur_node);
        const float* factors_ptr = get_qfactors(cur_node);
        const PID* ptr_nb = get_qneighbors(cur_node);

        scanner_.scan_neighbors(
            appro_dist.data(), lut.data(),
            q_l2sqr, vl, width, sumq_int,
            packed_codes_ptr, factors_ptr
        );

        // Filter + accurate re-score
        for (uint32_t i = 0; i < degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (visited_.get(cur_neighbor)) {
                continue;
            }
            visited_.set(cur_neighbor);

            // Fastscan filter: skip if approx distance worse than current bound
            if (res_pool.size() >= cur_ef_ && appro_dist[i] > lowerBound) {
                continue;
            }

            // Accurate re-score with CAQ
            // Prefetch the CAQ codes for this neighbor
            memory::mem_prefetch_l1(
                reinterpret_cast<const char*>(&vec_codes_[cur_neighbor * dimension_]),
                (dimension_ + 63) / 64
            );
            sqr_y = caq_l2_estimate(
                centered_q.data(), &vec_codes_[cur_neighbor * dimension_],
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

    while (res_pool.size() > knn) {
        res_pool.pop();
    }
    for (int i = knn - 1; i >= 0 && !res_pool.empty(); --i) {
        results[i] = res_pool.top().id;
        res_pool.pop();
    }
}

inline void QuantizedGraph::initialize() {
    // Build-time layout (unchanged)
    this->neighbor_offset_ = dimension_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );

    // Padded dim: round up to multiple of 64 for binary packing
    padded_dim_ = (dimension_ + 63) & ~63ULL;

    // Fastscan packed codes: degree_bound_ vectors, each padded_dim_ bits
    // pack_codes produces (degree_bound_/32) blocks, each block = padded_dim_ * 4 bytes
    size_t num_blocks = (degree_bound_ + kBatchSize - 1) / kBatchSize;
    fastscan_bytes_ = num_blocks * padded_dim_ * 4;

    // Layout: [fastscan_codes | triple_x[deg] | fac_dq[deg] | fac_vq[deg] | neighbor_IDs[deg]]
    q_factors_offset_ = fastscan_bytes_;
    size_t factors_bytes = 3 * degree_bound_ * sizeof(float);
    q_neighbor_offset_ = q_factors_offset_ + factors_bytes;
    q_row_offset_ = q_neighbor_offset_ + degree_bound_ * sizeof(PID);
    q_prefetch_lines_ = (q_row_offset_ + 63) / 64;

    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{num_points_, q_row_offset_}
    );

    caq_factors_.resize(num_points_);

    // Init scanner
    scanner_ = QGScanner(padded_dim_, degree_bound_);
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
