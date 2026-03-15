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
#include "../quantization/fastscan_impl.hpp"
#include "../quantization/saq_plan.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"
#include "../utils/scalar_quantize.hpp"

namespace symqg {

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    size_t padded_dim_ = 0;  // dimension rounded up to multiple of 16 for SIMD
    PID entry_point_ = 0;

    // PCA rotator (Phase 1)
    PCARotator pca_rotator_;

    // Segmentation plan from DP (Phase 2)
    QuantPlan quant_plan_;

    // Per-vector 8-bit CAQ codes in rotated space (for accurate re-scoring)
    std::vector<uint8_t> vec_codes_;

    // Per-vector CAQ factors
    std::vector<CaqFactors> caq_factors_;

    // Build-time: raw vectors + neighbor IDs
    //   Layout: [raw_vec (dimension_ floats) | neighbor_IDs (degree_bound_ PIDs)]
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // Query-time per-node layout:
    //   [packed_fastscan_codes | o_l2sqr[D] | fac_dot[D] | fac_sum[D] | neighbor_IDs]
    // where D = degree_bound_, fastscan codes are 1-bit-per-dim packed for 32 neighbors
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
    size_t fastscan_bytes_ = 0;      // packed 1-bit codes in fastscan format per node
    size_t q_factors_offset_ = 0;    // byte offset to o_l2sqr array
    size_t q_neighbor_offset_ = 0;   // byte offset to neighbor IDs
    size_t q_row_offset_ = 0;        // total bytes per node
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

    // Query-time accessors (qdata_)
    [[nodiscard]] const uint8_t* get_qrow(PID data_id) const {
        return &qdata_.at(q_row_offset_ * data_id);
    }
    [[nodiscard]] const PID* get_qneighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(get_qrow(data_id) + q_neighbor_offset_);
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
    , padded_dim_((dim + 15) & ~15UL)  // round up to 16 for SIMD
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

    // Rotate all vectors and CAQ-encode
    vec_codes_.resize(num_points_ * dimension_);
    caq_factors_.resize(num_points_);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* raw_vec = data + i * dimension_;
        std::vector<float> rotated(dimension_);
        pca_rotator_.rotate(raw_vec, rotated.data());
        caq_encode_single(
            rotated.data(), dimension_,
            &vec_codes_[i * dimension_], &caq_factors_[i]
        );
    }

    // Recompute qdata_ layout for per-node fastscan
    // fastscan packed codes: (padded_dim_/4) codebooks, 2 codebooks per byte → padded_dim_/8 bytes per vector
    // For 32 neighbors: need pack_codes_helper format
    // pack_codes_helper output size: (ncode_pad / 32) * (num_codebook / 2) * 32 bytes
    //   = num_batches * (padded_dim/8) * 32 bytes
    size_t num_codebook = padded_dim_ / 4;
    size_t num_batches = (degree_bound_ + kBatchSize - 1) / kBatchSize;
    fastscan_bytes_ = num_batches * (num_codebook / 2) * 32;
    q_factors_offset_ = fastscan_bytes_;
    q_neighbor_offset_ = q_factors_offset_ + 3 * degree_bound_ * sizeof(float);
    q_row_offset_ = q_neighbor_offset_ + degree_bound_ * sizeof(PID);
    q_prefetch_lines_ = (q_row_offset_ + 63) / 64;

    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{num_points_, q_row_offset_}
    );

    std::cout << "\tVectors Copied, PCA Rotated, CAQ Encoded\n";
    std::cout << "\tPer-node row: " << q_row_offset_ << " bytes ("
              << fastscan_bytes_ << " fastscan + "
              << 3 * degree_bound_ * sizeof(float) << " factors + "
              << degree_bound_ * sizeof(PID) << " IDs)\n";
}

inline void QuantizedGraph::finalize_index() {
    // Pack per-node: fastscan codes of 32 neighbors + factors + neighbor IDs
    size_t num_codebook_half = padded_dim_ / 8;  // pairs of codebooks

#pragma omp parallel for schedule(dynamic)
    for (size_t node = 0; node < num_points_; ++node) {
        const PID* neighbors = get_neighbors(node);
        uint8_t* qrow = &qdata_.at(q_row_offset_ * node);

        // Gather 1-bit codes (top bit of each 8-bit CAQ code) from neighbors
        // Format for pack_codes_helper: codes[vec_idx * num_codebook_half + cb_pair]
        // Each byte = 2 nibbles, one per codebook (each codebook covers 4 dims)
        std::vector<uint8_t> gathered(degree_bound_ * num_codebook_half, 0);

        for (size_t nb = 0; nb < degree_bound_; ++nb) {
            PID nb_id = neighbors[nb];
            const uint8_t* nb_codes = &vec_codes_[nb_id * dimension_];

            for (size_t cb_pair = 0; cb_pair < num_codebook_half; ++cb_pair) {
                size_t d = cb_pair * 8;  // starting dim for this codebook pair

                // First codebook: dims [d, d+1, d+2, d+3]
                // LUT convention: bit 3 → dim 0, bit 2 → dim 1, bit 1 → dim 2, bit 0 → dim 3
                uint8_t lo = 0;
                for (size_t k = 0; k < 4 && (d + k) < dimension_; ++k) {
                    uint8_t bit = (nb_codes[d + k] >> 7) & 1;
                    lo |= (bit << (3 - k));
                }

                // Second codebook: dims [d+4, d+5, d+6, d+7]
                uint8_t hi = 0;
                for (size_t k = 0; k < 4 && (d + 4 + k) < dimension_; ++k) {
                    uint8_t bit = (nb_codes[d + 4 + k] >> 7) & 1;
                    hi |= (bit << (3 - k));
                }

                gathered[nb * num_codebook_half + cb_pair] = lo | (hi << 4);
            }
        }

        // Pack into fastscan format (interleaved for SIMD)
        pack_codes_helper(padded_dim_, gathered.data(), degree_bound_, qrow);

        // Write per-neighbor factors
        float* o_l2sqr_arr = reinterpret_cast<float*>(qrow + q_factors_offset_);
        float* fac_dot_arr = o_l2sqr_arr + degree_bound_;
        float* fac_sum_arr = fac_dot_arr + degree_bound_;
        for (size_t nb = 0; nb < degree_bound_; ++nb) {
            PID nb_id = neighbors[nb];
            o_l2sqr_arr[nb] = caq_factors_[nb_id].o_l2sqr;
            fac_dot_arr[nb] = caq_factors_[nb_id].fac_dot;
            fac_sum_arr[nb] = caq_factors_[nb_id].fac_sum;
        }

        // Write neighbor IDs
        PID* nb_ids = reinterpret_cast<PID*>(qrow + q_neighbor_offset_);
        std::memcpy(nb_ids, neighbors, degree_bound_ * sizeof(PID));
    }

    std::cout << "\tIndex finalized (fastscan packed, " << q_row_offset_ << " bytes/node)\n";
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

    // Save layout params
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&fastscan_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_factors_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_neighbor_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&q_row_offset_), sizeof(size_t));

    data_.save(output);
    qdata_.save(output);

    output.write(
        reinterpret_cast<const char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );
    output.write(
        reinterpret_cast<const char*>(vec_codes_.data()),
        static_cast<std::streamsize>(num_points_ * dimension_)
    );
    output.close();
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    pca_rotator_.load(input);

    size_t plan_size = 0;
    input.read(reinterpret_cast<char*>(&plan_size), sizeof(size_t));
    quant_plan_.resize(plan_size);
    for (auto& seg : quant_plan_) {
        input.read(reinterpret_cast<char*>(&seg), sizeof(QuantSegment));
    }

    input.read(reinterpret_cast<char*>(&padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&fastscan_bytes_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_factors_offset_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_neighbor_offset_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&q_row_offset_), sizeof(size_t));
    q_prefetch_lines_ = (q_row_offset_ + 63) / 64;

    data_.load(input);
    qdata_.load(input);

    caq_factors_.resize(num_points_);
    input.read(
        reinterpret_cast<char*>(caq_factors_.data()),
        static_cast<std::streamsize>(num_points_ * sizeof(CaqFactors))
    );

    vec_codes_.resize(num_points_ * dimension_);
    input.read(
        reinterpret_cast<char*>(vec_codes_.data()),
        static_cast<std::streamsize>(num_points_ * dimension_)
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
 * Search with fastscan batch distance + multi-stage filtering.
 *
 * Phase 3: For each visited node, batch-scan all degree_bound_ neighbors using
 *   4-bit LUT fastscan (1-bit per dim from CAQ top bit).
 *
 * Phase 4: Multi-stage filtering:
 *   Stage 1 (fast): fastscan approximate distance → filter distant neighbors
 *   Stage 2 (accurate): full 8-bit CAQ distance for survivors only
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();

    // Phase 1: PCA rotate query
    std::vector<float> rotated_q(dimension_);
    pca_rotator_.rotate(query, rotated_q.data());

    // Precompute query constants
    float q_l2sqr = 0;
    float sum_q = 0;
    for (size_t d = 0; d < dimension_; ++d) {
        q_l2sqr += rotated_q[d] * rotated_q[d];
        sum_q += rotated_q[d];
    }

    // Build 4-bit LUT from quantized rotated query
    std::vector<float, memory::AlignedAllocator<float>> padded_q(padded_dim_, 0);
    std::copy(rotated_q.begin(), rotated_q.end(), padded_q.begin());

    float lo_val = 0, hi_val = 0;
    scalar::data_range(padded_q.data(), padded_dim_, lo_val, hi_val);
    float width = (hi_val - lo_val) / ((1 << QG_BQUERY) - 1);
    if (width == 0) width = 1.0f;
    int32_t sumq_int = 0;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
    scalar::quantize(byte_query.data(), padded_q.data(), padded_dim_, lo_val, width, sumq_int);

    size_t lut_size = padded_dim_ * 4;  // padded_dim_/4 codebooks × 16 bytes each
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut(lut_size);
    pack_lut_impl(padded_dim_, byte_query.data(), lut.data());

    // Buffers for fastscan
    std::vector<uint16_t> fs_result(degree_bound_);
    std::vector<float> approx_dist(degree_bound_);

    MaxHeap search_pool, res_pool;

    // Entry point: use per-vector CAQ for accurate initial distance
    PID cur_node = entry_point_;
    float sqr_y = caq_l2_estimate(
        rotated_q.data(), &vec_codes_[cur_node * dimension_],
        &caq_factors_[cur_node], q_l2sqr, sum_q, dimension_
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

        const uint8_t* qrow = get_qrow(cur_node);
        const float* o_l2sqr_arr = reinterpret_cast<const float*>(qrow + q_factors_offset_);
        const float* fac_dot_arr = o_l2sqr_arr + degree_bound_;
        const float* fac_sum_arr = fac_dot_arr + degree_bound_;
        const PID* ptr_nb = get_qneighbors(cur_node);

        // Prefetch next candidate's qrow
        if (!search_pool.empty()) {
            memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(get_qrow(search_pool.top().id)),
                q_prefetch_lines_
            );
        }

        // Phase 3: Fastscan batch distance for all neighbors
        std::fill(fs_result.begin(), fs_result.end(), 0);
        const uint8_t* code_ptr = qrow;
        size_t code_stride = (padded_dim_ / 4 / 2) * 32;  // bytes per batch in fastscan format
        for (size_t batch = 0; batch < degree_bound_; batch += kBatchSize) {
            accumulate_impl(padded_dim_, code_ptr, lut.data(), &fs_result[batch]);
            code_ptr += code_stride;
        }

        // Convert to approximate L2 distances using CAQ factors
        // raw = 2*result - sumq  (signed fastscan result)
        // est_ip = fac_dot * width * raw + fac_sum * sum_q
        // dist = o_l2sqr + q_l2sqr - 2*est_ip
        for (size_t i = 0; i < degree_bound_; ++i) {
            float raw = static_cast<float>(
                (static_cast<int>(fs_result[i]) << 1) - sumq_int
            );
            float est_ip = fac_dot_arr[i] * width * raw + fac_sum_arr[i] * sum_q;
            float dist = o_l2sqr_arr[i] + q_l2sqr - 2.0f * est_ip;
            approx_dist[i] = std::max(0.0f, dist);
        }

        // Phase 4: Multi-stage filtering
        // Stage 1: use fastscan approximate distance as filter
        // Stage 2: for promising candidates, optionally re-score with full CAQ
        // (Currently all fastscan results are used directly; Stage 2 can be
        // enabled by uncommenting the re-scoring block below for higher recall)
        for (size_t i = 0; i < degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (visited_.get(cur_neighbor)) {
                continue;
            }
            visited_.set(cur_neighbor);

            sqr_y = approx_dist[i];

            // Phase 4 Stage 2 (optional accurate re-score for borderline candidates):
            // If sqr_y is close to lowerBound, re-compute with full 8-bit CAQ
            if (res_pool.size() >= cur_ef_ && sqr_y > lowerBound) {
                continue;
            }

            // Re-score with accurate per-vector CAQ (Phase 4, Stage 2)
            sqr_y = caq_l2_estimate(
                rotated_q.data(), &vec_codes_[cur_neighbor * dimension_],
                &caq_factors_[cur_neighbor], q_l2sqr, sum_q, dimension_
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
    this->neighbor_offset_ = dimension_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );

    // qdata_ placeholder — real allocation happens in copy_vectors after segmentation
    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{1, 1}
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
