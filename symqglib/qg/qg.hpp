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
#include "../quantization/sq8.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"

namespace symqg {

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    PID entry_point_ = 0;

    std::vector<float> sq8_min_;    // per-dimension min values (SoA)
    std::vector<float> sq8_scale_;  // per-dimension scale values (SoA)

    // Build-time: raw vectors + neighbor IDs
    //   Layout: [raw_vec (dimension_ floats) | neighbor_IDs (degree_bound_ PIDs)]
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // Query-time: SQ8 vectors + neighbor IDs colocated for cache locality
    //   Layout: [sq8_vec (dimension_ bytes) | neighbor_IDs (degree_bound_ * 4 bytes)]
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
    size_t q_prefetch_lines_ = 0;  // cache lines per qdata_ row

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

    // Build-time: find candidate neighbors using exact L2
    void find_candidates(
        PID, size_t, std::vector<Candidate<float>>&,
        HashBasedBooleanSet&, const std::vector<uint32_t>&
    ) const;

    // Write neighbor IDs to graph (data_ only, call finalize_index after build)
    void update_qg(PID, const std::vector<Candidate<float>>&);

    // Copy neighbor IDs from data_ to qdata_ (colocated layout)
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

    // SQ8 quantize into temporary contiguous buffer
    std::vector<uint8_t> tmp_quantized(num_points_ * dimension_);
    sq8_quantize(
        data, dimension_, num_points_, tmp_quantized.data(),
        sq8_min_.data(), sq8_scale_.data()
    );

    // Scatter into qdata_ rows (SQ8 vec portion only; neighbors copied later)
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::memcpy(
            &qdata_.at(q_row_offset_ * i),
            &tmp_quantized[dimension_ * i],
            dimension_
        );
    }
    std::cout << "\tVectors Copied and Quantized with SQ8\n";
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
    data_.save(output);
    qdata_.save(output);
    output.write(
        reinterpret_cast<const char*>(sq8_min_.data()), dimension_ * sizeof(float)
    );
    output.write(
        reinterpret_cast<const char*>(sq8_scale_.data()), dimension_ * sizeof(float)
    );
    output.close();
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    data_.load(input);
    qdata_.load(input);
    input.read(reinterpret_cast<char*>(sq8_min_.data()), dimension_ * sizeof(float));
    input.read(reinterpret_cast<char*>(sq8_scale_.data()), dimension_ * sizeof(float));
    input.close();
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    cur_ef_ = cur_ef;
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

// Max-heap by distance: largest distance at top, for easy eviction of worst candidates
struct CandidateComparator {
    bool operator()(const Candidate<float>& lhs, const Candidate<float>& rhs) const {
        return lhs.distance < rhs.distance;
    }
};

using MaxHeap = std::
    priority_queue<Candidate<float>, std::vector<Candidate<float>>, CandidateComparator>;

/*
 * Query-time search using SQ8 approximate distances.
 * All data accessed from qdata_ (colocated SQ8 vec + neighbor IDs).
 * Prefetches future neighbors to hide memory latency.
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();

    MaxHeap search_pool, res_pool;
    constexpr size_t PREFETCH_AHEAD = 4;

    // Start from entry point
    PID cur_node = entry_point_;
    float sqr_y = sq8_l2_sqr(
        query, get_qvector(cur_node), sq8_min_.data(), sq8_scale_.data(), dimension_
    );
    float lowerBound = sqr_y;
    search_pool.emplace(cur_node, -sqr_y);
    res_pool.emplace(cur_node, sqr_y);
    visited_.set(cur_node);

    while (!search_pool.empty()) {
        auto& candidate = search_pool.top();
        cur_node = candidate.id;
        if (-candidate.distance > lowerBound)
            break;
        search_pool.pop();

        // Read neighbors from qdata_ (colocated with SQ8 vecs)
        const PID* ptr_nb = get_qneighbors(cur_node);

        // Prefetch initial batch of neighbors' full rows (SQ8 vec + neighbor IDs)
        for (uint32_t j = 0; j < std::min(degree_bound_, PREFETCH_AHEAD); ++j) {
            memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(get_qvector(ptr_nb[j])),
                q_prefetch_lines_
            );
        }

        for (uint32_t i = 0; i < degree_bound_; ++i) {
            // Prefetch ahead: when processing neighbor i, prefetch neighbor i+PREFETCH_AHEAD
            if (i + PREFETCH_AHEAD < degree_bound_) {
                memory::mem_prefetch_l2(
                    reinterpret_cast<const char*>(get_qvector(ptr_nb[i + PREFETCH_AHEAD])),
                    q_prefetch_lines_
                );
            }

            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                sqr_y = sq8_l2_sqr(
                    query, get_qvector(cur_neighbor),
                    sq8_min_.data(), sq8_scale_.data(), dimension_
                );
                if (res_pool.size() < cur_ef_ || lowerBound > sqr_y) {
                    search_pool.emplace(cur_neighbor, -sqr_y);
                    res_pool.emplace(cur_neighbor, sqr_y);
                    if (res_pool.size() > cur_ef_)
                        res_pool.pop();
                    lowerBound = res_pool.top().distance;
                }
            }
        }
    }

    while (res_pool.size() > knn)
        res_pool.pop();
    for (int i = knn - 1; i >= 0 && !res_pool.empty(); --i) {
        results[i] = res_pool.top().id;
        res_pool.pop();
    }
}

inline void QuantizedGraph::initialize() {
    // Build-time layout (data_): raw_vec + neighbor_IDs
    this->neighbor_offset_ = dimension_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );

    // Query-time layout (qdata_): sq8_vec + neighbor_IDs (colocated)
    this->q_neighbor_offset_ = dimension_;
    this->q_row_offset_ = dimension_ + degree_bound_ * sizeof(PID);
    this->q_prefetch_lines_ = (q_row_offset_ + 63) / 64;
    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{num_points_, q_row_offset_}
    );

    sq8_min_.resize(dimension_);
    sq8_scale_.resize(dimension_);
}

// Build-time: beam search on graph using exact L2 distance
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

        // Scan neighbors with exact L2 distance
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
