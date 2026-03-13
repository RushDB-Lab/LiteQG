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
    std::vector<float> sq8_scale_; // per-dimension scale values (SoA)

    // Raw vectors + neighbor IDs (used during build, raw vectors also kept for reranking)
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // SQ8 quantized vectors (used during query)
    data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>
        qdata_;

    size_t cur_ef_ = 0;
    HashBasedBooleanSet visited_;

    /*
     * Data layout per node in data_:
     *      RawVector (dimension_ floats) + NeighborIDs (degree_bound_ PIDs)
     */
    size_t neighbor_offset_ = 0;
    size_t row_offset_ = 0;

    void initialize();

    void copy_vectors(const float*);

    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] uint8_t* get_qvector(PID data_id) {
        return &qdata_.at(dimension_ * data_id);
    }

    [[nodiscard]] const uint8_t* get_qvector(PID data_id) const {
        return &qdata_.at(dimension_ * data_id);
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

    // Build-time: find candidate neighbors using exact L2
    void find_candidates(
        PID, size_t, std::vector<Candidate<float>>&,
        HashBasedBooleanSet&, const std::vector<uint32_t>&
    ) const;

    // Write neighbor IDs to graph
    void update_qg(PID, const std::vector<Candidate<float>>&);

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
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }
    sq8_quantize(data, dimension_, num_points_, qdata_.data(), sq8_min_.data(), sq8_scale_.data());
    std::cout << "\tVectors Copied and Quantized with SQ8\n";
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
 * Standard greedy beam search with two heaps:
 *   search_pool (min-heap via negated distances) - candidates to explore
 *   res_pool    (max-heap) - best results found so far
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();

    MaxHeap search_pool, res_pool;

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

        const PID* ptr_nb = get_neighbors(cur_node);
        for (uint32_t i = 0; i < degree_bound_; ++i) {
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
    this->neighbor_offset_ = dimension_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
    qdata_ = data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
        std::vector<size_t>{num_points_, dimension_}
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
