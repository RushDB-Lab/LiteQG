#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>

#include "../common.hpp"
#include "../quantization/sq8.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
/**
 * @brief this Factor only for illustration, the true storage is continous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0
    float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
    float factor_vq;  // Factor of v_l * ||q_r||
};

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dimension_ = 0;     // dimension
    size_t padded_dim_ = 0;    // padded dimension
    PID entry_point_ = 0;      // Entry point of graph

    std::vector<SQ8Params> sq8_params_;  // 每维度一个 min 和 scale

    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<
            float,
            1 << 22,
            true>>
        data_;  // vectors + graph + quantization codes

    data::Array<
        uint8_t,
        std::vector<size_t>,
        memory::AlignedAllocator<
            uint8_t,
            1 << 22,
            true>>
        qdata_;  // u8 vectors

    size_t cur_ef_;

    QGScanner scanner_;
    FHTRotator rotator_;
    HashBasedBooleanSet visited_;
    buffer::SearchBuffer search_pool_;

    /*
     * Position of different data in each row
     *      RawData + QuantizationCodes + Factors + neighborIDs
     * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
     * we do not need to store the degree for each vertex
     */
    size_t code_offset_ = 0;      // pos of packed code
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    size_t row_offset_ = 0;       // length of entire row

    void initialize();

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

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

    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(&data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] const uint8_t* get_packed_code(PID data_id) const {
        return reinterpret_cast<const uint8_t*>(
            &data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void
    find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&) const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float*);

    float scan_neighbors(
        const QGQuery& q_obj,
        const float* cur_data,
        float* appro_dist,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree
    ) const;

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

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void search_qg_with_heap(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_(1 << ceil_log2(dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(dimension_)
    , visited_(100)
    , search_pool_(0) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }
    sq8_quantize(data, dimension_, num_points_, qdata_.data(), sq8_params_.data());
    std::cout << "\tVectors Copied and Quantized with SQ8\n";
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::ofstream output(filename, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    data_.save(output);
    qdata_.save(output);
    output.write(
        reinterpret_cast<const char*>(sq8_params_.data()), dimension_ * sizeof(SQ8Params)
    );
    output.close();
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    data_.load(input);
    qdata_.load(input);
    input.read(reinterpret_cast<char*>(sq8_params_.data()), dimension_ * sizeof(SQ8Params));
    input.close();
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->search_pool_.resize(cur_ef);
    cur_ef_ = cur_ef;
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

/*
 * search single query
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    /* Init query matrix */
    this->visited_.clear();
    this->search_pool_.clear();
    // search_qg(query, knn, results);
    search_qg_with_heap(query, knn, results);
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   searh res
 */
inline void QuantizedGraph::search_qg(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    // query preparation
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    search_pool_.insert(this->entry_point_, FLT_MAX);

    /* Result pool */
    buffer::ResultBuffer res_pool(knn);

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis

    while (search_pool_.has_next()) {
        PID cur_node = search_pool_.pop();
        if (visited_.get(cur_node)) {
            continue;
        }
        visited_.set(cur_node);

        float sqr_y = scan_neighbors(
            q_obj,
            get_vector(cur_node),
            appro_dist.data(),
            this->search_pool_,
            this->degree_bound_
        );
        res_pool.insert(cur_node, sqr_y);
    }

    update_results(res_pool, query);
    res_pool.copy_results(results);
}

struct CandidateComparator {
    bool operator()(const Candidate<float>& lhs, const Candidate<float>& rhs) const {
        return lhs.distance < rhs.distance;  // Min-heap (smallest distance has priority)
    }
};

using MaxHeap = std::
    priority_queue<Candidate<float>, std::vector<Candidate<float>>, CandidateComparator>;

inline void QuantizedGraph::search_qg_with_heap(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    MaxHeap search_pool, res_pool;
    visited_.clear();

    float decoded_vec[dimension_];
    PID cur_node = entry_point_;  // 显式声明并初始化为入口点
    sq8_dequantize(get_qvector(cur_node), dimension_, decoded_vec, sq8_params_.data());
    float sqr_y = space::l2_sqr(query, decoded_vec, dimension_);
    float lowerBound = sqr_y;
    search_pool.emplace(cur_node, -sqr_y);
    res_pool.emplace(cur_node, sqr_y);
    visited_.set(cur_node);

    while (!search_pool.empty()) {
        auto& candidate = search_pool.top();
        cur_node = candidate.id;  // 更新 cur_node 为当前处理的节点
        if (-candidate.distance > lowerBound)
            break;
        search_pool.pop();

        const PID* ptr_nb = get_neighbors(cur_node);
        for (uint32_t i = 0; i < degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                sq8_dequantize(
                    get_qvector(cur_neighbor), dimension_, decoded_vec, sq8_params_.data()
                );
                sqr_y = space::l2_sqr(query, decoded_vec, dimension_);
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

    for (int i = knn - 1; i >= 0 && !res_pool.empty(); --i) {
        results[i] = res_pool.top().id;
        res_pool.pop();
    }
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    /* Compute approximate distance by Fast Scan */
    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist,
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        packed_code,
        factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
#if defined(DEBUG)
        std::cout << "Neighbor ID " << cur_neighbor << '\n';
        std::cout << "Appro " << appro_dist[i] << '\t';
        float __gt_dist__ = l2_sqr(query, get_vector(cur_neighbor), dimension_);
        std::cout << "GT " << __gt_dist__ << '\t';
        std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        std::cout << "sqr_y " << sqr_y << '\n';
#endif
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }

    return sqr_y;
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, space::l2_sqr(query, get_vector(cur_neighbor), dimension_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

inline void QuantizedGraph::initialize() {
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);
    this->neighbor_offset_ = dimension_;  // 仅保留邻居偏移
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
    sq8_params_.resize(dimension_);
}

// find candidate neighbors for cur_id, exclude the vertex itself
inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        auto sqr_y = scan_neighbors(
            q_obj, get_vector(cur_candi), appro_dist.data(), tmp_pool, cur_degree
        );
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
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
    RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
    x_pad.setZero();
    c_pad.setZero();

    /* Copy data */
    for (size_t i = 0; i < cur_degree; ++i) {
        auto neighbor_id = new_neighbors[i].id;
        const auto* cur_data = get_vector(neighbor_id);
        std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    const auto* cur_cent = get_vector(cur_id);
    std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));
}
}  // namespace symqg
