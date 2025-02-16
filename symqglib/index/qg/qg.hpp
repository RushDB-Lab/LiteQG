#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

#include "../../common.hpp"
#include "../../space/bitwise.hpp"
#include "../../space/l2.hpp"
#include "../../third/ngt/hashset.hpp"
#include "../../third/hnswlib/visited_list_pool.hpp"
#include "../../third/svs/array.hpp"
#include "../../utils/buffer.hpp"
#include "../../utils/io.hpp"
#include "../../utils/memory.hpp"
#include "../../utils/rotator.hpp"
#include "../fastscan/pack_codes.hpp"
#include "../fastscan/scanner.hpp"
#include "./quantizer.hpp"

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
        q_data_;  // quantization vectors

    data::Array<
    uint8_t,
    std::vector<size_t>,
    memory::AlignedAllocator<
        uint8_t,
        1 << 22,
        true>>
        q_deg_;  // quantization vectors
    
    data::Array<
    uint32_t,
    std::vector<size_t>,
    memory::AlignedAllocator<
        uint32_t,
        1 << 22,
        true>>
        q_nei_;  // quantization vectors
    
    Scanner scanner_;
    FHTRotator rotator_;
    QueryQuantizer query_quantizer_;
    HashBasedBooleanSet visited_;
    symqg::VisitedListPool *visited_list_pool_;
    vsag::Allocator *allocator_;
    buffer::SearchBuffer search_pool_;

    /*
     * Position of different data in each row
     *      RawData + Degree + QuantizationCodes + Factors + neighborIDs
     * (RawData + Degree)'size is rounded up to mutiple of 16 because of
     * 64bytes alignment: 16*sizeof(float); Degree is a mutiple 32, so the
     * packed codes are properly aligned to 64bytes; similar for factors and neighbor ids
     */
    size_t degree_offset_ = 0;    // pos of degree
    size_t code_offset_ = 0;      // pos of packed code (aligned)
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    size_t row_offset_ = 0;       // length of entire row

    size_t q_degree_offset_ = 0;    // pos of degree
    size_t q_neighbor_offset_ = 0;  // pos of Neighbors
    size_t q_row_offset_ = 0;       // length of entire row

    size_t cur_ef_ = 0;

    void initialize();

    // compute codes for RaBitQ
    void prepare_codes(const uint64_t* binary, PID cur_node);

    // compute factors
    void prepare_factors(
        const RowMatrix<float>&, const RowMatrix<float>&, const uint64_t*, PID
    );

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query,
        const float* __restrict__ rd_query,
        uint32_t knn,
        uint32_t* __restrict__ results
    );

    // search on quantized graph
    void search_qg_with_heap(
        const float* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results
    );

    void copy_vectors(const float*);

    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] uint8_t* get_qvector(PID data_id) {
        return reinterpret_cast<uint8_t*>(&q_data_.at(q_row_offset_ * data_id));
    }

    [[nodiscard]] auto get_qdegree(PID data_id) {
        return &q_deg_.at(data_id * q_degree_offset_);
    }

    [[nodiscard]] PID* get_qneighbors(PID data_id) {
        return &q_nei_.at(data_id * q_neighbor_offset_);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    void set_degree(PID data_id, uint32_t degree) {
        std::memcpy(&data_[row_offset_ * data_id + degree_offset_], &degree, sizeof(float));
    }

    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(&data_.at(row_offset_ * data_id + code_offset_));
    }

    [[nodiscard]] const uint8_t* get_packed_code(PID data_id) const {
        return reinterpret_cast<const uint8_t*>(
            &data_.at(row_offset_ * data_id + code_offset_)
        );
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &data_.at(row_offset_ * data_id + factor_offset_);
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &data_.at(row_offset_ * data_id + factor_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at(row_offset_ * data_id + neighbor_offset_));
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at(row_offset_ * data_id + neighbor_offset_)
        );
    }

    void
    find_candidates(const float*, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&)
        const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float* __restrict__, vl_type*&, vl_type&);

   public:
    // QuantizedGraph() = default;

    explicit QuantizedGraph(size_t, size_t, size_t, vsag::Allocator *);

    ~QuantizedGraph() = default;

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dimension_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    [[nodiscard]] auto get_degree(PID data_id) const {
        return *reinterpret_cast<const uint32_t*>(
            &data_.at(row_offset_ * data_id + degree_offset_)
        );
    }

    void save(const char*) const;

    void save(std::ostream& out_stream) const;

    void load(const char*);

    void load(std::istream& input);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void qstore();
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim, vsag::Allocator *allocator)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_(1 << log2_roundup(dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(dimension_)
    , query_quantizer_(padded_dim_)
    , visited_(100)
    , search_pool_(0)
    , allocator_(allocator) {
    visited_list_pool_ = allocator_->New<VisitedListPool>(num_points_, allocator_);
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + dimension_ * i;
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + dimension_ * i;
        uint8_t* dst = get_qvector(i);
        for (int i = 0; i < dimension_; ++i) {
            dst[i] = static_cast<uint8_t>(src[i]);
        }
        // std::copy(src, src + dimension_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

inline void QuantizedGraph::prepare_codes(const uint64_t* binary, PID cur_node) {
    pack_codes(this->padded_dim_, binary, get_degree(cur_node), get_packed_code(cur_node));
}

inline void QuantizedGraph::prepare_factors(
    const RowMatrix<float>& fac_x0,
    const RowMatrix<float>& fac_x1,
    const uint64_t* binary,
    PID cur_node
) {
    double fac_norm = 1.0 / std::sqrt(static_cast<double>(padded_dim_));
    float* fac_ptr = get_factor(cur_node);
    float* triple_x = fac_ptr;
    float* factor_dq = triple_x + this->degree_bound_;
    float* factor_vq = factor_dq + this->degree_bound_;
    size_t cur_degree = get_degree(cur_node);

    for (size_t j = 0; j < cur_degree; ++j) {
        PID cur_neighbor = get_neighbors(cur_node)[j];
        double cur_x = std::sqrt(static_cast<double>(space::l2_sqr(
            get_vector(cur_node),
            get_vector(cur_neighbor),
            this->dimension_
        )));  // current dist 2 centroid
        auto cur_x0 =
            static_cast<double>(fac_x0(static_cast<long>(j), 0));  // current <o, o_bar>
        auto cur_x1 =
            static_cast<double>(fac_x1(static_cast<long>(j), 0));  // current <c, o_bar>
        long double x_x0 = static_cast<long double>(cur_x) / cur_x0;

        triple_x[j] = static_cast<float>(cur_x * cur_x + 2 * x_x0 * cur_x1);
        factor_dq[j] = static_cast<float>(-2 * x_x0 * fac_norm);
        /* Notice we need binary code to calc factor_vq, since packing will
         * reorgnized bits */
        /* uint may overflow, cast to int */
        factor_vq[j] = static_cast<float>(
            -2 * x_x0 * fac_norm *
            static_cast<int>(
                space::popcount(padded_dim_, binary + j * (padded_dim_ / 64)) * 2 -
                padded_dim_
            )
        );
    }
}

inline void QuantizedGraph::save(const char* filename) const {
    std::cerr << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_.save(output);

    output.close();
    std::cerr << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::save(std::ostream& out_stream) const {
    // assert(out_stream.is_open());

    /* Basic variants */
    out_stream.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data */
    // data_.save(out_stream);
    q_data_.save(out_stream);
    q_deg_.save(out_stream);
    q_nei_.save(out_stream);

    /* Rotator */
    // this->rotator_.save(out_stream);

    // output.close();
    // std::cerr << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load(const char* filename) {
    std::cerr << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exist(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    /* Check file size */
    size_t filesize = get_filesize(filename);
    size_t correct_sizse = sizeof(PID) + sizeof(float) * num_points_ * row_offset_ +
                           sizeof(float) * padded_dim_;
    if (filesize != correct_sizse) {
        std::cerr << "Index file size error! Please make sure the index and "
                     "init parameters are corrrect\n";
        return;
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_.load(input);

    input.close();
    std::cerr << "Quantized graph loaded!\n";
}

inline void QuantizedGraph::load(std::istream& input) {
    /* Check file size */
    // size_t filesize = get_filesize();
    // size_t correct_sizse = sizeof(PID) + sizeof(float) * num_points_ * row_offset_ +
    //                        sizeof(float) * padded_dim_;
    // if (filesize != correct_sizse) {
    //     std::cerr << "Index file size error! Please make sure the index and "
    //                  "init parameters are corrrect\n";
    //     return;
    // }

    // std::ifstream input(filename, std::ios::binary);
    // assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data */
    // data_.load(input);
    q_data_.load(input);
    q_deg_.load(input);
    q_nei_.load(input);

    /* Rotator */
    // this->rotator_.load(input);

    // input.close();
    // std::cerr << "Quantized graph loaded!\n";
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    // this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
    cur_ef_ = cur_ef;
}

/*
 * search single query
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    search_qg_with_heap(query, knn, results);
    // if (knn < 10000) {
    //     this->search_pool_.resize(cur_ef_);
    //     this->search_pool_.clear();
    //      /* Init query matrix */
    //     std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
    //     this->rotator_.rotate(query, rd_query.data());
    //     search_qg(query, rd_query.data(), knn, results);
    // } else {
    //     search_qg_with_heap(query, knn, results);
    // }
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param rd_query  rotated query vector, padded_dim_ elements
 * @param knn       num of nearest neighbors
 * @param results   searh res
 */
inline void QuantizedGraph::search_qg(
    const float* __restrict__ query,
    const float* __restrict__ rd_query,
    uint32_t knn,
    uint32_t* __restrict__ results
) {
    /* Variables used for quantizing query */
    float width = 0;      // delta in quantization
    float lower_val = 0;  // lower bound in quantization
    float upper_val = 0;  // upper bound
    int32_t sumq = 0;     // sum of unsigned integers in quantinized query

    /* Variables used for searching */
    float sqr_y = 0;                       // square of distance from query to centroid
    PID cur_node = 0;                      // current visiting node in graph
    uint32_t cur_degree = 0;               // degree of current node in graph
    PID cur_neighbor = 0;                  // current neighbor
    const PID* ptr_nb = nullptr;           // ptr of neighbors
    const uint8_t* packed_code = nullptr;  // ptr of the quantization code
    const float* factor = nullptr;         // ptr of the pre-computed factor

    /* Quantize query */
    this->query_quantizer_.query_range(rd_query, lower_val, upper_val);
    width = (upper_val - lower_val) / ((1 << QG_BQUERY) - 1);
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
    this->query_quantizer_.quantize(byte_query.data(), rd_query, lower_val, width, sumq);

    /* Create lookup table */
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut(
        padded_dim_ << 2
    );  // B / 4 * 16
    this->scanner_.pack_lut(byte_query.data(), lut.data());

    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    /* Searching pool initialization */
    search_pool_.insert(this->entry_point_, FLT_MAX);

    /* Result pool */
    buffer::ResultBuffer res_pool(knn);

    /* Current version of fast scan compute 32 distances */
    std::vector<float, memory::AlignedAllocator<float>> appro_dist(degree_bound_
    );  // approximate dis

    // uint8_t qquery[dimension_];
    // for (int i = 0; i < dimension_; ++i) {
    //     qquery[i] = static_cast<uint8_t>(query[i]);
    // }

    while (search_pool_.has_next()) {
        cur_node = search_pool_.pop();
        if (visited_array[cur_node] != visited_array_tag) {
            visited_array[cur_node] = visited_array_tag;

            sqr_y = space::l2_sqr(query, get_vector(cur_node), dimension_);
            cur_degree = get_degree(cur_node);
            res_pool.insert(cur_node, sqr_y);

            /* Compute approximate distance by Fast Scan */
            packed_code = get_packed_code(cur_node);
            factor = get_factor(cur_node);
            this->scanner_.fastscan(
                appro_dist.data(),
                lut.data(),
                sqr_y,
                lower_val,
                width,
                sumq,
                packed_code,
                factor
            );

            ptr_nb = get_neighbors(cur_node);
            for (uint32_t i = 0; i < cur_degree; ++i) {
                cur_neighbor = ptr_nb[i];
                float tmp_dist = appro_dist[i];
    #if defined(DEBUG)
                std::cerr << "Neighbor ID " << cur_neighbor << '\n';
                std::cerr << "Appro " << appro_dist[i] << '\t';
                float __gt_dist__ = l2_sqr(query, get_vector(cur_neighbor), dimension_);
                std::cerr << "GT " << __gt_dist__ << '\t';
                std::cerr << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
                std::cerr << "sqr_y " << sqr_y << '\n';
    #endif
                if (search_pool_.is_full(tmp_dist) || visited_array[cur_neighbor] == visited_array_tag) {
                    continue;
                }
                search_pool_.insert(cur_neighbor, tmp_dist);
                // memory::mem_prefetch_l2(
                //     reinterpret_cast<char*>(get_vector(search_pool_.next_id())), 10
                // );
            }
        }
    }

    update_results(res_pool, query, visited_array, visited_array_tag);
    res_pool.copy_results(results);

    visited_list_pool_->releaseVisitedList(vl);
}

struct CandidateComparator {
    bool operator()(const Candidate<float>& lhs, const Candidate<float>& rhs) const {
        return lhs.distance < rhs.distance;  // Min-heap (smallest distance has priority)
    }
};

using MaxHeap = std::priority_queue<Candidate<float>,
                                               std::vector<Candidate<float>, vsag::AllocatorWrapper<Candidate<float>>>,
                                               CandidateComparator>;

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param rd_query  rotated query vector, padded_dim_ elements
 * @param knn       num of nearest neighbors
 * @param results   searh res
 */
inline void QuantizedGraph::search_qg_with_heap(
    const float* __restrict__ query,
    uint32_t knn,
    uint32_t* __restrict__ results
) {
    /* Variables used for searching */
    float sqr_y = 0;                       // square of distance from query to centroid
    PID cur_node = 0;                      // current visiting node in graph
    uint32_t cur_degree = 0;               // degree of current node in graph
    PID cur_neighbor = 0;                  // current neighbor
    const PID* ptr_nb = nullptr;           // ptr of neighbors

    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    // memory::AlignedAllocator<Candidate<float>> allocator;
    MaxHeap search_pool(allocator_);
    MaxHeap res_pool(allocator_);

    uint8_t qquery[dimension_];
    for (int i = 0; i < dimension_; ++i) {
        qquery[i] = static_cast<uint8_t>(query[i]);
    }

    cur_node = this->entry_point_;
    sqr_y = space::l2_spr_uint8(qquery, get_qvector(cur_node), dimension_);
    float lowerBound = sqr_y;
    search_pool.emplace(cur_node, -sqr_y);
    res_pool.emplace(cur_node, sqr_y);
    visited_array[cur_node] = visited_array_tag;
    
    while (!search_pool.empty()) {
        auto &candidate = search_pool.top();
        cur_node = candidate.id;
        if (-candidate.distance > lowerBound) {
            break;
        }
        search_pool.pop();
        
        cur_degree = *get_qdegree(cur_node);
        ptr_nb = get_qneighbors(cur_node);
        
        for (uint32_t i = 0; i < cur_degree; ++i) {
            cur_neighbor = ptr_nb[i];
            if (visited_array[cur_neighbor] != visited_array_tag) {
                visited_array[cur_neighbor] = visited_array_tag;
                sqr_y = space::l2_spr_uint8(qquery, get_qvector(cur_neighbor), dimension_);
                if (res_pool.size() < cur_ef_ || lowerBound > sqr_y) {
                    search_pool.emplace(cur_neighbor, -sqr_y);
                    res_pool.emplace(cur_neighbor, sqr_y);
                    if (res_pool.size() > cur_ef_) {
                        res_pool.pop();
                    }
                    if (!res_pool.empty()) {
                        lowerBound = res_pool.top().distance;
                    }
                }
            }
            memory::mem_prefetch_l2(
                reinterpret_cast<char*>(get_vector(cur_neighbor)), 10
            );
        }
    }

    while (res_pool.size() > knn) {
        res_pool.pop();
    }

    for (int i = 0; i < knn; ++i) {
        if (!res_pool.empty()) {
            results[i] = res_pool.top().id;
            res_pool.pop();
        } else {
            results[i] = 0;
        }
    }
    
    visited_list_pool_->releaseVisitedList(vl);
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* __restrict__ query, vl_type* &visited_array, vl_type &visited_array_tag
) {
    if (result_pool.is_full()) {
        return;
    }

    auto &ids = result_pool.ids();
    for (auto &data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        uint32_t deg = get_degree(data_id);
        for (uint32_t i = 0; i < deg; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (visited_array[cur_neighbor] != visited_array_tag) {
                visited_array[cur_neighbor] = visited_array_tag;
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
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    this->degree_offset_ = dimension_;  // Pos of degree 128
    this->code_offset_ =
        round_up2multiple(dimension_ + 1, 16);  // Pos of packed code (aligned) 144
    this->factor_offset_ =
        code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor 144+128=272
    this->neighbor_offset_ =
        factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float); // 
    this->row_offset_ = neighbor_offset_ + degree_bound_;

    /* Check alignment */
    assert(row_offset_ * sizeof(float) % 32 == 0);

    /* Allocate memory of data*/
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );

    // (1000000 * (1 + 32) * 4 + 128 * 1000000)/1024/1024 = 247.9MB
    // auto k = 1+32+32;
    q_degree_offset_ = 1; // 32
    q_neighbor_offset_ = degree_bound_; // 48
    q_row_offset_ = dimension_; // 80

    /* Check alignment */
    // assert(q_row_offset_ * sizeof(uint32_t) % 32 == 0);

    q_data_ = data::
        Array<uint8_t, std::vector<size_t>, memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
            std::vector<size_t>{num_points_, q_row_offset_}
        );

    q_deg_ = data::
        Array<uint8_t, std::vector<size_t>, memory::AlignedAllocator<uint8_t, 1 << 22, true>>(
            std::vector<size_t>{num_points_, q_degree_offset_}
        );
    
    q_nei_ = data::
        Array<uint32_t, std::vector<size_t>, memory::AlignedAllocator<uint32_t, 1 << 22, true>>(
            std::vector<size_t>{num_points_, q_neighbor_offset_}
        );
}

inline void QuantizedGraph::qstore() {
    for (int i = 0; i < num_points_; ++i) {
        auto deg = get_degree(i);
        *get_qdegree(i) = deg;
        auto nei = get_neighbors(i);
        auto q_nei = get_qneighbors(i);
        for (int i = 0; i < deg; ++i) {
            q_nei[i] = nei[i];
        }
    }
}

inline void QuantizedGraph::find_candidates(
    const float* query,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis
) const {
    std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
    this->rotator_.rotate(query, rd_query.data());

    /* Variables used for quantizing query */
    float width = 0;  // delta in quantization
    float lower_val = 0;
    float upper_val = 0;  // upper bound & lower bound in quantization
    int32_t sumq = 0;     // sum of unsigned integers in quantinized query

    /* Quantize query */
    this->query_quantizer_.query_range(rd_query.data(), lower_val, upper_val);
    width = (upper_val - lower_val) / ((1 << QG_BQUERY) - 1);
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
    this->query_quantizer_.quantize(
        byte_query.data(), rd_query.data(), lower_val, width, sumq
    );

    /* Create lookup table */
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut(
        padded_dim_ << 2
    );  // B / 4 * 16
    this->scanner_.pack_lut(byte_query.data(), lut.data());

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float, memory::AlignedAllocator<float, 64>> appro_dist(degree_bound_
    );  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_node = tmp_pool.pop();
        if (vis.get(cur_node)) {
            continue;
        }
        vis.set(cur_node);

        const auto* cur_data = get_vector(cur_node);
        auto sqr_y = space::l2_sqr(query, cur_data, dimension_);
        results.emplace_back(cur_node, sqr_y);

        auto cur_degree = *reinterpret_cast<const uint32_t*>(&cur_data[degree_offset_]);

        /* Compute approximate distance by Fast Scan */
        const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
        const auto* factor = &cur_data[factor_offset_];
        this->scanner_.fastscan(
            appro_dist.data(),
            lut.data(),
            sqr_y,
            lower_val,
            width,
            sumq,
            packed_code,
            factor
        );

        const auto* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
        for (uint32_t i = 0; i < cur_degree; ++i) {
            auto cur_neighbor = ptr_nb[i];
            auto tmp_dist = appro_dist[i];
            if (tmp_pool.is_full(tmp_dist) || vis.get(cur_neighbor)) {
                continue;
            }
            tmp_pool.insert(cur_neighbor, tmp_dist);
            memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(get_vector(tmp_pool.next_id())), 10
            );
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();
    set_degree(cur_id, cur_degree);

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    double fac_norm = 1 / std::sqrt(static_cast<double>(padded_dim_));

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

    /* rotate Matrix */
    RowMatrix<float> x_rotated(cur_degree, padded_dim_);
    RowMatrix<float> c_rotated(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
    }
    this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    /* Compute residual */
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        x_rotated.row(i) = x_rotated.row(i) - c_rotated;
    }

    /* Binary rentry_pointresentation */
    RowMatrix<int> bin_x_rotated(cur_degree, padded_dim_);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        for (uint32_t j = 0; j < padded_dim_; ++j) {
            bin_x_rotated(i, j) = static_cast<int>(x_rotated(i, j) > 0);
        }
    }

    /* Signed quantized vectors */
    RowMatrix<float> ones(cur_degree, padded_dim_);
    ones.setOnes();
    RowMatrix<float> sign_x_rotated = 2 * bin_x_rotated.cast<float>() - ones;

    /* fac_x0: (cur_degree, 1) */
    RowMatrix<float> fac_x0 =
        (x_rotated.array() * sign_x_rotated.array() * fac_norm).rowwise().sum();
    RowMatrix<float> x_rotated_norm = x_rotated.rowwise().norm();
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        float cur_x0 = fac_x0(i, 0);
        fac_x0(i, 0) = cur_x0 / x_rotated_norm(i, 0);
    }

    /* fac_x1: (cur_degree, 1) */
    RowMatrix<float> fac_x1(cur_degree, 1);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        fac_x1(i, 0) = static_cast<float>(
            (c_rotated.array() * sign_x_rotated.row(i).array()).sum() * fac_norm
        );
    }

    /* change bin_x_rotated to uint64  */
    std::vector<uint64_t> binary(cur_degree * (padded_dim_ / 64));
    space::pack_binary(bin_x_rotated, binary.data());

    /* Compute codes of RaBitQ */
    prepare_codes(binary.data(), cur_id);

    /* Compute factors for searching */
    prepare_factors(fac_x0, fac_x1, binary.data(), cur_id);
}
}  // namespace symqg
