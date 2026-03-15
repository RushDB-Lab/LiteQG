#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"

namespace symqg {

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    size_t padded_dim_ = 0;
    PID entry_point_ = 0;

    PCARotator pca_rotator_;
    QuantPlan quant_plan_;

    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    QGScanner scanner_;
    HashBasedBooleanSet visited_;
    buffer::SearchBuffer search_pool_;

    // Pre-allocated search buffers (avoid per-query malloc)
    QGQuery query_obj_;
    std::vector<float> appro_dist_;

    // Prefetch lines for one data row
    size_t prefetch_lines_ = 0;

    size_t code_offset_ = 0;
    size_t factor_offset_ = 0;
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
    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(
            &data_.at((row_offset_ * data_id) + code_offset_)
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
        return reinterpret_cast<PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }
    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void find_candidates(
        PID, size_t, std::vector<Candidate<float>>&,
        HashBasedBooleanSet&, const std::vector<uint32_t>&
    ) const;

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
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_((dim + 63) & ~63ULL)
    , pca_rotator_(dim)
    , scanner_(padded_dim_, degree_bound_)
    , visited_(100)
    , search_pool_(0)
    , query_obj_(padded_dim_)
    , appro_dist_(degree_bound_) {
    initialize();
}

inline void QuantizedGraph::initialize() {
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    size_t fascscan_dim = padded_dim_ * 4;
    size_t num_blocks = (degree_bound_ + kBatchSize - 1) / kBatchSize;
    size_t fascscan_bytes = num_blocks * fascscan_dim * 4;

    code_offset_ = dimension_;
    size_t code_size_floats = fascscan_bytes / sizeof(float);
    factor_offset_ = code_offset_ + code_size_floats;
    neighbor_offset_ = factor_offset_ + 4 * degree_bound_;
    row_offset_ = neighbor_offset_ + degree_bound_;

    // row_offset_ floats * 4 bytes, 64 bytes per cache line
    prefetch_lines_ = (row_offset_ * sizeof(float) + 63) / 64;

    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }

    pca_rotator_.fit(data, num_points_, dimension_);

    constexpr float kAvgBits = 4.0f;
    quant_plan_ = segment_dp(pca_rotator_.eigenvalues().data(), dimension_, kAvgBits);

    std::cout << "\tVectors Copied, PCA Fitted\n";
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

    std::vector<float> centroid_rot(padded_dim_, 0.0f);
    pca_rotator_.rotate(get_vector(cur_id), centroid_rot.data());

    std::vector<float> nb_rot(padded_dim_, 0.0f);
    std::vector<float> residual(padded_dim_, 0.0f);
    std::vector<uint8_t> codes_4bit(padded_dim_);
    std::vector<uint8_t> all_codes(cur_degree * padded_dim_, 0);

    float* fac_ptr = get_factor(cur_id);
    float* fac_a = fac_ptr;
    float* fac_b = fac_a + degree_bound_;
    float* fac_c = fac_b + degree_bound_;
    float* fac_d = fac_c + degree_bound_;

    for (size_t i = 0; i < cur_degree; ++i) {
        PID nb_id = new_neighbors[i].id;

        std::fill(nb_rot.begin(), nb_rot.end(), 0.0f);
        pca_rotator_.rotate(get_vector(nb_id), nb_rot.data());

        for (size_t d = 0; d < padded_dim_; ++d) {
            residual[d] = nb_rot[d] - centroid_rot[d];
        }

        float o_l2sqr, delta, vmin, rescale, sum_code;
        caq_encode_4bit(
            residual.data(), padded_dim_, codes_4bit.data(),
            &o_l2sqr, &delta, &vmin, &rescale, &sum_code, 6
        );

        std::memcpy(&all_codes[i * padded_dim_], codes_4bit.data(), padded_dim_);

        float ip_c_r = 0;
        for (size_t d = 0; d < padded_dim_; ++d) {
            ip_c_r += centroid_rot[d] * residual[d];
        }

        fac_a[i] = o_l2sqr + 2.0f * ip_c_r;
        fac_b[i] = -2.0f * rescale * delta;
        fac_c[i] = -2.0f * rescale * delta * sum_code;
        fac_d[i] = -2.0f * rescale * (0.5f * delta + vmin);
    }

    pack_codes_4bit(padded_dim_, all_codes.data(), cur_degree, get_packed_code(cur_id));
}

inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist, q_obj.lut(), sqr_y,
        q_obj.width(), q_obj.vl_half(), q_obj.sum_q_float(),
        packed_code, factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())),
            prefetch_lines_
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

inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    this->visited_.clear();
    this->search_pool_.clear();

    query_obj_.prepare(query, pca_rotator_, scanner_);

    search_pool_.insert(this->entry_point_, FLT_MAX);
    buffer::ResultBuffer res_pool(knn);

    while (search_pool_.has_next()) {
        PID cur_node = search_pool_.pop();
        if (visited_.get(cur_node)) {
            continue;
        }
        visited_.set(cur_node);

        float sqr_y = scan_neighbors(
            query_obj_, get_vector(cur_node),
            appro_dist_.data(), this->search_pool_, this->degree_bound_
        );
        res_pool.insert(cur_node, sqr_y);
    }

    update_results(res_pool, query);
    res_pool.copy_results(results);
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

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->search_pool_.resize(cur_ef);
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    pca_rotator_.save(output);

    size_t plan_size = quant_plan_.size();
    output.write(reinterpret_cast<const char*>(&plan_size), sizeof(size_t));
    for (auto& seg : quant_plan_) {
        output.write(reinterpret_cast<const char*>(&seg), sizeof(QuantSegment));
    }

    data_.save(output);
    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::cout << "Loading quantized graph " << filename << '\n';

    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    pca_rotator_.load(input);

    size_t plan_size = 0;
    input.read(reinterpret_cast<char*>(&plan_size), sizeof(size_t));
    quant_plan_.resize(plan_size);
    for (auto& seg : quant_plan_) {
        input.read(reinterpret_cast<char*>(&seg), sizeof(QuantSegment));
    }

    data_.load(input);
    input.close();
    std::cout << "Quantized graph loaded!\n";
}

}  // namespace symqg
