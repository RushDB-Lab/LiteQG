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
#include "../space/bitwise.hpp"
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

// Compute split dimension from QuantPlan: first dim with < 4 bits, rounded up to 64.
inline size_t compute_split_dim(const QuantPlan& plan, size_t padded_dim) {
    size_t split = padded_dim;
    for (auto& seg : plan) {
        if (seg.bits < 4 && seg.dim_offset < split) {
            split = seg.dim_offset;
        }
    }
    split = (split + 63) & ~63ULL;
    return std::min(split, padded_dim);
}

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    size_t padded_dim_ = 0;
    size_t split_dim_ = 0;  // seg0=[0,split), seg1=[split,padded)
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

    QGQuery query_obj_;
    std::vector<float> appro_dist_;

    size_t prefetch_lines_ = 0;

    // Layout offsets (float units)
    size_t code_offset_ = 0;
    size_t seg1_code_offset_ = 0;
    size_t factor_offset_ = 0;
    size_t neighbor_offset_ = 0;
    size_t row_offset_ = 0;

    void initialize();
    void copy_vectors(const float*);

    [[nodiscard]] float* get_vector(PID id) { return &data_.at(row_offset_ * id); }
    [[nodiscard]] const float* get_vector(PID id) const { return &data_.at(row_offset_ * id); }
    [[nodiscard]] uint8_t* get_seg0_code(PID id) {
        return reinterpret_cast<uint8_t*>(&data_.at(row_offset_ * id + code_offset_));
    }
    [[nodiscard]] const uint8_t* get_seg0_code(PID id) const {
        return reinterpret_cast<const uint8_t*>(&data_.at(row_offset_ * id + code_offset_));
    }
    [[nodiscard]] uint8_t* get_seg1_code(PID id) {
        return reinterpret_cast<uint8_t*>(&data_.at(row_offset_ * id + seg1_code_offset_));
    }
    [[nodiscard]] const uint8_t* get_seg1_code(PID id) const {
        return reinterpret_cast<const uint8_t*>(&data_.at(row_offset_ * id + seg1_code_offset_));
    }
    [[nodiscard]] float* get_factor(PID id) { return &data_.at(row_offset_ * id + factor_offset_); }
    [[nodiscard]] const float* get_factor(PID id) const { return &data_.at(row_offset_ * id + factor_offset_); }
    [[nodiscard]] PID* get_neighbors(PID id) {
        return reinterpret_cast<PID*>(&data_.at(row_offset_ * id + neighbor_offset_));
    }
    [[nodiscard]] const PID* get_neighbors(PID id) const {
        return reinterpret_cast<const PID*>(&data_.at(row_offset_ * id + neighbor_offset_));
    }

    void find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&) const;
    void update_qg(PID, const std::vector<Candidate<float>>&);
    void update_results(buffer::ResultBuffer&, const float*);
    float scan_neighbors(const QGQuery&, const float*, float*, buffer::SearchBuffer&, uint32_t) const;

   public:
    explicit QuantizedGraph(size_t, size_t, size_t);

    [[nodiscard]] auto num_vertices() const { return num_points_; }
    [[nodiscard]] auto dimension() const { return dimension_; }
    [[nodiscard]] auto degree_bound() const { return degree_bound_; }
    [[nodiscard]] auto entry_point() const { return entry_point_; }
    void set_ep(PID entry) { entry_point_ = entry; }

    void save_index(const char*) const;
    void load_index(const char*);
    void set_ef(size_t);
    void search(const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results);
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_((dim + 63) & ~63ULL)
    , split_dim_(padded_dim_)  // default: all 4-bit (updated after PCA fit)
    , pca_rotator_(dim)
    , scanner_(padded_dim_, padded_dim_, degree_bound_)  // will be reconstructed
    , visited_(100)
    , search_pool_(0)
    , query_obj_(padded_dim_, padded_dim_)  // will be reconstructed
    , appro_dist_(degree_bound_) {
    initialize();
}

inline void QuantizedGraph::initialize() {
    assert(padded_dim_ % 64 == 0);
    size_t seg1_dim = padded_dim_ - split_dim_;

    // Seg0: 4-bit codes (split_dim codebooks)
    size_t seg0_fascscan_dim = split_dim_ * 4;
    size_t seg0_blocks = (degree_bound_ + kBatchSize - 1) / kBatchSize;
    size_t seg0_bytes = seg0_blocks * seg0_fascscan_dim * 4;
    size_t seg0_floats = seg0_bytes / sizeof(float);

    // Seg1: 1-bit codes (seg1_dim/4 codebooks)
    size_t seg1_bytes = 0;
    size_t seg1_floats = 0;
    if (seg1_dim > 0) {
        size_t seg1_blocks = (degree_bound_ + kBatchSize - 1) / kBatchSize;
        seg1_bytes = seg1_blocks * seg1_dim * 4;
        seg1_floats = seg1_bytes / sizeof(float);
    }

    code_offset_ = dimension_;
    seg1_code_offset_ = code_offset_ + seg0_floats;
    factor_offset_ = seg1_code_offset_ + seg1_floats;
    neighbor_offset_ = factor_offset_ + 6 * degree_bound_;  // 6 SAQ factors
    row_offset_ = neighbor_offset_ + degree_bound_;

    prefetch_lines_ = (row_offset_ * sizeof(float) + 63) / 64;

    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::copy(data + dimension_ * i, data + dimension_ * (i + 1), get_vector(i));
    }

    pca_rotator_.fit(data, num_points_, dimension_);

    constexpr float kAvgBits = 4.0f;
    quant_plan_ = segment_dp(pca_rotator_.eigenvalues().data(), dimension_, kAvgBits);

    // Compute split and reinitialize layout
    split_dim_ = compute_split_dim(quant_plan_, padded_dim_);
    std::cout << "\tSplit dim: " << split_dim_
              << " (seg0=" << split_dim_ << " 4-bit, seg1=" << (padded_dim_ - split_dim_) << " 1-bit)\n";

    initialize();  // recompute offsets with new split_dim_
    scanner_ = QGScanner(padded_dim_, split_dim_, degree_bound_);
    query_obj_ = QGQuery(padded_dim_, split_dim_);

    // Re-copy raw vectors into new layout
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        std::copy(data + dimension_ * i, data + dimension_ * (i + 1), get_vector(i));
    }

    std::cout << "\tVectors Copied, PCA Fitted\n";
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();
    if (cur_degree == 0) return;

    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i)
        neighbor_ptr[i] = new_neighbors[i].id;

    std::vector<float> centroid_rot(padded_dim_, 0.0f);
    pca_rotator_.rotate(get_vector(cur_id), centroid_rot.data());

    std::vector<float> nb_rot(padded_dim_, 0.0f);
    std::vector<float> residual(padded_dim_, 0.0f);

    // Seg0: 4-bit codes
    std::vector<uint8_t> codes_4bit(split_dim_);
    std::vector<uint8_t> all_codes_seg0(cur_degree * split_dim_, 0);

    // Seg1: 1-bit codes
    size_t seg1_dim = padded_dim_ - split_dim_;
    size_t words_seg1 = seg1_dim / 64;
    std::vector<int> binary_seg1(seg1_dim, 0);
    std::vector<uint64_t> all_binary_seg1(cur_degree * words_seg1, 0);

    // 6 factors: fac_a, fac_b0, fac_b1, fac_c, fac_d0, fac_d1
    float* fac = get_factor(cur_id);
    float* fac_a  = fac;
    float* fac_b0 = fac_a  + degree_bound_;
    float* fac_b1 = fac_b0 + degree_bound_;
    float* fac_c  = fac_b1 + degree_bound_;
    float* fac_d0 = fac_c  + degree_bound_;
    float* fac_d1 = fac_d0 + degree_bound_;

    for (size_t i = 0; i < cur_degree; ++i) {
        std::fill(nb_rot.begin(), nb_rot.end(), 0.0f);
        pca_rotator_.rotate(get_vector(new_neighbors[i].id), nb_rot.data());

        for (size_t d = 0; d < padded_dim_; ++d)
            residual[d] = nb_rot[d] - centroid_rot[d];

        // Centroid IP with full residual
        float ip_c_r = 0;
        float r_l2sqr = 0;
        for (size_t d = 0; d < padded_dim_; ++d) {
            ip_c_r += centroid_rot[d] * residual[d];
            r_l2sqr += residual[d] * residual[d];
        }
        fac_a[i] = r_l2sqr + 2.0f * ip_c_r;

        // --- Seg0: 4-bit CAQ on dims [0, split_dim) ---
        float o_l2sqr_0, delta_0, vmin_0, rescale_0, sum_code_0;
        caq_encode_4bit(
            residual.data(), split_dim_, codes_4bit.data(),
            &o_l2sqr_0, &delta_0, &vmin_0, &rescale_0, &sum_code_0, 6
        );
        std::memcpy(&all_codes_seg0[i * split_dim_], codes_4bit.data(), split_dim_);

        fac_b0[i] = -2.0f * rescale_0 * delta_0;
        float fac_c0 = -2.0f * rescale_0 * delta_0 * sum_code_0;
        fac_d0[i] = -2.0f * rescale_0 * (0.5f * delta_0 + vmin_0);

        // --- Seg1: 1-bit sign on dims [split_dim, padded_dim) ---
        if (seg1_dim > 0) {
            const float* r1 = &residual[split_dim_];
            float o_l2sqr_1 = 0, sum_code_1 = 0;
            float v_min_1 = r1[0], v_max_1 = r1[0];
            for (size_t d = 0; d < seg1_dim; ++d) {
                o_l2sqr_1 += r1[d] * r1[d];
                v_min_1 = std::min(v_min_1, r1[d]);
                v_max_1 = std::max(v_max_1, r1[d]);
                binary_seg1[d] = (r1[d] > 0) ? 1 : 0;
                sum_code_1 += binary_seg1[d];
            }

            // 1-bit: delta = range/2, vmin stays
            float delta_1 = (v_max_1 - v_min_1) / 2.0f;
            if (delta_1 < 1e-10f) delta_1 = 1e-10f;

            // Compute rescale = ||r1||² / <r1, oa1>
            double ip_r1_oa1 = 0;
            for (size_t d = 0; d < seg1_dim; ++d) {
                double oa = (binary_seg1[d] + 0.5) * delta_1 + v_min_1;
                ip_r1_oa1 += r1[d] * oa;
            }
            float rescale_1 = (ip_r1_oa1 > 0) ? static_cast<float>(o_l2sqr_1 / ip_r1_oa1) : 0;

            fac_b1[i] = -2.0f * rescale_1 * delta_1;
            float fac_c1 = -2.0f * rescale_1 * delta_1 * sum_code_1;
            fac_d1[i] = -2.0f * rescale_1 * (0.5f * delta_1 + v_min_1);

            // Merged fac_c = fac_c0 + fac_c1
            fac_c[i] = fac_c0 + fac_c1;

            // Pack binary for seg1
            space::pack_binary(binary_seg1.data(), &all_binary_seg1[i * words_seg1], seg1_dim);
        } else {
            fac_b1[i] = 0;
            fac_c[i] = fac_c0;
            fac_d1[i] = 0;
        }
    }

    // Pack seg0 4-bit codes
    pack_codes_4bit(split_dim_, all_codes_seg0.data(), cur_degree, get_seg0_code(cur_id));

    // Pack seg1 1-bit codes
    if (seg1_dim > 0) {
        pack_codes(seg1_dim, all_binary_seg1.data(), cur_degree, get_seg1_code(cur_id));
    }
}

inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj, const float* cur_data,
    float* appro_dist, buffer::SearchBuffer& search_pool, uint32_t cur_degree
) const {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    const auto* seg0_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* seg1_code = reinterpret_cast<const uint8_t*>(&cur_data[seg1_code_offset_]);
    const auto* factor = &cur_data[factor_offset_];

    scanner_.scan_neighbors(
        appro_dist, q_obj.lut_seg0(), q_obj.lut_seg1(),
        sqr_y, q_obj.width(), q_obj.vl_half(),
        q_obj.sum_q_seg0(), q_obj.sum_q_seg1(),
        seg0_code, seg1_code, factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) continue;
        search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), prefetch_lines_
        );
    }
    return sqr_y;
}

inline void QuantizedGraph::update_results(buffer::ResultBuffer& result_pool, const float* query) {
    if (result_pool.is_full()) return;
    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < degree_bound_; ++i) {
            PID nb = ptr_nb[i];
            if (!visited_.get(nb)) {
                visited_.set(nb);
                result_pool.insert(nb, space::l2_sqr(query, get_vector(nb), dimension_));
            }
        }
        if (result_pool.is_full()) break;
    }
}

inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();
    search_pool_.clear();

    query_obj_.prepare(query, pca_rotator_, scanner_);

    search_pool_.insert(entry_point_, FLT_MAX);
    buffer::ResultBuffer res_pool(knn);

    while (search_pool_.has_next()) {
        PID cur_node = search_pool_.pop();
        if (visited_.get(cur_node)) continue;
        visited_.set(cur_node);
        float sqr_y = scan_neighbors(
            query_obj_, get_vector(cur_node), appro_dist_.data(), search_pool_, degree_bound_
        );
        res_pool.insert(cur_node, sqr_y);
    }

    update_results(res_pool, query);
    res_pool.copy_results(results);
}

inline void QuantizedGraph::find_candidates(
    PID cur_id, size_t search_ef, std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis, const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(entry_point_, space::l2_sqr(query, get_vector(entry_point_), dimension_));

    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) continue;
        vis.set(cur_candi);
        float sqr_y = space::l2_sqr(query, get_vector(cur_candi), dimension_);
        const PID* ptr_nb = get_neighbors(cur_candi);
        auto cur_degree = degrees[cur_candi];
        for (uint32_t i = 0; i < cur_degree; ++i) {
            PID nb = ptr_nb[i];
            if (vis.get(nb)) continue;
            float dist = space::l2_sqr(query, get_vector(nb), dimension_);
            if (tmp_pool.is_full(dist)) continue;
            tmp_pool.insert(nb, dist);
        }
        if (cur_candi != cur_id)
            results.emplace_back(cur_candi, sqr_y);
    }
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    search_pool_.resize(cur_ef);
    visited_ = HashBasedBooleanSet(std::min(num_points_ / 10, cur_ef * cur_ef));
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&split_dim_), sizeof(size_t));
    pca_rotator_.save(output);
    size_t plan_size = quant_plan_.size();
    output.write(reinterpret_cast<const char*>(&plan_size), sizeof(size_t));
    for (auto& seg : quant_plan_)
        output.write(reinterpret_cast<const char*>(&seg), sizeof(QuantSegment));
    data_.save(output);
    output.close();
    std::cout << "Quantized graph saved to " << filename << "\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    if (!file_exists(filename)) { std::cerr << "Index not found!\n"; abort(); }
    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    input.read(reinterpret_cast<char*>(&split_dim_), sizeof(size_t));
    pca_rotator_.load(input);
    size_t plan_size = 0;
    input.read(reinterpret_cast<char*>(&plan_size), sizeof(size_t));
    quant_plan_.resize(plan_size);
    for (auto& seg : quant_plan_)
        input.read(reinterpret_cast<char*>(&seg), sizeof(QuantSegment));
    initialize();  // recompute offsets
    scanner_ = QGScanner(padded_dim_, split_dim_, degree_bound_);
    query_obj_ = QGQuery(padded_dim_, split_dim_);
    data_.load(input);
    input.close();
    std::cout << "Quantized graph loaded (split_dim=" << split_dim_ << ")\n";
}

}  // namespace symqg
