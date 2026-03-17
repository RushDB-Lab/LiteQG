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
#include "../quantization/rabitq.hpp"
#include "../quantization/saq_plan.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/pca_rotator.hpp"
#include "../utils/rotator.hpp"
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

    // Build phase: FHT + RaBitQ (baseline)
    FHTRotator rotator_;

    // Search phase: PCA + 4-bit SAQ
    PCARotator pca_rotator_;
    MixedPlan mixed_plan_;
    QGScanner scanner_;
    HashBasedBooleanSet visited_;
    buffer::SearchBuffer search_pool_;
    QGQuery query_obj_;
    std::vector<float> appro_dist_;
    size_t prefetch_lines_ = 0;

    // Single data array — layout depends on phase:
    //   Build: [raw_vec(dim) | RaBitQ_codes | RaBitQ_factors(3) | neighbors(deg)]
    //   After finalize: [raw_vec(dim) | SAQ_4bit_codes | SAQ_factors(4) | neighbors(deg)]
    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<float, 1 << 22, true>>
        data_;

    // Layout offsets (float units) — set for SAQ layout (superset of build layout)
    size_t code_offset_ = 0;
    size_t factor_offset_ = 0;
    size_t neighbor_offset_ = 0;
    size_t row_offset_ = 0;

    // Build-phase offsets (RaBitQ layout, within the same row allocation)
    size_t build_code_offset_ = 0;
    size_t build_factor_offset_ = 0;
    size_t build_neighbor_offset_ = 0;

    bool build_phase_ = true;  // true during build, false after finalize_saq

    void initialize();
    void copy_vectors(const float*);
    void finalize_saq();
    void encode_saq_node(PID cur_id, size_t degree);

    [[nodiscard]] float* get_vector(PID id) { return &data_.at(row_offset_ * id); }
    [[nodiscard]] const float* get_vector(PID id) const { return &data_.at(row_offset_ * id); }

    // Build-phase accessors (RaBitQ layout)
    [[nodiscard]] uint8_t* get_build_code(PID id) {
        return reinterpret_cast<uint8_t*>(&data_.at(row_offset_ * id + build_code_offset_));
    }
    [[nodiscard]] float* get_build_factor(PID id) {
        return &data_.at(row_offset_ * id + build_factor_offset_);
    }
    [[nodiscard]] PID* get_build_neighbors(PID id) {
        return reinterpret_cast<PID*>(&data_.at(row_offset_ * id + build_neighbor_offset_));
    }
    [[nodiscard]] const PID* get_build_neighbors(PID id) const {
        return reinterpret_cast<const PID*>(&data_.at(row_offset_ * id + build_neighbor_offset_));
    }

    // Search-phase accessors (SAQ layout)
    [[nodiscard]] uint8_t* get_saq_code(PID id) {
        return reinterpret_cast<uint8_t*>(&data_.at(row_offset_ * id + code_offset_));
    }
    [[nodiscard]] const uint8_t* get_saq_code(PID id) const {
        return reinterpret_cast<const uint8_t*>(&data_.at(row_offset_ * id + code_offset_));
    }
    [[nodiscard]] float* get_saq_factor(PID id) {
        return &data_.at(row_offset_ * id + factor_offset_);
    }
    [[nodiscard]] const float* get_saq_factor(PID id) const {
        return &data_.at(row_offset_ * id + factor_offset_);
    }
    [[nodiscard]] PID* get_neighbors(PID id) {
        return reinterpret_cast<PID*>(&data_.at(row_offset_ * id + neighbor_offset_));
    }
    [[nodiscard]] const PID* get_neighbors(PID id) const {
        return reinterpret_cast<const PID*>(&data_.at(row_offset_ * id + neighbor_offset_));
    }

    // Build: exact L2 candidate search + RaBitQ codes for baseline compatibility
    void find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&) const;
    void update_qg(PID, const std::vector<Candidate<float>>&);

    // Search: SAQ fascscan
    void scan_neighbors(const QGQuery&, const float*, float, float*, buffer::SearchBuffer&, uint32_t) const;

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
    , padded_dim_(1 << ceil_log2(dim))  // power of 2 for FHT compatibility
    , rotator_(dim)
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

    // SAQ layout (the one actually allocated — large enough for both phases)
    size_t saq_fascscan_dim = padded_dim_ * 4;
    size_t saq_code_floats = (degree_bound_ + kBatchSize - 1) / kBatchSize * saq_fascscan_dim * 4 / sizeof(float);

    code_offset_ = dimension_;
    factor_offset_ = code_offset_ + saq_code_floats;
    neighbor_offset_ = factor_offset_ + 4 * degree_bound_;
    row_offset_ = neighbor_offset_ + degree_bound_;

    // RaBitQ build layout (fits within the same row — offsets relative to row start)
    // RaBitQ codes are smaller: padded_dim/64 * 2 * degree = 128 floats for SIFT
    size_t rabitq_code_floats = padded_dim_ / 64 * 2 * degree_bound_;
    build_code_offset_ = dimension_;
    build_factor_offset_ = build_code_offset_ + rabitq_code_floats;
    build_neighbor_offset_ = build_factor_offset_ + 3 * degree_bound_;
    // Verify RaBitQ layout fits within SAQ row
    assert(build_neighbor_offset_ + degree_bound_ <= row_offset_);

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

    // PCA fit (used later in finalize_saq)
    pca_rotator_.fit(data, num_points_, dimension_);

    // DP for mixed-bitwidth plan (informational only for now)
    size_t kCodebookBudget = (dimension_ * 3 + 3) / 4;
    kCodebookBudget = (kCodebookBudget + 3) & ~3UL;
    mixed_plan_ = segment_dp_codebook(pca_rotator_.eigenvalues().data(), dimension_, kCodebookBudget);

    // Recompute search-phase layout for reduced SAQ dims
    size_t saq_dim = mixed_plan_.saq_dim;
    size_t saq_fascscan_dim = saq_dim * 4;
    size_t saq_code_floats = (degree_bound_ + kBatchSize - 1) / kBatchSize * saq_fascscan_dim * 4 / sizeof(float);
    code_offset_ = dimension_;
    factor_offset_ = code_offset_ + saq_code_floats;
    neighbor_offset_ = factor_offset_ + 4 * degree_bound_;
    assert(neighbor_offset_ + degree_bound_ <= row_offset_);
    prefetch_lines_ = ((neighbor_offset_ + degree_bound_) * sizeof(float) + 63) / 64;

    // Re-init scanner and query for reduced dims
    scanner_ = QGScanner(saq_dim, degree_bound_);
    query_obj_ = QGQuery(padded_dim_, saq_dim);

    std::cout << "\tVectors Copied, PCA Fitted\n";
    std::cout << "\tSearch layout: saq_dim=" << saq_dim << " code_floats=" << saq_code_floats
              << " row=" << (neighbor_offset_ + degree_bound_) << "/" << row_offset_ << " floats\n";
}

// Build-time update: RaBitQ codes + factors (baseline-compatible)
inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();
    if (cur_degree == 0) return;

    // Write neighbor IDs to BUILD position
    PID* nb_ptr = get_build_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i)
        nb_ptr[i] = new_neighbors[i].id;

    // RaBitQ encoding (FHT rotation + 1-bit binarization)
    RowMatrix<float> x_pad(cur_degree, padded_dim_);
    RowMatrix<float> c_pad(1, padded_dim_);
    x_pad.setZero();
    c_pad.setZero();

    for (size_t i = 0; i < cur_degree; ++i) {
        const auto* vec = get_vector(new_neighbors[i].id);
        std::copy(vec, vec + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    std::copy(get_vector(cur_id), get_vector(cur_id) + dimension_, &c_pad(0, 0));

    RowMatrix<float> x_rot(cur_degree, padded_dim_);
    RowMatrix<float> c_rot(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i)
        rotator_.rotate(&x_pad(i, 0), &x_rot(i, 0));
    rotator_.rotate(&c_pad(0, 0), &c_rot(0, 0));

    float* fac = get_build_factor(cur_id);
    rabitq_codes(x_rot, c_rot, get_build_code(cur_id), fac, fac + degree_bound_, fac + 2 * degree_bound_);
}

// Build-time find_candidates: use RaBitQ fascscan (baseline-compatible)
inline void QuantizedGraph::find_candidates(
    PID cur_id, size_t search_ef, std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis, const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);

    // Use FHT+RaBitQ fascscan for build (identical to baseline)
    // Create query object with FHT rotation + 6-bit quantization
    std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
    rotator_.rotate(query, rd_query.data());

    float lo, hi;
    scalar::data_range(rd_query.data(), padded_dim_, lo, hi);
    float width = (hi - lo) / ((1 << QG_BQUERY) - 1);
    if (width < 1e-10f) width = 1e-10f;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_q(padded_dim_);
    int32_t sumq = 0;
    scalar::quantize(byte_q.data(), rd_query.data(), padded_dim_, lo, width, sumq);

    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut(padded_dim_ * 4);
    pack_lut_impl(padded_dim_, byte_q.data(), lut.data());

    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);

    std::vector<uint16_t> fs_result(degree_bound_);
    std::vector<float> fs_float(degree_bound_);
    std::vector<float> appro_dist(degree_bound_);

    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) continue;
        vis.set(cur_candi);

        const float* cur_data = get_vector(cur_candi);
        float sqr_y = space::l2_sqr(query, cur_data, dimension_);

        // RaBitQ fascscan on build layout
        const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[build_code_offset_]);
        accumulate_impl(padded_dim_, packed_code, lut.data(), fs_result.data());

        // Convert: 2*result - sumq (binary → signed)
        for (size_t i = 0; i < degree_bound_; ++i)
            fs_float[i] = static_cast<float>((static_cast<int>(fs_result[i]) << 1) - sumq);

        // RaBitQ distance formula
        const float* fac = &cur_data[build_factor_offset_];
        const float* triple_x = fac;
        const float* fac_dq = &triple_x[degree_bound_];
        const float* fac_vq = &fac_dq[degree_bound_];
        for (size_t i = 0; i < degree_bound_; ++i) {
            appro_dist[i] = sqr_y + triple_x[i] + fac_dq[i] * width * fs_float[i] + fac_vq[i] * lo;
        }

        // Insert neighbors into pool
        auto cur_degree = degrees[cur_candi];
        const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[build_neighbor_offset_]);
        for (uint32_t i = 0; i < cur_degree; ++i) {
            PID nb = ptr_nb[i];
            if (tmp_pool.is_full(appro_dist[i]) || vis.get(nb)) continue;
            tmp_pool.insert(nb, appro_dist[i]);
        }

        if (cur_candi != cur_id)
            results.emplace_back(cur_candi, sqr_y);
    }
}

// Post-build: encode all nodes with PCA + 4-bit SAQ
inline void QuantizedGraph::finalize_saq() {
    std::cout << "\tEncoding SAQ (PCA + 4-bit)...\n";

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        // Copy neighbor IDs from build position to SAQ position
        const PID* build_nb = get_build_neighbors(i);
        PID* saq_nb = get_neighbors(i);
        std::memcpy(saq_nb, build_nb, degree_bound_ * sizeof(PID));

        encode_saq_node(i, degree_bound_);
    }

    build_phase_ = false;
    std::cout << "\tSAQ encoding complete\n";
}

inline void QuantizedGraph::encode_saq_node(PID cur_id, size_t degree) {
    size_t saq_dim = mixed_plan_.saq_dim;

    std::vector<float> centroid_rot(padded_dim_, 0.0f);
    pca_rotator_.rotate(get_vector(cur_id), centroid_rot.data());

    std::vector<float> nb_rot(padded_dim_, 0.0f);
    std::vector<float> residual(padded_dim_, 0.0f);
    std::vector<uint8_t> codes_4bit(saq_dim);
    std::vector<uint8_t> all_codes(degree * saq_dim, 0);

    float* fac = get_saq_factor(cur_id);
    float* fac_a = fac;
    float* fac_b = fac_a + degree_bound_;
    float* fac_c = fac_b + degree_bound_;
    float* fac_d = fac_c + degree_bound_;

    const PID* nbs = get_neighbors(cur_id);

    for (size_t i = 0; i < degree; ++i) {
        std::fill(nb_rot.begin(), nb_rot.end(), 0.0f);
        pca_rotator_.rotate(get_vector(nbs[i]), nb_rot.data());

        for (size_t d = 0; d < padded_dim_; ++d)
            residual[d] = nb_rot[d] - centroid_rot[d];

        // fac_a: exact over ALL dims (including 0-bit)
        float r_l2sqr = 0, ip_c_r = 0;
        for (size_t d = 0; d < padded_dim_; ++d) {
            r_l2sqr += residual[d] * residual[d];
            ip_c_r += centroid_rot[d] * residual[d];
        }

        // 4-bit CAQ encode only the top saq_dim PCA dims
        float o_l2sqr, delta, vmin, rescale, sum_code;
        caq_encode_4bit(residual.data(), saq_dim, codes_4bit.data(),
            &o_l2sqr, &delta, &vmin, &rescale, &sum_code, 6);

        std::memcpy(&all_codes[i * saq_dim], codes_4bit.data(), saq_dim);

        fac_a[i] = r_l2sqr + 2.0f * ip_c_r;
        fac_b[i] = -2.0f * rescale * delta;
        fac_c[i] = -2.0f * rescale * delta * sum_code;
        fac_d[i] = -2.0f * rescale * (0.5f * delta + vmin);
    }

    pack_codes_4bit(saq_dim, all_codes.data(), degree, get_saq_code(cur_id));
}

// Search: SAQ 4-bit fascscan (approx-only — sqr_y from pool, no exact L2)
inline void QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj, const float* cur_data, float sqr_y,
    float* appro_dist, buffer::SearchBuffer& pool, uint32_t cur_degree
) const {
    const auto* code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    scanner_.scan_neighbors(
        appro_dist, q_obj.lut(), sqr_y,
        q_obj.width(), q_obj.vl_half(), q_obj.sum_q_float(),
        code, factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID nb = ptr_nb[i];
        float dist = appro_dist[i];
        if (pool.is_full(dist) || visited_.get(nb)) continue;
        pool.insert(nb, dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(pool.next_id())), prefetch_lines_
        );
    }
}

inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    visited_.clear();
    search_pool_.clear();
    query_obj_.prepare(query, pca_rotator_, scanner_);

    // Entry point: exact L2 to bootstrap approx distances (1 node, negligible cost)
    float ep_dist = space::l2_sqr(query, get_vector(entry_point_), dimension_);
    search_pool_.insert(entry_point_, ep_dist);

    buffer::ResultBuffer res_pool(knn);
    while (search_pool_.has_next()) {
        auto [cur, cur_dist] = search_pool_.pop_with_dist();
        if (visited_.get(cur)) continue;
        visited_.set(cur);
        scan_neighbors(
            query_obj_, get_vector(cur), cur_dist,
            appro_dist_.data(), search_pool_, degree_bound_
        );
        res_pool.insert(cur, cur_dist);
    }
    res_pool.copy_results(results);
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    search_pool_.resize(cur_ef);
    visited_ = HashBasedBooleanSet(std::min(num_points_ / 10, cur_ef * cur_ef));
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::ofstream out(filename, std::ios::binary);
    assert(out.is_open());
    out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    pca_rotator_.save(out);
    size_t ps = mixed_plan_.segments.size();
    out.write(reinterpret_cast<const char*>(&ps), sizeof(size_t));
    for (auto& s : mixed_plan_.segments)
        out.write(reinterpret_cast<const char*>(&s), sizeof(QuantSegment));
    data_.save(out);
    out.close();
    std::cout << "Quantized graph saved to " << filename << "\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    if (!file_exists(filename)) { std::cerr << "Index not found!\n"; abort(); }
    std::ifstream in(filename, std::ios::binary);
    assert(in.is_open());
    in.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    pca_rotator_.load(in);
    size_t ps = 0;
    in.read(reinterpret_cast<char*>(&ps), sizeof(size_t));
    mixed_plan_.segments.resize(ps);
    for (auto& s : mixed_plan_.segments)
        in.read(reinterpret_cast<char*>(&s), sizeof(QuantSegment));

    // Reconstruct mixed plan from segments
    mixed_plan_.saq_dim = 0;
    for (auto& seg : mixed_plan_.segments)
        if (seg.bits == 4) mixed_plan_.saq_dim += seg.dim_len;

    // Recompute search layout
    size_t saq_dim = mixed_plan_.saq_dim > 0 ? mixed_plan_.saq_dim : padded_dim_;
    size_t saq_fascscan_dim = saq_dim * 4;
    size_t saq_code_floats = (degree_bound_ + kBatchSize - 1) / kBatchSize * saq_fascscan_dim * 4 / sizeof(float);
    code_offset_ = dimension_;
    factor_offset_ = code_offset_ + saq_code_floats;
    neighbor_offset_ = factor_offset_ + 4 * degree_bound_;
    prefetch_lines_ = ((neighbor_offset_ + degree_bound_) * sizeof(float) + 63) / 64;

    scanner_ = QGScanner(saq_dim, degree_bound_);
    query_obj_ = QGQuery(padded_dim_, saq_dim);

    data_.load(in);
    in.close();
    build_phase_ = false;
    std::cout << "Quantized graph loaded (saq_dim=" << saq_dim << ")\n";
}

}  // namespace symqg
