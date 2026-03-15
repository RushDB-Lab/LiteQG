#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "../common.hpp"
#include "../third/svs/array.hpp"
#include "./memory.hpp"

namespace symqg {

// PCA rotation: data-adaptive rotation that orders dimensions by variance.
// Enables dimension segmentation (Phase 2) by concentrating variance in leading dims.
class PCARotator {
   private:
    size_t dimension_ = 0;
    // Rotation matrix R (D x D), stored row-major.
    // Rows are eigenvectors sorted by descending eigenvalue.
    std::vector<float> rotation_matrix_;
    // Per-dimension variances in rotated space (eigenvalues, descending order)
    std::vector<float> eigenvalues_;

   public:
    PCARotator() = default;

    explicit PCARotator(size_t dim) : dimension_(dim) {}

    // Fit PCA on dataset: compute covariance → eigendecomposition → store R and eigenvalues.
    void fit(const float* data, size_t num_points, size_t dim) {
        dimension_ = dim;

        // Compute mean
        std::vector<double> mean(dim, 0.0);
        for (size_t i = 0; i < num_points; ++i) {
            const float* vec = data + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                mean[d] += vec[d];
            }
        }
        double inv_n = 1.0 / static_cast<double>(num_points);
        for (size_t d = 0; d < dim; ++d) {
            mean[d] *= inv_n;
        }

        // Compute covariance matrix using Eigen
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(dim, dim);

#pragma omp parallel
        {
            Eigen::MatrixXd local_cov = Eigen::MatrixXd::Zero(dim, dim);
#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < num_points; ++i) {
                Eigen::VectorXd centered(dim);
                const float* vec = data + i * dim;
                for (size_t d = 0; d < dim; ++d) {
                    centered(d) = vec[d] - mean[d];
                }
                local_cov.noalias() += centered * centered.transpose();
            }
#pragma omp critical
            cov += local_cov;
        }
        cov *= inv_n;

        // Eigendecomposition (symmetric → SelfAdjointEigenSolver)
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        // Eigenvalues come in ascending order; we want descending
        Eigen::VectorXd evals = solver.eigenvalues().reverse();
        Eigen::MatrixXd evecs = solver.eigenvectors().rowwise().reverse();

        // Store eigenvalues
        eigenvalues_.resize(dim);
        for (size_t d = 0; d < dim; ++d) {
            eigenvalues_[d] = static_cast<float>(std::max(evals(d), 0.0));
        }

        // Store rotation matrix: R[i][j] = evecs(j, i) (column i of evecs → row i of R)
        // So rotate(x) = R * x where R rows are eigenvectors
        rotation_matrix_.resize(dim * dim);
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                rotation_matrix_[i * dim + j] = static_cast<float>(evecs(j, i));
            }
        }

        std::cout << "\tPCA fitted: top eigenvalue=" << eigenvalues_[0]
                  << ", bottom=" << eigenvalues_[dim - 1] << "\n";
    }

    // Rotate a single vector: dst = R * src  (O(D²))
    void rotate(const float* __restrict__ src, float* __restrict__ dst) const {
        for (size_t i = 0; i < dimension_; ++i) {
            float sum = 0;
            const float* row = &rotation_matrix_[i * dimension_];
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            size_t j = 0;
            for (; j + 16 <= dimension_; j += 16) {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(&row[j]),
                    _mm512_loadu_ps(&src[j]),
                    acc
                );
            }
            sum = _mm512_reduce_add_ps(acc);
            for (; j < dimension_; ++j) {
                sum += row[j] * src[j];
            }
#elif defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            size_t j = 0;
            for (; j + 8 <= dimension_; j += 8) {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(&row[j]),
                    _mm256_loadu_ps(&src[j]),
                    acc
                );
            }
            // horizontal sum
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ps(s, _mm_movehdup_ps(s));
            sum = _mm_cvtss_f32(s);
            for (; j < dimension_; ++j) {
                sum += row[j] * src[j];
            }
#else
            for (size_t j = 0; j < dimension_; ++j) {
                sum += row[j] * src[j];
            }
#endif
            dst[i] = sum;
        }
    }

    [[nodiscard]] size_t dimension() const { return dimension_; }
    [[nodiscard]] const std::vector<float>& eigenvalues() const { return eigenvalues_; }

    void save(std::ofstream& output) const {
        output.write(reinterpret_cast<const char*>(&dimension_), sizeof(size_t));
        output.write(
            reinterpret_cast<const char*>(rotation_matrix_.data()),
            static_cast<std::streamsize>(dimension_ * dimension_ * sizeof(float))
        );
        output.write(
            reinterpret_cast<const char*>(eigenvalues_.data()),
            static_cast<std::streamsize>(dimension_ * sizeof(float))
        );
    }

    void load(std::ifstream& input) {
        input.read(reinterpret_cast<char*>(&dimension_), sizeof(size_t));
        rotation_matrix_.resize(dimension_ * dimension_);
        eigenvalues_.resize(dimension_);
        input.read(
            reinterpret_cast<char*>(rotation_matrix_.data()),
            static_cast<std::streamsize>(dimension_ * dimension_ * sizeof(float))
        );
        input.read(
            reinterpret_cast<char*>(eigenvalues_.data()),
            static_cast<std::streamsize>(dimension_ * sizeof(float))
        );
    }
};
}  // namespace symqg
