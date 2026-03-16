#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "../common.hpp"
#include "../third/svs/array.hpp"
#include "./memory.hpp"

namespace symqg {

class PCARotator {
   private:
    size_t dimension_ = 0;
    // Row-major rotation matrix R (D x D): R[i][j] = rotation_matrix_[i*D+j]
    std::vector<float> rotation_matrix_;
    // Column-major (transposed) copy for fast rotate: R_T[j][i] = col_matrix_[j*D+i]
    std::vector<float> col_matrix_;
    std::vector<float> eigenvalues_;

    void build_col_matrix() {
        col_matrix_.resize(dimension_ * dimension_);
        for (size_t i = 0; i < dimension_; ++i) {
            for (size_t j = 0; j < dimension_; ++j) {
                col_matrix_[j * dimension_ + i] = rotation_matrix_[i * dimension_ + j];
            }
        }
    }

   public:
    PCARotator() = default;

    explicit PCARotator(size_t dim) : dimension_(dim) {}

    void fit(const float* data, size_t num_points, size_t dim) {
        dimension_ = dim;

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

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        Eigen::VectorXd evals = solver.eigenvalues().reverse();
        Eigen::MatrixXd evecs = solver.eigenvectors().rowwise().reverse();

        eigenvalues_.resize(dim);
        for (size_t d = 0; d < dim; ++d) {
            eigenvalues_[d] = static_cast<float>(std::max(evals(d), 0.0));
        }

        rotation_matrix_.resize(dim * dim);
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                rotation_matrix_[i * dim + j] = static_cast<float>(evecs(j, i));
            }
        }

        build_col_matrix();

        std::cout << "\tPCA fitted: top eigenvalue=" << eigenvalues_[0]
                  << ", bottom=" << eigenvalues_[dim - 1] << "\n";
    }

    // Column-oriented rotate: dst = R * src, eliminates horizontal reductions.
    // Accumulates dst[:] += src[j] * R_T[j, :] for each j.
    void rotate(const float* __restrict__ src, float* __restrict__ dst) const {
        const size_t dim = dimension_;
#if defined(__AVX512F__)
        const size_t dim16 = dim / 16;

        __m512 acc[8];  // up to D=128; for larger D the compiler will spill gracefully
        for (size_t k = 0; k < dim16; ++k) {
            acc[k] = _mm512_setzero_ps();
        }

        for (size_t j = 0; j < dim; ++j) {
            __m512 s = _mm512_set1_ps(src[j]);
            const float* col = &col_matrix_[j * dim];
            for (size_t k = 0; k < dim16; ++k) {
                acc[k] = _mm512_fmadd_ps(s, _mm512_loadu_ps(&col[k * 16]), acc[k]);
            }
        }

        for (size_t k = 0; k < dim16; ++k) {
            _mm512_storeu_ps(&dst[k * 16], acc[k]);
        }

        // Scalar tail
        for (size_t i = dim16 * 16; i < dim; ++i) {
            float sum = 0;
            for (size_t j = 0; j < dim; ++j) {
                sum += col_matrix_[j * dim + i] * src[j];
            }
            dst[i] = sum;
        }

#elif defined(__AVX2__)
        const size_t dim8 = dim / 8;

        __m256 acc[16];  // up to D=128
        for (size_t k = 0; k < dim8; ++k) {
            acc[k] = _mm256_setzero_ps();
        }

        for (size_t j = 0; j < dim; ++j) {
            __m256 s = _mm256_set1_ps(src[j]);
            const float* col = &col_matrix_[j * dim];
            for (size_t k = 0; k < dim8; ++k) {
                acc[k] = _mm256_fmadd_ps(s, _mm256_loadu_ps(&col[k * 8]), acc[k]);
            }
        }

        for (size_t k = 0; k < dim8; ++k) {
            _mm256_storeu_ps(&dst[k * 8], acc[k]);
        }

        for (size_t i = dim8 * 8; i < dim; ++i) {
            float sum = 0;
            for (size_t j = 0; j < dim; ++j) {
                sum += col_matrix_[j * dim + i] * src[j];
            }
            dst[i] = sum;
        }
#else
        for (size_t i = 0; i < dim; ++i) {
            float sum = 0;
            for (size_t j = 0; j < dim; ++j) {
                sum += rotation_matrix_[i * dim + j] * src[j];
            }
            dst[i] = sum;
        }
#endif
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
        build_col_matrix();
    }
};
}  // namespace symqg
