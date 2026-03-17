#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

struct Index {
    std::unique_ptr<symqg::QuantizedGraph> index = nullptr;

    explicit Index(
        const std::string& index_type,
        const std::string& metric,
        size_t num_points,
        size_t dim,
        size_t degree
    ) {
        if (metric != "L2") {
            std::cerr << "Only L2 distance supported currently\n";
            return;
        }

        if (degree < 32 || degree % 32 != 0) {
            std::cerr << "The degree bound must be a multiple of 32\n";
            return;
        }

        if (index_type == "QG") {
            index = std::make_unique<symqg::QuantizedGraph>(num_points, degree, dim);
        } else {
            std::cerr << "Index type [" << index_type << "] not supported\n";
            return;
        }
    }

    void load(const std::string& filename) const { index->load_index(filename.c_str()); }

    void save(const std::string& filename) const { index->save_index(filename.c_str()); }

    void set_ef(size_t ef_search) const { index->set_ef(ef_search); }

    void build_index(
        nb::ndarray<float, nb::c_contig, nb::device::cpu> items,
        size_t ef_indexing,
        size_t num_iter = 3,
        size_t num_threads = UINT_MAX
    ) const {
        size_t num = items.shape(0);
        size_t dim = (items.ndim() == 2) ? items.shape(1) : items.shape(0);
        if (items.ndim() == 1) {
            num = 1;
        }

        if (num != index->num_vertices() || dim != index->dimension()) {
            std::cerr
                << "The shape of data is different with initialization! Expected shape: ("
                << index->num_vertices() << ", " << index->dimension() << "), but got: ("
                << num << ", " << dim << ")\n";
            return;
        }
        symqg::QGBuilder builder(*index, ef_indexing, items.data(), num_threads);
        builder.build(num_iter);
        std::cout << "\tQuantizedGraph created\n";
    }

    nb::ndarray<nb::numpy, uint32_t> search(
        nb::ndarray<float, nb::c_contig, nb::device::cpu> query,
        uint32_t knn
    ) const {
        auto* result = new uint32_t[knn];
        index->search(query.data(), knn, result);

        size_t shape[1] = {knn};
        nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
        return nb::ndarray<nb::numpy, uint32_t>(result, 1, shape, std::move(owner));
    }
};

NB_MODULE(symphonyqg, m) {
    m.doc() = "Towards Symphonious Integration of Graph and Quantization";

    nb::class_<Index>(m, "Index")
        .def(
            nb::init<const std::string&, const std::string&, size_t, size_t, size_t>(),
            "index_type"_a,
            "metric"_a,
            "num_elements"_a,
            "dimension"_a,
            "degree_bound"_a = 32
        )
        .def("load", &Index::load, "filename"_a)
        .def("save", &Index::save, "filename"_a)
        .def("set_ef", &Index::set_ef, "EF"_a)
        .def(
            "build_index",
            &Index::build_index,
            "data"_a,
            "EF"_a,
            "num_iter"_a = 3,
            "num_thread"_a = UINT_MAX
        )
        .def("search", &Index::search, "query"_a, "k"_a);
}
