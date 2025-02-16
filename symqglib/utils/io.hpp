#pragma once

#include <sys/stat.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace symqg {
inline size_t get_filesize(const char* filename) {
    struct stat64 stat_buf;
    int tmp = stat64(filename, &stat_buf);
    return tmp == 0 ? stat_buf.st_size : -1;
}

inline bool file_exist(const char* filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        file.close();
        return false;
    }
    file.close();
    return true;
}

template <typename T, class M>
void load_vecs(const char* filename, M& row_mat) {
    if (!file_exist(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert((std::is_same_v<T*, decltype(row_mat.data())> == true));

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    row_mat = M(rows, cols);

    input.seekg(0, std::ifstream::beg);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}

template <typename T, class M>
void load_bin(const char* filename, M& row_mat) {
    if (!file_exist(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert((std::is_same_v<T*, decltype(row_mat.data)> == true));

    uint32_t rows;
    uint32_t cols;
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    row_mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}
}  // namespace symqg
