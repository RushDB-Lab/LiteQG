#pragma once

#if defined(_MSC_VER)  // Visual studio
#define thread_local __declspec(thread)
#elif defined(__GCC__)  // GCC
#define thread_local __thread
#endif

#include <ctime>
#include <random>
#include <thread>

namespace symqg {
template <typename T>
inline T rand_integer(T min, T max) {
    static thread_local std::mt19937* generator = nullptr;
    if (generator == nullptr) {
        generator = new std::mt19937(
            std::clock() + std::hash<std::thread::id>()(std::this_thread::get_id())
        );
    }
    std::uniform_int_distribution<T> distribution(min, max);
    return distribution(*generator);
}

constexpr size_t div_round_up(size_t val, size_t div) {
    return (val / div) + static_cast<size_t>((val % div) != 0);
}

constexpr size_t round_up2multiple(size_t val, size_t multiple_of) {
    return multiple_of * (div_round_up(val, multiple_of));
}

inline size_t log2_roundup(size_t val) {
    size_t res = 0;
    for (size_t i = 0; i < 31; ++i) {
        if ((1U << i) >= val) {
            res = i;
            break;
        }
    }
    return res;
}

// get number of threads of current sys
inline size_t total_threads() {
    const auto threads = std::thread::hardware_concurrency();
    return threads;
}
}  // namespace symqg