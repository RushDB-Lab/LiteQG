#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

#include "../utils/tools.hpp"

namespace symqg::memory {
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

template <typename Type, size_t AlignBytes = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(AlignBytes >= alignof(Type));

   public:
    using value_type = Type;
    static std::align_val_t constexpr kAlignment{AlignBytes};

    template <class OtherType>
    struct rebind {
        using other = AlignedAllocator<OtherType, AlignBytes>;
    };

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename OtherType>
    constexpr explicit AlignedAllocator(AlignedAllocator<
                                        OtherType,
                                        AlignBytes> const&) noexcept {}

    [[nodiscard]] value_type* allocate(std::size_t elements) {
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(value_type)) {
            throw std::bad_array_new_length();
        }

        auto const nbytes = round_up2multiple(
            elements * sizeof(value_type), static_cast<size_t>(kAlignment)
        );
        auto* ptr = ::operator new[](nbytes, kAlignment);
        if (HugePage) {
            madvise(ptr, nbytes, MADV_HUGEPAGE);
        }
        return reinterpret_cast<value_type*>(ptr);
    }

    void deallocate(value_type* ptr, [[maybe_unused]] std::size_t bytes) {
        ::operator delete[](ptr, kAlignment);
    }
    auto operator!=(const AlignedAllocator& other) {
        return other.kAlignment != this->kAlignment;
    }
    auto operator==(const AlignedAllocator& other) {
        return other.kAlignment == this->kAlignment;
    }
};

template <typename Type>
struct Allocator {
   public:
    using value_type = Type;

    constexpr Allocator() noexcept = default;

    template <typename OtherType>
    explicit constexpr Allocator(const Allocator<OtherType>&) noexcept {}

    [[nodiscard]] constexpr value_type* allocate(std::size_t n) {
        return static_cast<value_type*>(
            ::operator new(n * sizeof(value_type), std::align_val_t(alignof(value_type)))
        );
    }

    constexpr void deallocate(value_type* ptr, size_t count) noexcept {
        ::operator delete(ptr, count);
    }

    // Intercept zero-argument construction to do default initialization.
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

template <size_t Alignment>
inline void* align_allocate(size_t nbytes, bool huge_page = false) {
    size_t size = round_up2multiple(nbytes, Alignment);
    void* ptr = std::aligned_alloc(Alignment, size);
    if (huge_page) {
        madvise(ptr, nbytes, MADV_HUGEPAGE);
    }
    std::memset(ptr, 0, size);
    return ptr;
}

static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

inline void mem_prefetch_l2(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}
}  // namespace symqg::memory
