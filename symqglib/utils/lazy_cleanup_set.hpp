#include <unordered_map>
#include <vector>

namespace symqg {

template<class Value,
        class Hash = std::hash<Value>,
        class Pred = std::equal_to<Value>,
        class Alloc = std::allocator<std::pair<const Value, size_t>>>
class LazyCleanupSet {

public:
    LazyCleanupSet() = default;

    LazyCleanupSet(size_t size) : m(size) {
      c.reserve(size);
    }

    bool contains(const Value &v) const {
      auto it = m.find(v);
      if (it == m.end()) {
        return false;
      }
      return c[it->second];
    }
    symqglib/utils/lazy_cleanup_set.hpp
    void emplace(const Value &v) {
      auto it = m.find(v);
      if (it == m.end()) {
        m[v] = c.size();
        c.push_back(true);
        return;
      }
      c[it->second] = true;
    }

    void clear() { std::fill(c.begin(), c.end(), false); }

private:
    std::unordered_map<Value, size_t, Hash, Pred, Alloc> m;
    std::vector<bool> c{};
};

} // namespace symqg