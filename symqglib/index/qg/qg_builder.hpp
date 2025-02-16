#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "../../common.hpp"
#include "../../space/space.hpp"
#include "../../third/ngt/hashset.hpp"
#include "../../utils/tools.hpp"
#include "./qg.hpp"

namespace symqg {

// 定义最大二分搜索迭代次数
constexpr size_t kMaxBsIter = 5;

// 定义候选列表类型，存储浮点数类型的候选项
using CandidateList = std::vector<Candidate<float>>;

class QGBuilder {
   private:
    QuantizedGraph& qg_;                   // 引用到量化图对象
    size_t EF_;                            // 构建过程中的EF参数，控制候选列表大小
    size_t num_threads_;                   // 使用的线程数
    size_t num_nodes_;                     // 图中的节点总数
    size_t dim_;                           // 数据向量的维度
    size_t max_degree_;                    // 每个节点的最大邻居数
    size_t max_candidate_pool_size_ = 750; // 候选池的最大大小
    size_t max_pruned_size_ = 300;         // 被修剪邻居的最大大小
    DistFunc<float> dist_func_;            // 距离函数，用于计算距离
    std::vector<CandidateList> new_neighbors_;   // 存储新邻居的列表
    std::vector<CandidateList> pruned_neighbors_; // 存储被修剪邻居的列表

    // 初始化阶段，随机分配邻居
    void random_init();

    // 更新所有节点的邻居列表
    void update_neighbors(bool refine);

    // 使用启发式方法修剪候选邻居
    void heuristic_prune(PID, CandidateList&, CandidateList&, bool);

    // 添加反向边，确保图的对称性
    void add_reverse_edges(PID, std::vector<std::mutex>&, bool);

    // 从被修剪的邻居中添加边到新的邻居列表
    void add_pruned_edges(
        const CandidateList&, const CandidateList&, CandidateList&, float
    );

    // 对图进行进一步优化和精炼
    void graph_refine();

   public:
    /**
     * @brief 构造函数，初始化QGBuilder
     * 
     * @param index 引用到量化图对象
     * @param ef_build 构建过程中的EF参数
     * @param data 数据集指针
     * @param num_threads 使用的线程数
     */
    explicit QGBuilder(
        QuantizedGraph& index, uint32_t ef_build, const float* data, size_t num_threads
    )
        : qg_{index}
        , EF_{ef_build}
        , num_threads_{std::min(num_threads, total_threads())}
        , num_nodes_{qg_.num_vertices()}
        , dim_{qg_.dimension()}
        , max_degree_(qg_.degree_bound())
        , dist_func_{space::l2_sqr}
        , new_neighbors_(qg_.num_vertices())
        , pruned_neighbors_(qg_.num_vertices()) {
        omp_set_num_threads(static_cast<int>(num_threads_));
        // 计算中位点
        std::vector<float> medoid =
            space::compute_medioid(data, num_nodes_, dim_, num_threads_);
        // 计算入口点
        PID entry_point = space::compute_entrypoint(
            data, medoid.data(), num_nodes_, dim_, num_threads_, dist_func_
        );
        std::cout << "Setting entry_point to " << entry_point << '\n' << std::flush;
        qg_.set_ep(entry_point);
        // 复制数据向量到量化图
        qg_.copy_vectors(data);
        // 进行随机初始化
        random_init();
    }

    /**
     * @brief 构建量化图，包括更新邻居、添加反向边和优化图结构
     * 
     * @param refine 是否进行图优化
     */
    void build(bool refine) {
        if (refine) {
            // 如果需要优化，预先分配被修剪邻居的空间
            for (size_t i = 0; i < num_nodes_; ++i) {
                pruned_neighbors_[i].clear();
                pruned_neighbors_[i].reserve(max_pruned_size_);
            }
        }

        // 更新所有节点的邻居列表
        update_neighbors(refine);

        // 创建互斥锁列表，用于线程安全地添加反向边
        std::vector<std::mutex> locks(num_nodes_);
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_nodes_; ++i) {
            add_reverse_edges(i, locks, refine);
        }

        // 如果需要优化，进行图的精炼
        if (refine) {
            graph_refine();
        }

        // 并行更新量化图中的邻居列表
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_nodes_; ++i) {
            qg_.update_qg(i, new_neighbors_[i]);
        }
    }

    /**
     * @brief 检查图中是否存在重复的边
     */
    void check_dup() const {
#pragma omp parallel for
        for (size_t i = 0; i < num_nodes_; ++i) {
            std::unordered_set<PID> edges;
            for (auto nei : new_neighbors_[i]) {
                if (edges.find(nei.id) != edges.end()) {
                    std::cout << "dup edges\n";
                }
                edges.emplace(nei.id);
            }
        }
    }

    /**
     * @brief 计算图中每个节点的平均度数
     * 
     * @return float 平均度数
     */
    [[nodiscard]] auto avg_degree() const -> float {
        size_t degrees = 0;
        for (size_t i = 0; i < num_nodes_; ++i) {
            degrees += qg_.get_degree(i);
        }
        float res = static_cast<float>(degrees) / static_cast<float>(num_nodes_);
        return res;
    }
};

/**
 * @brief 从被修剪的邻居列表中添加边到新的邻居列表，基于余弦相似度阈值
 * 
 * @param result 当前的邻居列表
 * @param pruned_list 被修剪的邻居列表
 * @param new_result 新的邻居列表
 * @param threshold 余弦相似度阈值
 */
inline void QGBuilder::add_pruned_edges(
    const CandidateList& result,
    const CandidateList& pruned_list,
    CandidateList& new_result,
    float threshold
) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    // 遍历被修剪的邻居，尝试添加到新的邻居列表中
    while (new_result.size() < max_degree_ && start < pruned_list.size()) {
        const auto& cur = pruned_list[start];
        bool occlude = false;
        const float* cur_data = qg_.get_vector(cur.id);
        float dik_sqr = cur.distance;
        // 检查当前候选是否被已有邻居遮蔽
        for (auto& nei : new_result) {
            if (cur.id == nei.id) {
                occlude = true;
                break;
            }
            float dij_sqr = nei.distance;
            if (dij_sqr > dik_sqr) {
                break;
            }
            float djk_sqr = dist_func_(qg_.get_vector(nei.id), cur_data, dim_);
            float cosine =
                (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
            if (cosine > threshold) {
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            new_result.emplace_back(cur);
            std::sort(new_result.begin(), new_result.end());
        }

        ++start;
    }
}

/**
 * @brief 使用启发式方法修剪候选邻居列表，保留最优的邻居
 * 
 * @param cur_id 当前节点的ID
 * @param pool 候选邻居列表
 * @param pruned_results 修剪后的邻居结果
 * @param refine 是否进行进一步优化
 */
inline void QGBuilder::heuristic_prune(
    PID cur_id, CandidateList& pool, CandidateList& pruned_results, bool refine
) {
    if (pool.empty()) {
        return;
    }
    pruned_results.clear();
    size_t poolsize = pool.size();

    // 如果候选池大小不超过最大度数，直接保留所有候选
    if (poolsize <= max_degree_) {
        for (auto&& nei : pool) {
            pruned_results.emplace_back(nei);
        }
        return;
    }

    // 使用标记数组记录被修剪的候选
    std::vector<bool> pruned(poolsize, false);
    size_t start = 0;

    // 遍历候选列表，选择合适的邻居
    while (pruned_results.size() < max_degree_ && start < poolsize) {
        auto candidate_id = pool[start].id;
        if (pruned[start] || candidate_id == cur_id) {
            ++start;
            continue;
        }

        pruned_results.emplace_back(pool[start]);
        const float* data_j = qg_.get_vector(candidate_id);

        // 检查其他候选是否应被遮蔽
        for (size_t i = start + 1; i < poolsize; ++i) {
            if (pruned[i]) {
                continue;
            }
            float dik = pool[i].distance;
            auto djk = dist_func_(data_j, qg_.get_vector(pool[i].id), dim_);

            if (djk < dik) {
                if (refine && pruned_neighbors_[cur_id].size() < max_pruned_size_) {
                    pruned_neighbors_[cur_id].emplace_back(pool[i]);
                }
                pruned[i] = true;
            }
        }

        ++start;
    }
}

/**
 * @brief 更新所有节点的邻居列表，包括寻找候选和修剪
 * 
 * @param refine 是否进行修剪优化
 */
inline void QGBuilder::update_neighbors(bool refine) {
    // 为每个线程创建一个访问集合，用于记录已访问的节点
    std::vector<HashBasedBooleanSet> visited_list(
        num_threads_, HashBasedBooleanSet(std::max(EF_ * EF_, num_nodes_ / 10))
    );
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        PID cur_id = i;
        auto tid = omp_get_thread_num();
        CandidateList candidates;
        HashBasedBooleanSet& vis = visited_list[tid];
        candidates.reserve(2 * max_candidate_pool_size_);
        vis.clear();
        const float* query = qg_.get_vector(cur_id);
        // 查找候选邻居
        qg_.find_candidates(query, EF_, candidates, vis);

        // 添加当前已有的邻居到候选列表中
        for (auto& nei : new_neighbors_[cur_id]) {
            auto neighbor_id = nei.id;
            if (neighbor_id != cur_id && !vis.get(neighbor_id)) {
                candidates.emplace_back(nei);
            }
        }

        // 对候选列表进行部分排序，保留最小的max_candidate_pool_size_个
        size_t min_size = std::min(candidates.size(), max_candidate_pool_size_);
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<long>(min_size),
            candidates.end()
        );
        candidates.resize(min_size);

        // 使用启发式方法修剪候选列表
        heuristic_prune(cur_id, candidates, new_neighbors_[cur_id], refine);
    }
}

/**
 * @brief 添加反向边，确保图的对称性
 * 
 * @param data_id 当前节点的ID
 * @param locks 互斥锁列表，用于线程安全
 * @param refine 是否进行修剪优化
 */
inline void QGBuilder::add_reverse_edges(
    PID data_id, std::vector<std::mutex>& locks, bool refine
) {
    for (auto&& nei : new_neighbors_[data_id]) {
        PID dst = nei.id;
        bool dup = false;
        CandidateList& dst_neighbors = new_neighbors_[dst];
        // 加锁以确保线程安全地修改目标节点的邻居列表
        std::lock_guard lock(locks[dst]);
        for (auto& nei : dst_neighbors) {
            if (nei.id == data_id) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }

        // 如果目标节点的邻居数量未达到最大度数，直接添加反向边
        if (dst_neighbors.size() < max_degree_) {
            dst_neighbors.emplace_back(data_id, nei.distance);
        } else {
            // 如果已达到最大度数，尝试通过修剪添加新的边
            CandidateList tmp_pool = dst_neighbors;
            tmp_pool.reserve(max_degree_ + 1);
            tmp_pool.emplace_back(data_id, nei.distance);
            std::sort(tmp_pool.begin(), tmp_pool.end());
            heuristic_prune(dst, tmp_pool, dst_neighbors, refine);
        }
    }
}

/**
 * @brief 随机初始化每个节点的邻居列表
 * 
 * 该函数为每个节点随机选择邻居，计算距离，并更新量化图。
 */
inline void QGBuilder::random_init() {
    const PID min_id = 0;
    const PID max_id = num_nodes_ - 1;
#pragma omp parallel for
    for (size_t i = 0; i < num_nodes_; ++i) {
        std::unordered_set<PID> neighbor_set;
        neighbor_set.reserve(max_degree_);
        // 随机选择邻居，确保不自连
        while (neighbor_set.size() < max_degree_) {
            PID rand_id = rand_integer<PID>(min_id, max_id);
            if (rand_id != i) {
                neighbor_set.emplace(rand_id);
            }
        }

        const float* cur_data = qg_.get_vector(i);
        new_neighbors_[i].reserve(max_degree_);
        // 计算并存储每个邻居的距离
        for (PID cur_neigh : neighbor_set) {
            new_neighbors_[i].emplace_back(
                cur_neigh, dist_func_(cur_data, qg_.get_vector(cur_neigh), dim_)
            );
        }
    }

    // 并行更新量化图中的邻居列表
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        qg_.update_qg(i, new_neighbors_[i]);
    }
}

/**
 * @brief 对图进行精炼，补充边以增强图的连通性
 * 
 * 该函数尝试从被修剪的邻居中添加边，并通过二分搜索调整余弦相似度阈值。
 * 如果仍然不足，则随机添加边以满足最大度数要求。
 */
inline void QGBuilder::graph_refine() {
    std::cout << "Supplementing edges...\n";

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        CandidateList& cur_neighbors = new_neighbors_[i];
        size_t cur_degree = cur_neighbors.size();
        // 如果当前度数已经达到最大，不需要补充
        if (cur_degree >= max_degree_) {
            continue;
        }

        CandidateList& pruned_list = pruned_neighbors_[i];
        CandidateList new_result;
        new_result.reserve(max_degree_);

        float left = 0.5;  // 余弦相似度下界
        float right = 1.0; // 余弦相似度上界
        size_t iter = 0;

        // 按距离排序被修剪的邻居
        std::sort(pruned_list.begin(), pruned_list.end());

        // 进行二分搜索以找到合适的余弦相似度阈值
        while (iter++ < kMaxBsIter) {
            float mid = (left + right) / 2;
            add_pruned_edges(cur_neighbors, pruned_list, new_result, mid);
            if (new_result.size() < max_degree_) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // 如果仍然不足，尝试使用上界阈值添加边
        if (new_result.size() < max_degree_) {
            add_pruned_edges(cur_neighbors, pruned_list, new_result, right);
            // 如果仍然不足，随机添加边以满足最大度数
            if (new_result.size() < max_degree_) {
                std::unordered_set<PID> ids;
                ids.reserve(max_degree_);
                for (auto& neighbor : new_result) {
                    ids.emplace(neighbor.id);
                }
                while (new_result.size() < max_degree_) {
                    PID rand_id = rand_integer<PID>(0, static_cast<PID>(num_nodes_) - 1);
                    if (rand_id != static_cast<PID>(i) && ids.find(rand_id) == ids.end()) {
                        new_result.emplace_back(
                            rand_id,
                            dist_func_(qg_.get_vector(rand_id), qg_.get_vector(i), dim_)
                        );
                        ids.emplace(rand_id);
                    }
                }
            }
        }

        // 更新当前节点的邻居列表
        cur_neighbors = new_result;
    }
    std::cout << "Supplementing finished...\n";
}

}  // namespace symqg
