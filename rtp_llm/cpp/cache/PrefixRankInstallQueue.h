#pragma once

#include <deque>
#include <future>
#include <mutex>
#include <unordered_map>

#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"

namespace rtp_llm {

class PrefixRankInstallQueue {
public:
    struct Collective {
        std::function<void(torch::Tensor&)>                    broadcast;
        std::function<bool(bool)>                              all_ready;
        std::function<void()>                                  synchronize;
        std::function<void(const std::vector<torch::Tensor>&)> broadcast_data;
    };
    using Describe = std::function<std::vector<BlockInfo>(BlockIdxType)>;

    PrefixRankInstallQueue(PrefixCacheTransactionManager& manager,
                           size_t                         max_pending             = 64,
                           size_t                         max_blocks_per_step     = 32,
                           size_t                         max_bytes_per_step      = 32 * 1024 * 1024,
                           size_t                         max_chunks_per_boundary = 1);
    ~PrefixRankInstallQueue();
    std::shared_future<bool>
         enqueue(uint64_t id, uint64_t epoch, uint64_t version, const PrefixControllerIdentity& controller);
    bool pending() const;
    void close();
    bool drained() const;
    bool reopen();
    void process(int rank, int rank_count, bool supported, const Describe& describe, const Collective& collective);

private:
    struct Entry {
        PrefixCacheTransactionManager::ImportOperation operation;
        PrefixCacheTransactionManager::ResourceLease   pin;
        std::promise<bool>                             promise;
        std::shared_future<bool>                       future    = promise.get_future().share();
        size_t                                         cursor    = 0;
        bool                                           completed = false;
    };
    void complete(const std::shared_ptr<Entry>& entry, bool success);
    bool current(const Entry& entry) const;
    void
    processActive(int rank, int rank_count, bool supported, const Describe& describe, const Collective& collective);

    PrefixCacheTransactionManager&                       manager_;
    const size_t                                         max_pending_;
    const size_t                                         max_blocks_per_step_;
    const size_t                                         max_bytes_per_step_;
    const size_t                                         max_chunks_per_boundary_;
    mutable std::mutex                                   mutex_;
    bool                                                 closed_     = false;
    bool                                                 processing_ = false;
    std::deque<std::shared_ptr<Entry>>                   pending_;
    std::unordered_map<uint64_t, std::shared_ptr<Entry>> entries_;
    std::shared_ptr<Entry>                               active_;
};

}  // namespace rtp_llm
