#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtp_llm {

struct PrefixCopyRecord {
    uint64_t    generation       = 0;
    uint64_t    operation_id     = 0;
    uint64_t    import_epoch     = 0;
    uint64_t    cache_key        = 0;
    int32_t     block_index      = 0;
    int64_t     published_ns     = -1;
    int64_t     first_reuse_ns   = -1;
    int64_t     evicted_ns       = -1;
    int64_t     physical_free_ns = -1;
    int64_t     first_request_id = 0;
    std::string first_attempt_id;
    bool        reused             = false;
    bool        evicted            = false;
    bool        physically_freed   = false;
    bool        retired            = false;
    bool        tracking_valid     = true;
    uint64_t    pending_executions = 0;
};

struct PrefixCopySnapshot {
    int64_t                       sampled_ns               = -1;
    uint64_t                      published_blocks         = 0;
    uint64_t                      reused_blocks            = 0;
    uint64_t                      wasted_blocks            = 0;
    uint64_t                      unknown_retired_blocks   = 0;
    uint64_t                      dropped_terminal_records = 0;
    bool                          counters_saturated       = false;
    std::vector<PrefixCopyRecord> records;
};

class PrefixCopyLedger: public std::enable_shared_from_this<PrefixCopyLedger> {
    struct State {
        PrefixCopyRecord record;
        int64_t          last_ns = -1;
    };

public:
    using Now = std::function<int64_t()>;

    class Use {
    public:
        ~Use();
        void finish(bool success);
        Use(const Use&)            = delete;
        Use& operator=(const Use&) = delete;

    private:
        friend class PrefixCopyLedger;
        Use(std::shared_ptr<PrefixCopyLedger>   owner,
            std::vector<std::shared_ptr<State>> records,
            int64_t                             request_id,
            std::string                         attempt_id);
        std::shared_ptr<PrefixCopyLedger>   owner_;
        std::vector<std::shared_ptr<State>> records_;
        int64_t                             request_id_;
        std::string                         attempt_id_;
    };

    explicit PrefixCopyLedger(size_t terminal_limit = 4096, Now now = monotonicNanos);
    void
    publish(uint64_t generation, uint64_t operation_id, uint64_t import_epoch, uint64_t cache_key, int32_t block_index);
    void                 evict(uint64_t generation);
    void                 invalidate();
    void                 physicalFree(int32_t block_index);
    std::unique_ptr<Use> capture(const std::vector<int32_t>& blocks, int64_t request_id, const std::string& attempt_id);
    PrefixCopySnapshot   snapshot() const;

private:
    static int64_t monotonicNanos();
    int64_t        elapsed() const;
    void           timestamp(State& state, int64_t& destination);
    void           finish(const std::vector<std::shared_ptr<State>>& records,
                          bool                                       success,
                          int64_t                                    request_id,
                          const std::string&                         attempt_id);
    void           retire(const std::shared_ptr<State>& state);
    void           increment(uint64_t& counter);

    size_t                                               terminal_limit_;
    Now                                                  now_;
    int64_t                                              origin_ns_;
    mutable std::mutex                                   mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<State>> live_;
    std::unordered_map<int32_t, uint64_t>                physical_;
    std::deque<PrefixCopyRecord>                         terminal_;
    PrefixCopySnapshot                                   totals_;
};

}  // namespace rtp_llm
