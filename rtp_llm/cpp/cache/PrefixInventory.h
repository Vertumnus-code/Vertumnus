#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "rtp_llm/cpp/cache/PrefixCopyLedger.h"

namespace rtp_llm {

struct PrefixReuseEvent {
    int64_t     sequence;
    int64_t     request_id;
    int64_t     device_reuse_tokens;
    int64_t     input_tokens;
    bool        prefill = true;
    std::string attempt_id;
};

struct PrefixInventoryReport {
    std::string                   namespace_id;
    std::string                   worker_id;
    int                           dp_lane = 0;
    std::string                   epoch;
    std::string                   layout_fingerprint;
    std::string                   snapshot_kind               = "FULL";
    int64_t                       version                     = 0;
    int64_t                       rank_readiness_generation   = 0;
    bool                          device_ready                = false;
    int                           cp_degree                   = 1;
    int                           block_size                  = 0;
    int64_t                       physical_free_bytes         = 0;
    int64_t                       physical_block_bytes        = 0;
    int64_t                       physical_capacity_bytes     = 0;
    int64_t                       evictable_bytes             = 0;
    int64_t                       active_bytes                = 0;
    int64_t                       resident_bytes              = 0;
    int64_t                       operation_reserved_bytes    = 0;
    bool                          physical_memory_stats_valid = false;
    PrefixCopySnapshot            copy_lifetime;
    std::vector<int64_t>          device_keys;
    std::vector<PrefixReuseEvent> reuse_events;
    int64_t                       dropped_reuse_events = 0;
};

class PrefixInventory {
public:
    void record(int64_t request_id, int64_t reuse_tokens, int64_t input_tokens, const std::string& attempt_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.size() == 4096) {
            events_.pop_front();
            ++dropped_;
        }
        events_.push_back({++sequence_, request_id, reuse_tokens, input_tokens, true, attempt_id});
    }

    PrefixInventoryReport snapshot(PrefixInventoryReport report) {
        std::sort(report.device_keys.begin(), report.device_keys.end());
        return snapshotSorted(std::move(report));
    }

    PrefixInventoryReport snapshotSorted(PrefixInventoryReport report) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (keys_ != report.device_keys || ready_ != report.device_ready) {
            keys_  = report.device_keys;
            ready_ = report.device_ready;
            ++version_;
        }
        report.version = version_;
        report.reuse_events.assign(events_.begin(), events_.end());
        report.dropped_reuse_events = dropped_;
        return report;
    }

private:
    std::mutex                   mutex_;
    std::vector<int64_t>         keys_;
    std::deque<PrefixReuseEvent> events_;
    bool                         ready_    = false;
    int64_t                      version_  = 0;
    int64_t                      sequence_ = 0;
    int64_t                      dropped_  = 0;
};

}  // namespace rtp_llm
