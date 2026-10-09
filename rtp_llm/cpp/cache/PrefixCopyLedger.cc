#include "rtp_llm/cpp/cache/PrefixCopyLedger.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rtp_llm {

int64_t PrefixCopyLedger::monotonicNanos() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

PrefixCopyLedger::PrefixCopyLedger(size_t terminal_limit, Now now):
    terminal_limit_(terminal_limit), now_(std::move(now)), origin_ns_(now_()) {}

int64_t PrefixCopyLedger::elapsed() const {
    const auto current = now_();
    return origin_ns_ >= 0 && current >= origin_ns_ ? current - origin_ns_ : -1;
}

void PrefixCopyLedger::timestamp(State& state, int64_t& destination) {
    const auto current = elapsed();
    if (current < 0 || current < state.last_ns) {
        state.record.tracking_valid = false;
        destination                 = -1;
    } else {
        destination   = current;
        state.last_ns = current;
    }
}

void PrefixCopyLedger::increment(uint64_t& counter) {
    if (counter == std::numeric_limits<uint64_t>::max()) {
        totals_.counters_saturated = true;
    } else {
        ++counter;
    }
}

void PrefixCopyLedger::publish(
    uint64_t generation, uint64_t operation_id, uint64_t import_epoch, uint64_t cache_key, int32_t block_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation == 0 || operation_id == 0 || import_epoch == 0 || block_index <= 0 || live_.count(generation)
        || physical_.count(block_index)) {
        throw std::logic_error("invalid copied block generation");
    }
    auto state                 = std::make_shared<State>();
    state->record.generation   = generation;
    state->record.operation_id = operation_id;
    state->record.import_epoch = import_epoch;
    state->record.cache_key    = cache_key;
    state->record.block_index  = block_index;
    timestamp(*state, state->record.published_ns);
    live_.emplace(generation, state);
    physical_.emplace(block_index, generation);
    increment(totals_.published_blocks);
}

void PrefixCopyLedger::evict(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = live_.find(generation);
    if (found != live_.end() && !found->second->record.evicted) {
        auto& state          = *found->second;
        state.record.evicted = true;
        timestamp(state, state.record.evicted_ns);
    }
}

void PrefixCopyLedger::invalidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : live_) {
        if (!entry.second->record.reused) {
            entry.second->record.tracking_valid = false;
        }
    }
}

void PrefixCopyLedger::physicalFree(int32_t block_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = physical_.find(block_index);
    if (found == physical_.end()) {
        return;
    }
    auto state = live_.at(found->second);
    physical_.erase(found);
    state->record.physically_freed = true;
    if (!state->record.evicted) {
        state->record.tracking_valid = false;
    }
    timestamp(*state, state->record.physical_free_ns);
    retire(state);
}

std::unique_ptr<PrefixCopyLedger::Use>
PrefixCopyLedger::capture(const std::vector<int32_t>& blocks, int64_t request_id, const std::string& attempt_id) {
    std::lock_guard<std::mutex>         lock(mutex_);
    std::vector<std::shared_ptr<State>> records;
    std::unordered_set<uint64_t>        seen;
    for (const auto block : blocks) {
        const auto found = physical_.find(block);
        if (found != physical_.end() && seen.insert(found->second).second) {
            auto state = live_.at(found->second);
            ++state->record.pending_executions;
            records.push_back(std::move(state));
        }
    }
    if (records.empty()) {
        return nullptr;
    }
    return std::unique_ptr<Use>(new Use(shared_from_this(), std::move(records), request_id, attempt_id));
}

PrefixCopyLedger::Use::Use(std::shared_ptr<PrefixCopyLedger>   owner,
                           std::vector<std::shared_ptr<State>> records,
                           int64_t                             request_id,
                           std::string                         attempt_id):
    owner_(std::move(owner)),
    records_(std::move(records)),
    request_id_(request_id),
    attempt_id_(std::move(attempt_id)) {}

PrefixCopyLedger::Use::~Use() {
    finish(false);
}

void PrefixCopyLedger::Use::finish(bool success) {
    if (owner_) {
        owner_->finish(records_, success, request_id_, attempt_id_);
        owner_.reset();
        records_.clear();
    }
}

void PrefixCopyLedger::finish(const std::vector<std::shared_ptr<State>>& records,
                              bool                                       success,
                              int64_t                                    request_id,
                              const std::string&                         attempt_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& state : records) {
        auto& record = state->record;
        --record.pending_executions;
        if (!success && !record.reused) {
            record.tracking_valid = false;
        }
        if (success && !record.reused) {
            record.reused           = true;
            record.first_request_id = request_id;
            record.first_attempt_id = attempt_id;
            timestamp(*state, record.first_reuse_ns);
            increment(totals_.reused_blocks);
        }
        retire(state);
    }
}

void PrefixCopyLedger::retire(const std::shared_ptr<State>& state) {
    auto& record = state->record;
    if (record.retired || !record.physically_freed || record.pending_executions != 0) {
        return;
    }
    record.retired = true;
    if (!record.reused) {
        increment(record.tracking_valid ? totals_.wasted_blocks : totals_.unknown_retired_blocks);
    }
    terminal_.push_back(record);
    live_.erase(record.generation);
    if (terminal_.size() > terminal_limit_) {
        terminal_.pop_front();
        increment(totals_.dropped_terminal_records);
    }
}

PrefixCopySnapshot PrefixCopyLedger::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        result = totals_;
    result.sampled_ns                  = elapsed();
    result.records.assign(terminal_.begin(), terminal_.end());
    for (const auto& entry : live_) {
        result.records.push_back(entry.second->record);
    }
    std::sort(result.records.begin(), result.records.end(), [](const auto& left, const auto& right) {
        return left.generation < right.generation;
    });
    return result;
}

}  // namespace rtp_llm
