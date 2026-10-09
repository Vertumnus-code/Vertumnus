#pragma once
#include <algorithm>
#include <atomic>
#include <list>
#include <shared_mutex>
#include <unordered_map>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {
class RpcServerRuntimeMeta {
public:
    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finished_version) {
        std::shared_lock<std::shared_mutex> lock(read_write_lock_);
        EngineScheduleInfo                  info;
        for (auto& iter : running_streams_) {
            auto task_info              = iter.second.task_info;
            task_info.execution_time_ms = std::max<int64_t>(
                0, (autil::TimeUtility::currentTimeInMicroSeconds() - iter.second.stream->beginTimeUs()) / 1000);
            task_info.phase      = schedulePhase(iter.second.stream->admissionPhase());
            task_info.is_waiting = task_info.phase != EngineScheduleInfo::TaskPhase::RUNNING;
            info.running_task_info_list.push_back(task_info);
        }
        // When no new finished tasks, keep client's version (monotonic). Otherwise client/flexlb
        // would overwrite latestFinishedTaskVersion with 0 when finished queue is empty.
        int64_t version = latest_finished_version;
        for (auto& iter : finished_streams_) {
            if (iter.first > latest_finished_version) {
                info.finished_task_info_list.push_back(iter.second);
                if (iter.first > version) {
                    version = iter.first;
                }
            }
        }
        info.latest_finished_version = version;
        return info;
    }

    void enqueue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        cancelled_operation_epochs_.erase(request_id);
        EngineScheduleInfo::TaskInfo task_info(
            {request_id, stream->prefixLength(), stream->inputLength(), stream->getTimeInfo().wait_time_us});
        task_info.phase              = EngineScheduleInfo::TaskPhase::RECEIVED;
        running_streams_[request_id] = RunningTask{task_info, stream};
    }

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            return;
        }
        auto& task_info = ptr->second.task_info;
        if (finished_streams_.size() >= finished_capacity_) {
            cancelled_operation_epochs_.erase(finished_streams_.front().second.request_id);
            finished_streams_.pop_front();
        }
        int64_t current           = autil::TimeUtility::currentTimeInMilliSeconds();
        task_info.end_time_ms     = current;
        task_info.prefix_length   = stream->prefixLength();
        task_info.input_length    = stream->inputLength();
        task_info.waiting_time_ms = stream->getTimeInfo().wait_time_us / 1000;
        task_info.iterate_count   = stream->iterCount();

        int64_t version = version_.fetch_add(1, std::memory_order_relaxed);
        finished_streams_.push_back(std::make_pair(version, task_info));
        running_streams_.erase(ptr);
    }

    struct CancelResult {
        enum class Outcome {
            CANCELLED,
            NOT_FOUND,
            NOT_RECEIVED,
            ALREADY_CANCELLED,
        };
        Outcome                       outcome;
        EngineScheduleInfo::TaskPhase observed_phase;
        bool                          idempotent_replay;
    };

    CancelResult cancelIfReceived(int64_t request_id, int64_t operation_epoch) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                cancelled = cancelled_operation_epochs_.find(request_id);
        if (cancelled != cancelled_operation_epochs_.end() && cancelled->second == operation_epoch) {
            return {CancelResult::Outcome::ALREADY_CANCELLED, EngineScheduleInfo::TaskPhase::RECEIVED, true};
        }
        auto iter = running_streams_.find(request_id);
        if (iter == running_streams_.end()) {
            return {CancelResult::Outcome::NOT_FOUND, EngineScheduleInfo::TaskPhase::PENDING, false};
        }
        auto phase = schedulePhase(iter->second.stream->admissionPhase());
        if (phase != EngineScheduleInfo::TaskPhase::RECEIVED
            || !iter->second.stream->cancelIfReceived("reroute received request for CP composition switch")) {
            phase = schedulePhase(iter->second.stream->admissionPhase());
            return {CancelResult::Outcome::NOT_RECEIVED, phase, false};
        }
        cancelled_operation_epochs_[request_id] = operation_epoch;
        return {CancelResult::Outcome::CANCELLED, EngineScheduleInfo::TaskPhase::RECEIVED, false};
    }

protected:
    struct RunningTask {
        EngineScheduleInfo::TaskInfo task_info;
        GenerateStreamPtr            stream;
    };

    static EngineScheduleInfo::TaskPhase schedulePhase(GenerateStream::AdmissionPhase phase) {
        switch (phase) {
            case GenerateStream::AdmissionPhase::RECEIVED:
                return EngineScheduleInfo::TaskPhase::RECEIVED;
            case GenerateStream::AdmissionPhase::KV_ALLOCATED:
                return EngineScheduleInfo::TaskPhase::KV_ALLOCATED;
            case GenerateStream::AdmissionPhase::RUNNING:
            case GenerateStream::AdmissionPhase::FINISHED:
                return EngineScheduleInfo::TaskPhase::RUNNING;
        }
        return EngineScheduleInfo::TaskPhase::PENDING;
    }

    // Note: finished_streams_ pairs are (monotonic_version, TaskInfo).
    // The list is ordered by insertion time (append-only via push_back), so end_time_ms
    // is approximately monotonic. The break below relies on this approximate time ordering
    // to stop early — it does NOT use pair.first (version) for trimming decisions.
    void trimFinishedStreams() {
        auto current = autil::TimeUtility::currentTimeInMilliSeconds();
        auto iter    = finished_streams_.begin();
        while (iter != finished_streams_.end()) {
            int64_t end_time_ms = iter->second.end_time_ms;
            if (end_time_ms > current) {
                RTP_LLM_LOG_WARNING("find task: %ld end time: %ld bigger than current time: %ld",
                                    iter->second.request_id,
                                    end_time_ms,
                                    current);
                iter = finished_streams_.erase(iter);
            } else if (end_time_ms + timeout_ms_ <= current) {
                cancelled_operation_epochs_.erase(iter->second.request_id);
                iter = finished_streams_.erase(iter);
            } else {
                break;
            }
        }
    }
    std::unordered_map<int64_t, RunningTask>                    running_streams_;
    std::list<std::pair<int64_t, EngineScheduleInfo::TaskInfo>> finished_streams_;
    std::unordered_map<int64_t, int64_t>                        cancelled_operation_epochs_;
    std::atomic<int64_t>                                        version_{0};
    mutable std::shared_mutex                                   read_write_lock_;
    int64_t                                                     timeout_ms_        = 5000;
    int64_t                                                     finished_capacity_ = 1000;
};

};  // namespace rtp_llm
