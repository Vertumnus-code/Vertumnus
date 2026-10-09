#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace rtp_llm {

struct EngineScheduleInfo {
    enum class TaskPhase {
        PENDING,
        RECEIVED,
        KV_ALLOCATED,
        RUNNING,
    };

    struct TaskInfo {
        int64_t   request_id;
        int64_t   prefix_length;
        int64_t   input_length;
        int64_t   waiting_time_ms;
        int64_t   iterate_count     = 0;
        int64_t   end_time_ms       = -1;
        bool      is_waiting        = true;
        int64_t   batch_id          = 0;
        TaskPhase phase             = TaskPhase::PENDING;
        int64_t   execution_time_ms = 0;
    };

    void mergeRunningSnapshot(const std::vector<TaskInfo>& running_snapshot, bool refresh_prefix) {
        for (auto& task : running_task_info_list) {
            for (const auto& running : running_snapshot) {
                if (task.request_id != running.request_id) {
                    continue;
                }
                task.is_waiting = false;
                task.phase      = TaskPhase::RUNNING;
                // The enqueue metadata can predate cache lookup; use the scheduler's actual prefix.
                if (refresh_prefix && running.input_length == task.input_length && running.input_length > 0
                    && running.prefix_length >= 0 && running.prefix_length <= running.input_length) {
                    task.prefix_length = running.prefix_length;
                }
            }
        }
    }

    std::vector<TaskInfo> running_task_info_list;
    std::vector<TaskInfo> finished_task_info_list;
    int64_t               last_schedule_delta;
    int64_t               latest_finished_version = 0;
};

}  // namespace rtp_llm
