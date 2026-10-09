#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams> propose_params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream>   makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream>   enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::vector<GenerateStreamPtr>    batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void                              enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;
    absl::Status                      restart() override;

    KVCacheInfo  getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status step();
    absl::Status startLoop();
    int64_t      getLastScheduleTime() override;
    void         reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool         updateEplbConfig(const EPLBConfig& config) override;
    absl::Status switchParallelTopology(int64_t            tp_size,
                                        int64_t            dp_size,
                                        int64_t            timeout_ms,
                                        const std::string& switch_id = "") override;
    absl::Status prepareParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") override;
    absl::Status armParallelTopologySwitch(int64_t pause_at_ms, const std::string& switch_id = "") override;
    void         resumeParallelTopologySwitch() override;
    absl::Status preparePrefixTopologySwitch(int64_t            tp_size,
                                             int64_t            dp_size,
                                             int64_t            timeout_ms,
                                             const std::string& switch_id) override;
    absl::Status finishParallelTopologySwitch(bool cancel, const std::string& switch_id) override;
    absl::Status prepareResidentPrefixTopologySwitch(int64_t            tp_size,
                                                     int64_t            dp_size,
                                                     const std::string& switch_id,
                                                     bool               leader_merge = false) override;
    absl::Status drainParallelTopologySwitch(int64_t            timeout_ms,
                                             const std::string& switch_id,
                                             const std::string& expected_epoch = "") override;
    absl::Status releaseParallelTopologySwitch(const std::string& switch_id) override;
    absl::Status quarantineParallelTopologySwitch(const std::string& switch_id) override;
    absl::Status settleParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") override;
    absl::Status flushParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") override;
    void         startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) override;

private:
    void                            touchPrefixAtAdmission(const std::shared_ptr<GenerateStream>& stream);
    void                            initScheduler();
    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens);
    WarmUpResult                    warmUp(const EngineInitParams& params);
    WarmUpResult                    prefillWarmUp(const EngineInitParams& params);
    WarmUpResult                    decodeWarmUp(const EngineInitParams& params);
    void                            initLoadBalance();
    absl::Status                    trySaveStepError() const;
    void                            loop();
    void initCacheManager(std::optional<WarmUpResult> warm_up_result, const CacheStoreConfig& cache_store_config);
    absl::Status                   initSystemPrompt();
    std::shared_ptr<GenerateInput> makeFakeInput(size_t seq_len);
    void                           mayAddFakeStream(std::list<GenerateStreamPtr>& streams);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;

private:
    autil::ThreadPtr                              loop_thread_;
    std::atomic<bool>                             running_{false};
    std::unique_ptr<Executor>                     executor_;
    ModelConfig                                   model_config_;
    ParallelismConfig                             parallelism_config;
    RuntimeConfig                                 runtime_config;
    EPLBConfig                                    eplb_config;
    PDSepConfig                                   pd_sep_config;
    ProfilingDebugLoggingConfig                   profiling_debug_logging_config;
    KVCacheConfig                                 kv_cache_config;
    FfnDisAggregateConfig                         ffn_disaggregate_config;
    ModelSpecificConfig                           model_specific_config;
    SpeculativeExecutionConfig                    sp_config;
    kmonitor::MetricsReporterPtr                  metrics_reporter_;
    std::unique_ptr<ProposeModelEngineInitParams> propose_params_;
    StepWindowProfiler                            step_profiler_;
    int                                           reserve_step_             = 0;
    bool                                          last_prefix_control_step_ = false;
    std::mutex                                    topology_switch_mutex_;
    std::mutex                                    execution_mutex_;
    enum class PrefixTopologyPhase {
        NONE,
        PREPARED,
        COMMITTED,
        QUARANTINED
    };
    std::atomic<PrefixTopologyPhase> prefix_topology_phase_{PrefixTopologyPhase::NONE};
    std::string                      topology_switch_id_;
    std::string                      last_topology_switch_id_;
    static constexpr size_t          MAX_TOPOLOGY_SWITCH_IDENTITIES = 65536;
    std::unordered_set<std::string>  topology_switch_identities_;
    absl::Status                     claimTopologySwitchLocked(const std::string& switch_id);
    uint64_t                         prefix_topology_generation_ = 0;
    int64_t                          prefix_topology_target_tp_  = 0;
    int64_t                          prefix_topology_target_dp_  = 0;
    void                             quarantinePrefixTopologyLocked();
    std::atomic<bool>                topology_switch_armed_{false};
    std::atomic<int64_t>             topology_switch_pause_at_ms_{0};
    std::atomic<int64_t>             topology_switch_step_count_{0};
    std::atomic<int64_t>             topology_switch_resume_step_count_{-1};
};

}  // namespace rtp_llm
