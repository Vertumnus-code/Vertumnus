#pragma once

#include <atomic>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {

enum preRunMode {
    prefill_warm_up     = 0,
    decode_warm_up      = 1,
    build_system_prompt = 2
};

inline std::string preRunModeToString(preRunMode mode) {
    switch (mode) {
        case prefill_warm_up:
            return "prefill_warm_up";
        case decode_warm_up:
            return "decode_warm_up";
        case build_system_prompt:
            return "build_system_prompt";
        default:
            return "unknown pre run mode";
    }
}

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase();

    void initRuntime(const EngineInitParams& params);

    void pause() {
        pause_ = true;
    }

    virtual absl::Status restart() {
        pause_ = false;
        return absl::OkStatus();
    }

    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;

    virtual void enqueue(std::shared_ptr<GenerateStream>& stream) = 0;

    virtual std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs);

    virtual std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input);

    virtual absl::Status stop() = 0;

    virtual absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                     preRunMode                            mode) = 0;

    virtual KVCacheInfo getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) = 0;

    virtual const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    virtual SchedulerBase& getScheduler() {
        return *scheduler_;
    }

    virtual int64_t getLastScheduleTime() {
        return autil::TimeUtility::currentTimeInMilliSeconds();
    }

    virtual bool isMTPEagle() {
        return false;
    }

    virtual bool isEagle() {
        return false;
    }

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }
    virtual absl::Status
    switchParallelTopology(int64_t tp_size, int64_t dp_size, int64_t timeout_ms, const std::string& switch_id = "") {
        return absl::UnimplementedError("switchParallelTopology is not implemented for this engine");
    }
    virtual absl::Status prepareParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") {
        return absl::UnimplementedError("prepareParallelTopologySwitch is not implemented for this engine");
    }
    virtual absl::Status armParallelTopologySwitch(int64_t pause_at_ms, const std::string& switch_id = "") {
        return absl::UnimplementedError("armParallelTopologySwitch is not implemented for this engine");
    }
    virtual void resumeParallelTopologySwitch() {
        (void)restart();
    }
    virtual absl::Status
    preparePrefixTopologySwitch(int64_t tp_size, int64_t dp_size, int64_t timeout_ms, const std::string& switch_id) {
        return absl::OkStatus();
    }
    virtual absl::Status prepareResidentPrefixTopologySwitch(int64_t            tp_size,
                                                             int64_t            dp_size,
                                                             const std::string& switch_id,
                                                             bool               leader_merge = false) {
        return absl::UnimplementedError("resident prefix topology switch is not implemented for this engine");
    }
    virtual absl::Status drainParallelTopologySwitch(int64_t            timeout_ms,
                                                     const std::string& switch_id,
                                                     const std::string& expected_epoch = "") {
        return absl::UnimplementedError("distributed topology drain is not implemented for this engine");
    }
    virtual absl::Status releaseParallelTopologySwitch(const std::string& switch_id) {
        return absl::UnimplementedError("distributed topology release is not implemented for this engine");
    }
    virtual absl::Status finishParallelTopologySwitch(bool cancel, const std::string& switch_id) {
        resumeParallelTopologySwitch();
        return absl::OkStatus();
    }
    virtual absl::Status quarantineParallelTopologySwitch(const std::string& switch_id) {
        pause();
        return absl::OkStatus();
    }
    virtual absl::Status settleParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") {
        return absl::OkStatus();
    }
    virtual absl::Status flushParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id = "") {
        return absl::OkStatus();
    }
    virtual void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {}

    std::shared_ptr<KVCacheManager> getCacheManager() const;
    void                            wakeupForPrefixCacheInstall() {
        if (scheduler_) {
            scheduler_->wakeupForTopologySwitch();
        }
    }

protected:
    ResourceContext                resource_context_;
    MlaOpsType                     mla_ops_type_       = MlaOpsType::AUTO;
    int32_t                        kv_cache_group_num_ = 1;
    std::vector<int32_t>           kv_cache_layer_to_group_;
    std::unique_ptr<SchedulerBase> scheduler_ = nullptr;
    std::atomic<bool>              pause_     = false;
};

}  // namespace rtp_llm
