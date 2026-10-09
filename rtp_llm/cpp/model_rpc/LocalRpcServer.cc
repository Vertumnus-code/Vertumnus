#include <memory>
#include <chrono>
#include <limits>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefixTopologyPlan.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"
#include "rtp_llm/cpp/cache/PrefixRankInstallQueue.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferDispatcher.h"
#include "autil/legacy/jsonizable.h"
#include "autil/legacy/exception.h"
#include "autil/StringUtil.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <cmath>

using namespace std;

namespace rtp_llm {

namespace {

struct ParallelTopologySwitchRequest: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("action", action, std::string(""));
        json.Jsonize("tp_size", tp_size, int64_t(0));
        json.Jsonize("dp_size", dp_size, int64_t(0));
        json.Jsonize("timeout_ms", timeout_ms, int64_t(300000));
        json.Jsonize("internal", internal, false);
        json.Jsonize("resident_cache", resident_cache, false);
        json.Jsonize("resident_leader_merge", resident_leader_merge, false);
        json.Jsonize("phase", phase, std::string("commit"));
        json.Jsonize("pause_at_ms", pause_at_ms, int64_t(0));
        json.Jsonize("switch_id", switch_id, std::string(""));
        json.Jsonize("expected_boot_id", expected_boot_id, std::string(""));
        json.Jsonize("expected_epoch", expected_epoch, std::string(""));
        json.Jsonize("expected_world_rank", expected_world_rank, int64_t(-1));
        json.Jsonize("prefix_controller", prefix_controller, PrefixTopologyController{});
        json.Jsonize("expected_participants", expected_participants, std::vector<PrefixTopologyParticipant>{});
    }

    std::string                            action;
    int64_t                                tp_size               = 0;
    int64_t                                dp_size               = 0;
    int64_t                                timeout_ms            = 300000;
    bool                                   internal              = false;
    bool                                   resident_cache        = false;
    bool                                   resident_leader_merge = false;
    std::string                            phase                 = "commit";
    int64_t                                pause_at_ms           = 0;
    std::string                            switch_id;
    std::string                            expected_boot_id;
    std::string                            expected_epoch;
    int64_t                                expected_world_rank = -1;
    PrefixTopologyController               prefix_controller;
    std::vector<PrefixTopologyParticipant> expected_participants;

    std::string controllerBinding() const {
        auto fields = std::vector<std::string>{switch_id,
                                               std::to_string(tp_size),
                                               std::to_string(dp_size),
                                               expected_boot_id,
                                               expected_epoch,
                                               std::to_string(expected_world_rank)};
        if (resident_cache) {
            fields.push_back("resident_prefix_split_v1");
        }
        if (resident_leader_merge) {
            fields.push_back("resident_leader_merge_v1");
        }
        return autil::legacy::ToJsonString(fields);
    }
};

ParallelismConfig makeSwitchedRpcParallelismConfig(ParallelismConfig config, int64_t tp_size, int64_t dp_size) {
    RTP_LLM_CHECK_WITH_INFO(tp_size > 0 && dp_size > 0, "tp_size and dp_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(config.world_size > 0 && config.world_size % tp_size == 0
                                && dp_size == config.world_size / tp_size,
                            "tp_size * dp_size must equal world_size, got tp=%ld dp=%ld world=%ld",
                            tp_size,
                            dp_size,
                            config.world_size);
    RTP_LLM_CHECK_WITH_INFO(config.prefill_cp_config.is_enabled() || tp_size == config.tp_size,
                            "hot topology switch cannot change weight-sharded TP without prefill CP enabled; "
                            "got current_tp=%ld target_tp=%ld",
                            config.tp_size,
                            tp_size);
    RTP_LLM_CHECK_WITH_INFO(config.ffn_sp_size > 0, "ffn_sp_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(tp_size % config.ffn_sp_size == 0,
                            "tp_size must be divisible by ffn_sp_size, got tp=%ld ffn_sp=%ld",
                            tp_size,
                            config.ffn_sp_size);
    config.tp_size = tp_size;
    config.dp_size = dp_size;
    if (config.ep_size != 1) {
        config.ep_size = tp_size * dp_size;
    }
    config.tp_rank     = config.world_rank % config.tp_size;
    config.dp_rank     = config.world_rank / config.tp_size;
    config.ep_rank     = config.world_rank % config.ep_size;
    config.ffn_tp_size = config.tp_size / config.ffn_sp_size;
    config.ffn_tp_rank = config.tp_rank % config.ffn_tp_size;
    config.enable_sp   = config.ffn_sp_size > 1;
    return config;
}

std::string makeInternalSwitchSchedulerInfo(const ParallelTopologySwitchRequest& req,
                                            const std::string&                   phase = "commit") {
    auto internal_request     = req;
    internal_request.internal = true;
    internal_request.phase    = phase;
    return autil::legacy::ToJsonString(internal_request);
}

int64_t getTopologySwitchPauseDelayMs() {
    constexpr int64_t kDefaultPauseDelayMs = 200;
    constexpr int64_t kMaxPauseDelayMs     = 5000;
    const char*       value                = std::getenv("RTP_LLM_SWITCH_PAUSE_DELAY_MS");
    if (value == nullptr || value[0] == '\0') {
        return kDefaultPauseDelayMs;
    }
    char*   end    = nullptr;
    int64_t parsed = std::strtoll(value, &end, 10);
    if (end == value || parsed < 0) {
        RTP_LLM_LOG_WARNING("invalid RTP_LLM_SWITCH_PAUSE_DELAY_MS=%s, fallback to %ld", value, kDefaultPauseDelayMs);
        return kDefaultPauseDelayMs;
    }
    return std::min(parsed, kMaxPauseDelayMs);
}

bool useSignalTopologySwitchPause() {
    const char* mode = std::getenv("RTP_LLM_SWITCH_PAUSE_MODE");
    if (mode == nullptr || mode[0] == '\0') {
        return true;
    }
    return std::string(mode) != "fixed";
}

std::vector<std::string> makeLocalWorldGrpcAddrs(const EngineInitParams& params) {
    std::vector<std::string> addrs;
    const auto&              pc = params.parallelism_config;
    if (pc.local_world_size != pc.world_size) {
        return addrs;
    }
    try {
        pybind11::gil_scoped_acquire gil;
        const int64_t                start_port = params.server_config.attr("start_port").cast<int64_t>();
        const int64_t worker_info_port_num      = params.server_config.attr("worker_info_port_num").cast<int64_t>();
        addrs.reserve(pc.world_size);
        for (int64_t rank = 0; rank < pc.world_size; ++rank) {
            addrs.push_back(
                autil::StringUtil::formatString("127.0.0.1:%ld", start_port + rank * worker_info_port_num + 1));
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("failed to build local world grpc addrs: %s", e.what());
        addrs.clear();
    }
    return addrs;
}

std::shared_ptr<BroadcastManager> makeTopologySwitchBroadcaster(const EngineInitParams& params) {
    const auto& pc = params.parallelism_config;
    if (pc.world_rank != 0) {
        return nullptr;
    }
    auto worker_grpc_addrs = params.runtime_config.worker_grpc_addrs;
    auto local_world_addrs = makeLocalWorldGrpcAddrs(params);
    if (!local_world_addrs.empty()) {
        RTP_LLM_LOG_INFO("use local world grpc addrs for topology switch broadcaster, size=%zu",
                         local_world_addrs.size());
        worker_grpc_addrs = std::move(local_world_addrs);
    }
    auto broadcaster = std::make_shared<BroadcastManager>(worker_grpc_addrs);
    if (!broadcaster->init()) {
        RTP_LLM_LOG_WARNING("failed to init topology switch broadcaster");
        return nullptr;
    }
    return broadcaster;
}

grpc::Status abslToGrpcStatus(const absl::Status& status) {
    if (status.ok()) {
        return grpc::Status::OK;
    }
    if (status.code() == absl::StatusCode::kInvalidArgument) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, std::string(status.message()));
    }
    if (status.code() == absl::StatusCode::kDeadlineExceeded) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, std::string(status.message()));
    }
    if (status.code() == absl::StatusCode::kFailedPrecondition) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, std::string(status.message()));
    }
    if (status.code() == absl::StatusCode::kUnavailable) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, std::string(status.message()));
    }
    if (status.code() == absl::StatusCode::kCancelled) {
        return grpc::Status(grpc::StatusCode::CANCELLED, std::string(status.message()));
    }
    if (status.code() == absl::StatusCode::kResourceExhausted) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, std::string(status.message()));
    }
    return grpc::Status(grpc::StatusCode::INTERNAL, std::string(status.message()));
}

}  // namespace

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  py::object                                    mm_process_engine,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_ = maga_init_params;
    weight_manager_   = maga_init_params.weight_manager;
    metrics_reporter_ = maga_init_params.metrics_reporter;
    RTP_LLM_LOG_INFO("LocalRpcServer aux_string %s", maga_init_params_.misc_config.aux_string.c_str());
    propose_maga_init_params_ = propose_params.get();
    profile_broadcaster_      = makeTopologySwitchBroadcaster(maga_init_params_);

    {
        pybind11::gil_scoped_release release;
        RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                "running engine init with gil held may cause program hang, please check");
        engine_.reset(new NormalEngine(maga_init_params, std::move(propose_params)));
    }
    if (!mm_process_engine.is_none()) {
        auto vit_separation = maga_init_params.vit_config.vit_separation;
        if (vit_separation == VitSeparation::VIT_SEPARATION_REMOTE) {
            mm_processor_.reset(new RemoteMultimodalProcessor(mm_process_engine,
                                                              maga_init_params.model_config_.mm_model_config,
                                                              maga_init_params.model_config_.max_seq_len));
        } else if (vit_separation == VitSeparation::VIT_SEPARATION_LOCAL) {
            mm_processor_.reset(new LocalMultimodalProcessor(mm_process_engine,
                                                             maga_init_params.model_config_.mm_model_config,
                                                             maga_init_params.model_config_.max_seq_len));
        } else {
            return grpc::Status(grpc::StatusCode::INTERNAL, "invalid vit separation value in config");
        }
    }

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    const auto& error_msg = error_info.ToString();
    RTP_LLM_LOG_WARNING("request [%s], error code [%s], error message [%s]",
                        request_key.c_str(),
                        ErrorCodeToString(error_info.code()).c_str(),
                        error_msg.c_str());
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        RTP_LLM_LOG_WARNING("request [%s] error details serialize to string failed", request_key.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*             context,
                                              const string&                    request_key,
                                              WriterInterface*                 writer,
                                              std::shared_ptr<GenerateStream>& stream) {
    RTP_LLM_PROFILE_FUNCTION();
    // 需要检查 !hasError(): 之前 finished() 表示完成且无错，现在 FINISHED 状态可能包含错误
    // 如果流有错误，应该停止消费输出
    while (stream->isActive() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, result.status());
            } else {
                break;
            }
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;

        QueryConverter::transResponse(&outputs_pb,
                                      &(result.value()),
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        if (context->IsCancelled()) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            stream->reportError(ErrorCode::CANCELLED, "write outputs pb failed");
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
            break;
        }
        if (stream->queryPdSep()) {
            stream->waitForRemoteGenerate();
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("request [%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

ErrorInfo LocalRpcServer::prepareInput(const GenerateInputPB& input_pb, std::shared_ptr<GenerateInput>& output) {
    output = QueryConverter::transQuery(&input_pb);
    if (mm_processor_ != nullptr && output->multimodal_inputs) {
        RTP_LLM_PROFILE_SCOPE("rpc.mm_update_features");
        auto mm_res = mm_processor_->updateMultimodalFeatures(output);
        if (!mm_res.ok()) {
            return mm_res;
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo LocalRpcServer::collectStreamOutput(grpc::ServerContext*                  context,
                                              std::shared_ptr<GenerateStream>&      stream,
                                              const std::shared_ptr<GenerateInput>& input,
                                              GenerateOutputs&                      last_outputs) {
    while (!stream->isFinished() || stream->hasOutput()) {
        if (context->IsCancelled()) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by client");
            return ErrorInfo(ErrorCode::CANCELLED, "request cancelled by client");
        }
        const auto output_result = stream->nextOutput();
        if (!output_result.ok()) {
            if (output_result.status().code() != ErrorCode::FINISHED) {
                return output_result.status();
            }
            break;
        }
        last_outputs = output_result.value();
    }
    return ErrorInfo::OkStatus();
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_SCOPE("rpc.generate_stream_call");
    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_, meta_);
    std::shared_ptr<GenerateInput> input;
    {
        auto mm_res = prepareInput(*request, input);
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    {
        RTP_LLM_PROFILE_SCOPE("rpc.enqueue_engine");
        generate_context.setStream(engine_->enqueue(input));
    }

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key, writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

grpc::Status LocalRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                               const BatchGenerateInputPB* request,
                                               BatchGenerateOutputsPB*     response) {
    RTP_LLM_PROFILE_SCOPE("rpc.batch_generate_call");
    AtomicGuard request_guard(onflight_requests_);
    const int   batch_size = request->inputs_size();
    RTP_LLM_LOG_INFO("receive batch generate request, batch_size=%d", batch_size);

    if (batch_size == 0) {
        return grpc::Status::OK;
    }

    std::vector<std::shared_ptr<GenerateInput>> inputs;
    inputs.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        std::shared_ptr<GenerateInput> input;
        auto                           err = prepareInput(request->inputs(i), input);
        if (!err.ok()) {
            // Fill error results for all requests (0..batch_size-1) to maintain 1:1 mapping
            for (int j = 0; j < batch_size; j++) {
                auto* result = response->add_results();
                auto* err_pb = result->mutable_error_info();
                err_pb->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
                if (j == i) {
                    err_pb->set_error_message("multimodal processing failed: " + err.ToString());
                } else {
                    err_pb->set_error_message("batch aborted due to multimodal failure at index " + std::to_string(i));
                }
            }
            return grpc::Status::OK;
        }
        inputs.push_back(input);
    }

    // batchEnqueue contract: returned vector is 1:1 with `inputs` (same size, same order).
    // Streams that failed checkInputLength carry an error reported via reportError() and surface
    // it through collectStreamOutput → nextOutput → ErrorInfo path below.
    auto streams = engine_->batchEnqueue(inputs);

    // collectStreamOutput is currently SERIAL: streams[0] must finish before streams[1] is drained.
    // For batch decode this is bounded (all streams advance together), but TODO: parallelize for
    // mixed-length batches.
    for (int i = 0; i < (int)streams.size(); i++) {
        auto* result = response->add_results();

        GenerateOutputs last_outputs;
        auto            err = collectStreamOutput(context, streams[i], inputs[i], last_outputs);
        if (!err.ok()) {
            auto* err_pb = result->mutable_error_info();
            err_pb->set_error_code(err.code() == ErrorCode::CANCELLED ? ErrorCodePB::CANCELLED :
                                                                        ErrorCodePB::UNKNOWN_ERROR);
            err_pb->set_error_message(err.ToString());
        } else {
            auto* output_pb = result->mutable_final_output();
            QueryConverter::transResponse(output_pb,
                                          &last_outputs,
                                          inputs[i]->generate_config->aux_info,
                                          maga_init_params_.misc_config.aux_string,
                                          streams[i]->specialTokens().eos_token_id);
        }
    }

    RTP_LLM_LOG_INFO("batch generate done, batch_size=%d", batch_size);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::GetCacheStatus(grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s, request cache version: [%d]",
                      context->peer().c_str(),
                      request->latest_cache_version());
    KVCacheInfo cache_status = getCacheStatusInfo(request->latest_cache_version(), request->need_cache_keys());
    response->set_available_kv_cache(cache_status.available_kv_cache);
    response->set_total_kv_cache(cache_status.total_kv_cache);
    response->set_block_size(cache_status.block_size);
    response->set_version(cache_status.version);
    const auto cache_manager = engine_->getCacheManager();
    if (cache_manager && cache_manager->prefixInventoryEnabled()) {
        const auto identity = cache_manager->prefixTopologyIdentity();
        auto*      output   = response->mutable_topology_identity();
        output->set_boot_id(identity.boot_id);
        output->set_world_rank(identity.world_rank);
        output->set_world_size(identity.world_size);
        if (const auto* manager = cache_manager->prefixTransactionManager()) {
            if (const auto receipt = manager->topologyReceipt()) {
                auto* recorded = response->mutable_topology_receipt();
                recorded->mutable_controller()->set_namespace_(receipt->controller.namespace_id);
                recorded->mutable_controller()->set_controller_id(receipt->controller.controller_id);
                recorded->mutable_controller()->set_term(receipt->controller.term);
                recorded->set_binding(receipt->binding);
                using State = PrefixCacheTransactionManager::TopologyState;
                switch (receipt->state) {
                    case State::ACTIVE:
                        recorded->set_state(PrefixTopologyReceiptPB::ACTIVE);
                        break;
                    case State::CANCELLED:
                        recorded->set_state(PrefixTopologyReceiptPB::CANCELLED);
                        break;
                    case State::RELEASED:
                        recorded->set_state(PrefixTopologyReceiptPB::RELEASED);
                        break;
                    case State::QUARANTINED:
                        recorded->set_state(PrefixTopologyReceiptPB::QUARANTINED);
                        break;
                    case State::CONFIRMED:
                        recorded->set_state(PrefixTopologyReceiptPB::CONFIRMED);
                        break;
                }
            }
        }
    }
    const auto& inventory = cache_status.inventory;
    if (!inventory.namespace_id.empty()) {
        serializePrefixInventory(inventory, *request, response->mutable_inventory());
    }
    auto* cache_map = response->mutable_cache_keys();
    for (const auto& key : cache_status.cached_keys) {
        (*cache_map)[static_cast<int64_t>(key)] = true;
    }
    return grpc::Status::OK;
}

void LocalRpcServer::serializePrefixInventory(const PrefixInventoryReport& inventory,
                                              const CacheVersionPB&        request,
                                              CacheReportV2PB*             report) {
    const auto& ack = request.prefix_inventory_ack();
    const bool  unchanged =
        request.has_prefix_inventory_ack() && inventory.snapshot_kind == "FULL" && !inventory.namespace_id.empty()
        && !inventory.worker_id.empty() && !inventory.epoch.empty() && !inventory.layout_fingerprint.empty()
        && inventory.dp_lane >= 0 && inventory.cp_degree > 0 && inventory.block_size > 0 && inventory.version >= 0
        && ack.rank_readiness_generation() >= 0 && ack.namespace_() == inventory.namespace_id
        && ack.worker_id() == inventory.worker_id && ack.dp_lane() == inventory.dp_lane
        && ack.epoch() == inventory.epoch && ack.layout_fingerprint() == inventory.layout_fingerprint
        && ack.version() == inventory.version && ack.device_ready() == inventory.device_ready
        && ack.cp_degree() == inventory.cp_degree && ack.block_size() == inventory.block_size
        && ack.rank_readiness_generation() <= inventory.rank_readiness_generation;
    report->Clear();
    report->set_namespace_(inventory.namespace_id);
    report->set_worker_id(inventory.worker_id);
    report->set_dp_lane(inventory.dp_lane);
    report->set_epoch(inventory.epoch);
    report->set_layout_fingerprint(inventory.layout_fingerprint);
    report->set_snapshot_kind(unchanged ? "NO_CHANGE" : inventory.snapshot_kind);
    report->set_version(inventory.version);
    report->set_rank_readiness_generation(inventory.rank_readiness_generation);
    report->set_device_ready(inventory.device_ready);
    report->set_cp_degree(inventory.cp_degree);
    report->set_block_size(inventory.block_size);
    report->set_physical_free_bytes(inventory.physical_free_bytes);
    report->set_physical_block_bytes(inventory.physical_block_bytes);
    report->set_physical_capacity_bytes(inventory.physical_capacity_bytes);
    report->set_evictable_bytes(inventory.evictable_bytes);
    report->set_active_bytes(inventory.active_bytes);
    report->set_resident_bytes(inventory.resident_bytes);
    report->set_operation_reserved_bytes(inventory.operation_reserved_bytes);
    report->set_physical_memory_stats_valid(inventory.physical_memory_stats_valid);
    auto* lifetime = report->mutable_copy_lifetime();
    lifetime->set_valid(inventory.physical_memory_stats_valid);
    lifetime->set_sampled_ns(inventory.copy_lifetime.sampled_ns);
    lifetime->set_published_blocks(inventory.copy_lifetime.published_blocks);
    lifetime->set_reused_blocks(inventory.copy_lifetime.reused_blocks);
    lifetime->set_wasted_blocks(inventory.copy_lifetime.wasted_blocks);
    lifetime->set_unknown_retired_blocks(inventory.copy_lifetime.unknown_retired_blocks);
    lifetime->set_dropped_terminal_records(inventory.copy_lifetime.dropped_terminal_records);
    lifetime->set_counters_saturated(inventory.copy_lifetime.counters_saturated);
    for (const auto& record : inventory.copy_lifetime.records) {
        auto* output = lifetime->add_records();
        output->set_generation(record.generation);
        output->set_operation_id(record.operation_id);
        output->set_import_epoch(record.import_epoch);
        output->set_cache_key(record.cache_key);
        output->set_block_index(record.block_index);
        output->set_published_ns(record.published_ns);
        output->set_first_reuse_ns(record.first_reuse_ns);
        output->set_evicted_ns(record.evicted_ns);
        output->set_physical_free_ns(record.physical_free_ns);
        output->set_first_request_id(record.first_request_id);
        output->set_first_attempt_id(record.first_attempt_id);
        output->set_reused(record.reused);
        output->set_evicted(record.evicted);
        output->set_physically_freed(record.physically_freed);
        output->set_retired(record.retired);
        output->set_tracking_valid(record.tracking_valid);
        output->set_pending_executions(record.pending_executions);
    }
    if (!unchanged) {
        for (auto key : inventory.device_keys) {
            report->add_device_keys(key);
        }
    }
    for (const auto& event : inventory.reuse_events) {
        auto* output = report->add_reuse_events();
        output->set_sequence(event.sequence);
        output->set_request_id(event.request_id);
        output->set_device_reuse_tokens(event.device_reuse_tokens);
        output->set_input_tokens(event.input_tokens);
        output->set_prefill(event.prefill);
        output->set_attempt_id(event.attempt_id);
    }
    report->set_dropped_reuse_events(inventory.dropped_reuse_events);
}

PrefixReplicatedLayoutPB LocalRpcServer::makePrefixReplicatedLayout(const KVCacheManager& manager,
                                                                    const ModelConfig&    model) {
    PrefixReplicatedLayoutPB result;
    const auto&              config    = manager.cacheConfig();
    const auto&              attention = model.attn_config;
    if (!manager.supportsPrefixRankReplication() || config.use_mla || config.is_sparse || attention.use_mla
        || attention.is_sparse || model.mm_model_config.is_multimodal || model.gen_num_per_cycle > 1
        || attention.gen_num_per_cycle > 1 || model.gen_num_per_cycle < 0 || attention.gen_num_per_cycle < 0
        || config.layer_num == 0 || config.layer_num > 4096 || config.layer_all_num != config.layer_num
        || model.num_layers != config.layer_num || config.block_num < 3 || config.layer_ids.size() != 1
        || config.global_layer_ids.size() != 1 || config.global_layer_ids[0].size() != config.layer_num
        || config.layer_ids[0] != config.global_layer_ids[0] || config.layer_attn_types.size() != config.layer_num
        || config.seq_size_per_block == 0 || config.seq_size_per_block != config.kernel_seq_size_per_block
        || attention.tokens_per_block != config.seq_size_per_block || attention.head_num == 0
        || attention.kv_head_num == 0 || attention.head_num % attention.kv_head_num != 0 || attention.head_num > 65536
        || attention.kv_head_num > 65536 || attention.size_per_head > 4096 || config.seq_size_per_block > 1048576
        || !std::isfinite(model.layernorm_eps) || !std::isfinite(model.partial_rotary_factor)
        || !std::isfinite(attention.q_scaling) || !std::isfinite(attention.softmax_extra_scale)) {
        return result;
    }
    const auto* spec                  = dynamic_cast<const MHAKVCacheSpec*>(config.cache_specs[0].get());
    const auto  expected_quantization = config.dtype == TYPE_INT8     ? KvCacheDataType::INT8 :
                                        config.dtype == TYPE_FP8_E4M3 ? KvCacheDataType::FP8 :
                                                                        KvCacheDataType::BASE;
    if (!spec || spec->local_head_num_kv != attention.kv_head_num || spec->size_per_head != attention.size_per_head
        || spec->size_per_head == 0 || spec->dtype != config.dtype
        || spec->seq_size_per_block != config.seq_size_per_block
        || (config.dtype != TYPE_FP16 && config.dtype != TYPE_BF16 && config.dtype != TYPE_INT8
            && config.dtype != TYPE_FP8_E4M3)
        || config.kv_block_stride_bytes != spec->block_size_bytes()
        || config.kv_scale_stride_bytes != spec->scale_block_size_bytes()
        || config.kv_block_size_bytes != spec->block_size_bytes() * config.layer_num
        || config.kv_scale_size_bytes != spec->scale_block_size_bytes() * config.layer_num
        || config.block_size_bytes != config.kv_block_size_bytes + config.kv_scale_size_bytes) {
        return result;
    }
    if (attention.kv_cache_dtype != expected_quantization
        || (attention.kernel_tokens_per_block != 0
            && attention.kernel_tokens_per_block != config.kernel_seq_size_per_block)) {
        return result;
    }
    const auto& rope = attention.rope_config;
    if (!std::isfinite(rope.scale) || !std::isfinite(rope.factor1) || !std::isfinite(rope.factor2)
        || !std::isfinite(rope.extrapolation_factor) || !std::isfinite(rope.mscale)) {
        return result;
    }
    std::vector<size_t> expected;
    for (uint32_t layer = 0; layer < config.layer_num; ++layer) {
        if (config.global_layer_ids[0][layer] != static_cast<int>(layer)
            || config.layer_attn_types[layer] != CacheGroupType::FULL) {
            return result;
        }
        expected.push_back(spec->block_size_bytes());
        if (spec->scale_block_size_bytes()) {
            expected.push_back(spec->scale_block_size_bytes());
        }
    }
    for (int block : {1, static_cast<int>(config.block_num - 1)}) {
        const auto buffers = manager.prefixBlockBuffers(block);
        if (buffers.size() != expected.size()) {
            return result;
        }
        for (size_t index = 0; index < buffers.size(); ++index) {
            if (!buffers[index].addr || !buffers[index].is_cuda || buffers[index].size_bytes != expected[index]) {
                return result;
            }
        }
    }
    result.set_schema_version(1);
    result.set_storage_kind("REPLICATED_FULL_MHA_LAYER_MERGED_KV_V1");
    result.set_native_dtype(static_cast<int>(config.dtype));
    for (int layer : config.global_layer_ids[0]) {
        result.add_global_layer_ids(layer);
    }
    result.set_global_query_heads(attention.head_num);
    result.set_global_kv_heads(attention.kv_head_num);
    result.set_head_dimension(attention.size_per_head);
    result.set_tokens_per_block(config.seq_size_per_block);
    result.set_kernel_tokens_per_block(config.kernel_seq_size_per_block);
    result.set_kv_bytes_per_layer(spec->block_size_bytes());
    result.set_scale_bytes_per_layer(spec->scale_block_size_bytes());
    result.set_bytes_per_rank_block(config.block_size_bytes);
    for (size_t bytes : expected) {
        result.add_buffer_bytes(bytes);
    }
    auto* output = result.mutable_rope();
    output->set_style(static_cast<int>(rope.style));
    output->set_dim(rope.dim);
    output->set_base(rope.base);
    output->set_scale(rope.scale);
    output->set_factor1(rope.factor1);
    output->set_factor2(rope.factor2);
    output->set_max_pos(rope.max_pos);
    output->set_extrapolation_factor(rope.extrapolation_factor);
    output->set_mscale(rope.mscale);
    output->set_offset(rope.offset);
    output->set_index_factor(rope.index_factor);
    output->set_mrope_dim1(rope.mrope_dim1);
    output->set_mrope_dim2(rope.mrope_dim2);
    output->set_mrope_dim3(rope.mrope_dim3);
    output->set_is_neox_style(rope.is_neox_style);
    output->set_indexer_is_neox_style(rope.indexer_is_neox_style);
    result.set_qk_norm(model.qk_norm);
    result.set_layernorm_epsilon(model.layernorm_eps);
    result.set_position_ids_style(model.position_ids_style);
    result.set_partial_rotary_factor(model.partial_rotary_factor);
    result.set_q_scaling(attention.q_scaling);
    result.set_softmax_extra_scale(attention.softmax_extra_scale);
    result.set_use_logn_attention(attention.use_logn_attn);
    result.set_causal(attention.is_causal);
    result.set_need_rope_kv_cache(attention.need_rope_kv_cache);
    result.set_fuse_qkv_add_bias(attention.fuse_qkv_add_bias);
    result.set_attention_dtype(static_cast<int>(attention.dtype));
    result.set_kv_quantization(static_cast<int>(attention.kv_cache_dtype));
    result.set_max_sequence_length(model.max_seq_len);
    return result;
}

grpc::Status LocalRpcServer::GetPrefixCacheCapabilities(grpc::ServerContext*                    context,
                                                        const PrefixCacheCapabilitiesRequestPB* request,
                                                        PrefixCacheCapabilitiesResponsePB*      response) {
    (void)context;
    (void)request;
    auto        cache_status = getCacheStatusInfo(-1, true);
    const auto& inventory    = cache_status.inventory;
    response->set_namespace_(inventory.namespace_id);
    response->set_layout_fingerprint(inventory.layout_fingerprint);
    response->set_cp_degree(inventory.cp_degree);
    response->set_dp_lane(inventory.dp_lane);
    response->set_tp_size(maga_init_params_.parallelism_config.tp_size);
    response->set_block_size(inventory.block_size);
    response->set_device_ready(inventory.device_ready);
    auto layout = makePrefixReplicatedLayout(*engine_->getCacheManager(), maga_init_params_.model_config_);
    if (layout.schema_version() == 1) {
        *response->mutable_replicated_layout() = std::move(layout);
        response->add_supported_operations("replicated_mha_layout_v1");
    }
    response->add_supported_operations("inspect");
    response->add_supported_operations("controller_fencing_v1");
    response->add_supported_operations("topology_plan_binding_v1");
    response->add_supported_operations("topology_receipt_v1");
    response->add_supported_operations("transfer_identity_v2");
    response->add_supported_operations("missing_only_v1");
    if (engine_->getCacheManager()->supportsPrefixRankReplication()) {
        response->add_supported_operations("leader_rank_broadcast_v1");
        if (maga_init_params_.parallelism_config.prefill_cp_config.is_enabled()) {
            response->add_supported_operations("resident_prefix_split_v1");
            response->add_supported_operations("resident_leader_merge_v1");
        }
    }
    auto       coordinator = engine_->getCacheManager()->connectorCoordinator();
    const auto backend     = coordinator ? coordinator->prefixTransferBackend() : transfer::TransferBackendPair{};
    if (backend.sender && backend.receiver) {
        response->add_supported_operations("prefix_transfer_endpoint_v1");
    }
    response->add_supported_operations("prepare_export");
    response->add_supported_operations("prepare_import");
    response->add_supported_operations("commit_import");
    response->add_supported_operations("abort");
    response->add_supported_operations("reclaim");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::InspectPrefix(grpc::ServerContext*          context,
                                           const InspectPrefixRequestPB* request,
                                           InspectPrefixResponsePB*      response) {
    (void)context;
    auto        cache_status = getCacheStatusInfo(-1, true);
    const auto& inventory    = cache_status.inventory;
    if ((!request->namespace_().empty() && request->namespace_() != inventory.namespace_id)
        || (!request->epoch().empty() && request->epoch() != inventory.epoch)
        || (request->version() >= 0 && request->version() != inventory.version)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix cache snapshot mismatch");
    }
    response->set_epoch(inventory.epoch);
    response->set_version(inventory.version);
    response->set_physical_block_bytes(inventory.physical_block_bytes);
    std::unordered_set<int64_t> present(inventory.device_keys.begin(), inventory.device_keys.end());
    for (const auto key : request->cache_keys()) {
        if (present.find(key) == present.end()) {
            response->add_missing_keys(key);
        } else {
            response->add_present_keys(key);
        }
    }
    response->set_complete(response->missing_keys_size() == 0 && inventory.device_ready);
    response->set_destination_physical_bytes(response->missing_keys_size() * inventory.physical_block_bytes);
    return grpc::Status::OK;
}

namespace {

uint64_t prefixEpochId(const std::string& epoch) {
    return static_cast<uint64_t>(std::hash<std::string>{}(epoch));
}

std::vector<CacheKeyType> prefixKeys(const google::protobuf::RepeatedField<int64_t>& keys) {
    return std::vector<CacheKeyType>(keys.begin(), keys.end());
}

PrefixCacheTransactionManager::ControllerIdentity prefixControllerIdentity(const PrefixControllerPB& controller) {
    return {controller.namespace_(), controller.controller_id(), controller.term()};
}

PrefixTransferIdentity prefixTransferIdentity(const PrefixTransferPB& transfer) {
    return {transfer.transfer_id(),
            transfer.source_worker_id(),
            transfer.destination_worker_id(),
            transfer.source_epoch(),
            transfer.destination_epoch(),
            transfer.source_layout(),
            transfer.destination_layout()};
}

void prefixTransferResponse(const PrefixTransferIdentity& transfer, PrefixTransferPB* response) {
    response->set_transfer_id(transfer.transfer_id);
    response->set_source_worker_id(transfer.source_worker_id);
    response->set_destination_worker_id(transfer.destination_worker_id);
    response->set_source_epoch(transfer.source_epoch);
    response->set_destination_epoch(transfer.destination_epoch);
    response->set_source_layout(transfer.source_layout);
    response->set_destination_layout(transfer.destination_layout);
}

void prefixControllerResponse(const PrefixCacheTransactionManager::ControllerIdentity& controller,
                              PrefixControllerPB*                                      response) {
    response->set_namespace_(controller.namespace_id);
    response->set_controller_id(controller.controller_id);
    response->set_term(controller.term);
}

}  // namespace

grpc::Status LocalRpcServer::validatePrefixController(const PrefixControllerPB&            controller,
                                                      const std::string&                   epoch,
                                                      const PrefixInventoryReport&         inventory,
                                                      const PrefixCacheTransactionManager* manager) {
    const auto identity = prefixControllerIdentity(controller);
    if (!identity.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    if (epoch.empty() || epoch != inventory.epoch || identity.namespace_id != inventory.namespace_id
        || (manager && !manager->isControllerCurrent(identity))) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix controller or worker identity mismatch");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::validatePrefixTransfer(const PrefixTransferPB&      transfer,
                                                    const PrefixInventoryReport& inventory,
                                                    bool                         source) {
    const auto identity = prefixTransferIdentity(transfer);
    if (!identity.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "complete prefix transfer identity is required");
    }
    if ((source ? identity.source_worker_id : identity.destination_worker_id) != inventory.worker_id
        || (source ? identity.source_epoch : identity.destination_epoch) != inventory.epoch
        || (source ? identity.source_layout : identity.destination_layout) != inventory.layout_fingerprint) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix transfer worker, epoch or layout mismatch");
    }
    return grpc::Status::OK;
}

std::string LocalRpcServer::prefixTransferKey(const PrefixControllerPB& controller,
                                              const PrefixTransferPB&   transfer,
                                              uint64_t                  lease_id,
                                              uint64_t                  operation_id,
                                              CacheKeyType              key) {
    return prefixTransferIdentity(transfer).wirePrefix(prefixControllerIdentity(controller), lease_id, operation_id)
           + ":" + std::to_string(static_cast<uint64_t>(key));
}

grpc::Status LocalRpcServer::ActivatePrefixController(grpc::ServerContext*                     context,
                                                      const ActivatePrefixControllerRequestPB* request,
                                                      ActivatePrefixControllerResponsePB*      response) {
    (void)context;
    if (!prefixControllerIdentity(request->controller()).valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    const auto inventory = getCacheStatusInfo(-1, false).inventory;
    auto       status    = validatePrefixController(request->controller(), request->epoch(), inventory, nullptr);
    if (!status.ok()) {
        return status;
    }
    auto manager = engine_->getCacheManager()->prefixTransactionManager();
    if (!manager->activateController(prefixControllerIdentity(request->controller()))) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix controller activation rejected");
    }
    response->Clear();
    response->set_success(true);
    *response->mutable_controller() = request->controller();
    response->set_epoch(inventory.epoch);
    return grpc::Status::OK;
}

void LocalRpcServer::makePrefixExportResponse(const PrefixCacheTransactionManager::ExportLease& lease,
                                              const std::string&                                epoch,
                                              PreparePrefixExportResponsePB*                    response) {
    response->Clear();
    const auto remaining = std::max<int64_t>(0,
                                             std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 lease.expires_at - PrefixCacheTransactionManager::Clock::now())
                                                 .count());
    response->set_lease_id(lease.id);
    response->set_supports_missing_only(true);
    prefixControllerResponse(lease.controller, response->mutable_controller());
    prefixTransferResponse(lease.prepare.transfer, response->mutable_transfer());
    response->set_epoch(epoch);
    response->set_version(static_cast<int64_t>(lease.version));
    response->set_request_id(lease.prepare.request_id);
    response->set_deadline_unix_ms(lease.prepare.deadline_unix_ms);
    response->set_lease_active(lease.active && remaining > 0);
    response->set_lease_remaining_ms(lease.active ? remaining : 0);
    for (const auto key : lease.keys) {
        response->add_cache_keys(static_cast<int64_t>(key));
    }
}

void LocalRpcServer::makePrefixImportResponse(const PrefixCacheTransactionManager::ImportOperation& operation,
                                              const std::string&                                    epoch,
                                              PreparePrefixImportResponsePB*                        response) {
    response->Clear();
    const bool active = operation.state != PrefixCacheTransactionManager::State::COMMITTED
                        && operation.state != PrefixCacheTransactionManager::State::ABORTED;
    response->set_operation_id(operation.id);
    prefixControllerResponse(operation.controller, response->mutable_controller());
    prefixTransferResponse(operation.prepare.transfer, response->mutable_transfer());
    response->set_epoch(epoch);
    response->set_expected_version(static_cast<int64_t>(operation.expected_version));
    response->set_source_lease_id(operation.source_lease_id);
    response->set_request_id(operation.prepare.request_id);
    response->set_deadline_unix_ms(operation.prepare.deadline_unix_ms);
    response->set_state(static_cast<int32_t>(operation.state));
    response->set_committed_version(operation.committed_version);
    response->set_rank_count(operation.rank_count);
    response->set_has_copy_plan(true);
    for (const auto index : operation.missing_indices) {
        response->add_missing_indices(index);
    }
    response->set_lease_remaining_ms(
        active ? std::max<int64_t>(0,
                                   std::chrono::duration_cast<std::chrono::milliseconds>(
                                       operation.expires_at - PrefixCacheTransactionManager::Clock::now())
                                       .count()) :
                 0);
    for (const auto key : operation.keys) {
        response->add_cache_keys(static_cast<int64_t>(key));
    }
}

grpc::Status LocalRpcServer::PreparePrefixExport(grpc::ServerContext*                context,
                                                 const PreparePrefixExportRequestPB* request,
                                                 PreparePrefixExportResponsePB*      response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    if (request->request_id().empty() || request->request_id().size() > 128 || request->deadline_unix_ms() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix prepare identity is required");
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, true);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager->supportsPrefixRankReplication()) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "prefix transfer requires a replicated full-KV layout");
    }
    auto        manager      = cache_manager->prefixTransactionManager();
    auto        cache_status = getCacheStatusInfo(-1, true);
    const auto& inventory    = cache_status.inventory;
    if (request->namespace_() != controller.namespace_id || request->epoch() != inventory.epoch) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export identity mismatch");
    }
    const auto                                           epoch = prefixEpochId(request->epoch());
    const auto                                           keys  = prefixKeys(request->cache_keys());
    const PrefixCacheTransactionManager::PrepareIdentity identity{request->request_id(),
                                                                  request->namespace_(),
                                                                  request->deadline_unix_ms(),
                                                                  prefixTransferIdentity(request->transfer())};
    auto lease = manager->getPreparedExport(epoch, request->namespace_(), request->request_id());
    if (lease) {
        if (!(lease->controller == controller) || !(lease->prepare == identity) || lease->version != request->version()
            || lease->keys != keys) {
            return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "prefix request ID has different parameters");
        }
        lease = manager->prepareExport(epoch, request->version(), keys, identity, controller);
        if (!lease) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export replay fenced");
        }
        makePrefixExportResponse(*lease, request->epoch(), response);
        return grpc::Status::OK;
    }
    if (request->version() != inventory.version || !inventory.device_ready) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export snapshot is not available");
    }
    const std::unordered_set<int64_t> ready_keys(inventory.device_keys.begin(), inventory.device_keys.end());
    for (const auto key : request->cache_keys()) {
        if (ready_keys.find(key) == ready_keys.end()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export key is not rank-ready");
        }
    }
    if (!cache_manager->allocator()->getBlockPool()->blockCache()->isKnownPrefixPath(keys)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export requires a known root-to-leaf path");
    }
    lease = manager->prepareExport(epoch, request->version(), keys, identity, controller);
    if (!lease) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "prefix export prepare failed or deadline expired");
    }
    makePrefixExportResponse(*lease, request->epoch(), response);
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::PreparePrefixImport(grpc::ServerContext*                context,
                                                 const PreparePrefixImportRequestPB* request,
                                                 PreparePrefixImportResponsePB*      response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    if (request->request_id().empty() || request->request_id().size() > 128 || request->deadline_unix_ms() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix prepare identity is required");
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, false);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager->supportsPrefixRankReplication()) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "prefix transfer requires a replicated full-KV layout");
    }
    auto        manager      = cache_manager->prefixTransactionManager();
    auto        cache_status = getCacheStatusInfo(-1, true);
    const auto& inventory    = cache_status.inventory;
    if (request->namespace_() != controller.namespace_id || request->epoch() != inventory.epoch) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix import identity mismatch");
    }
    const auto epoch      = prefixEpochId(request->epoch());
    const auto keys       = prefixKeys(request->cache_keys());
    const int  rank_count = request->rank_count() == 0 ? 1 : request->rank_count();
    if (rank_count != maga_init_params_.parallelism_config.tp_size) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "prefix import rank count must match the worker group");
    }
    const PrefixCacheTransactionManager::PrepareIdentity identity{request->request_id(),
                                                                  request->namespace_(),
                                                                  request->deadline_unix_ms(),
                                                                  prefixTransferIdentity(request->transfer())};
    auto operation = manager->getPreparedImport(epoch, request->namespace_(), request->request_id());
    if (operation) {
        if (!(operation->controller == controller) || !(operation->prepare == identity)
            || operation->expected_version != request->expected_version() || operation->keys != keys
            || operation->rank_count != rank_count || operation->source_lease_id != request->source_lease_id()) {
            return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "prefix request ID has different parameters");
        }
        operation = manager->prepareImport(
            epoch, request->expected_version(), keys, rank_count, request->source_lease_id(), identity, controller);
        if (!operation) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix import replay fenced");
        }
        makePrefixImportResponse(*operation, request->epoch(), response);
        return grpc::Status::OK;
    }
    if (request->expected_version() != inventory.version || !inventory.device_ready) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix import snapshot is not available");
    }
    operation = manager->prepareImport(
        epoch, request->expected_version(), keys, rank_count, request->source_lease_id(), identity, controller);
    if (!operation) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "prefix import prepare failed or deadline expired");
    }
    makePrefixImportResponse(*operation, request->epoch(), response);
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetPrefixPrepare(grpc::ServerContext*             context,
                                              const GetPrefixPrepareRequestPB* request,
                                              GetPrefixPrepareResponsePB*      response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    if (request->request_id().empty() || request->request_id().size() > 128
        || (request->kind() != PREFIX_PREPARE_EXPORT && request->kind() != PREFIX_PREPARE_IMPORT)) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid prefix prepare query");
    }
    auto cache_status = getCacheStatusInfo(-1, false);
    if (request->namespace_() != controller.namespace_id || request->epoch() != cache_status.inventory.epoch) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix prepare query identity mismatch");
    }
    auto       manager = engine_->getCacheManager()->prefixTransactionManager();
    const auto epoch   = prefixEpochId(request->epoch());
    response->Clear();
    if (request->kind() == PREFIX_PREPARE_EXPORT) {
        auto source = manager->getPreparedExport(epoch, request->namespace_(), request->request_id());
        if (source) {
            makePrefixExportResponse(*source, request->epoch(), response->mutable_export_result());
            return grpc::Status::OK;
        }
    } else {
        auto operation = manager->getPreparedImport(epoch, request->namespace_(), request->request_id());
        if (operation) {
            makePrefixImportResponse(*operation, request->epoch(), response->mutable_import_result());
            return grpc::Status::OK;
        }
    }
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "prefix prepare request not found");
}

grpc::Status LocalRpcServer::CommitPrefixImport(grpc::ServerContext*               context,
                                                const CommitPrefixImportRequestPB* request,
                                                PrefixOperationResponsePB*         response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, false);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto       cache_manager    = engine_->getCacheManager();
    auto       cache_status     = getCacheStatusInfo(-1, false);
    const auto operation        = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
    const bool remote_import    = operation.source_lease_id != 0;
    const bool committed_replay = operation.state == PrefixCacheTransactionManager::State::COMMITTED;
    auto       identity =
        validatePrefixCommit(operation,
                             *request,
                             cache_status.inventory.epoch,
                             cache_status.inventory.version,
                             operation.pinned_path_binding
                                 && cache_manager->prefixTransactionManager()->importSnapshotCurrent(operation.id));
    if (!identity.ok()) {
        return identity;
    }
    const auto epoch          = prefixEpochId(request->epoch());
    auto       receive_handle = prefix_receives_.get(request->operation_id(), epoch);
    if (receive_handle && !receive_handle->wait()) {
        prefix_receives_.cancel(request->operation_id(), epoch, [&] {
            return cache_manager->prefixTransactionManager()->abortImport(request->operation_id(), epoch, controller);
        });
        const auto current   = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
        const bool committed = remote_import && current.state == PrefixCacheTransactionManager::State::COMMITTED
                               && cache_manager->prefixTransactionManager()->commitImportedData(
                                   request->operation_id(), request->lease_id(), epoch, request->version(), controller);
        response->set_success(committed);
        response->set_version(static_cast<int64_t>(
            committed ? current.committed_version : cache_manager->prefixTransactionManager()->mutationVersion()));
        response->set_message(committed ? "" : "prefix receive did not complete");
        return grpc::Status::OK;
    }
    bool success = false;
    if (remote_import) {
        if (receive_handle && operation.state == PrefixCacheTransactionManager::State::COPYING) {
            cache_manager->prefixTransactionManager()->markImportRankReady(
                request->operation_id(), prefixEpochId(request->epoch()), 0, controller);
        }
        if (!committed_replay && operation.rank_count > 1 && (receive_handle || operation.missing_indices.empty())) {
            auto completion =
                cache_manager->prefixRankInstallQueue()->enqueue(operation.id, epoch, request->version(), controller);
            engine_->wakeupForPrefixCacheInstall();
            const auto remaining = context->deadline() - std::chrono::system_clock::now();
            const auto now       = PrefixCacheTransactionManager::Clock::now();
            const auto deadline =
                now
                + std::min(operation.expires_at - now,
                           std::chrono::duration_cast<PrefixCacheTransactionManager::Clock::duration>(remaining));
            if (completion.wait_until(deadline) != std::future_status::ready || !completion.get()) {
                cache_manager->prefixTransactionManager()->abortImport(operation.id, epoch, controller);
                prefix_receives_.release(operation.id, epoch);
                response->set_success(false);
                response->set_version(cache_manager->prefixTransactionManager()->mutationVersion());
                response->set_message("prefix rank installation failed or timed out");
                return grpc::Status::OK;
            }
        }
        if (receive_handle || committed_replay || operation.missing_indices.empty()
            || cache_manager->prefixTransactionManager()->getImport(request->operation_id()).state
                   == PrefixCacheTransactionManager::State::COMMITTED) {
            success = cache_manager->prefixTransactionManager()->commitImportedData(request->operation_id(),
                                                                                    request->lease_id(),
                                                                                    prefixEpochId(request->epoch()),
                                                                                    request->version(),
                                                                                    controller);
        }
    } else {
        success = cache_manager->prefixTransactionManager()->commitImport(request->operation_id(),
                                                                          request->lease_id(),
                                                                          prefixEpochId(request->epoch()),
                                                                          request->version(),
                                                                          controller);
    }
    response->set_success(success);
    const auto committed_operation = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
    if (success || committed_operation.state == PrefixCacheTransactionManager::State::ABORTED) {
        prefix_receives_.release(request->operation_id(), epoch);
    }
    response->set_version(static_cast<int64_t>(success ? committed_operation.committed_version :
                                                         cache_manager->prefixTransactionManager()->mutationVersion()));
    if (!success) {
        response->set_message("prefix import commit rejected");
    } else {
        engine_->wakeupForPrefixCacheInstall();
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::validatePrefixCommit(const PrefixCacheTransactionManager::ImportOperation& operation,
                                                  const CommitPrefixImportRequestPB&                    request,
                                                  const std::string&                                    current_epoch,
                                                  uint64_t                                              current_version,
                                                  bool pinned_path_valid) {
    if (operation.id == 0 || operation.id != request.operation_id()) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "prefix operation not found");
    }
    const auto source_lease_id = operation.source_lease_id ? operation.source_lease_id : operation.local_export_id;
    const auto controller      = prefixControllerIdentity(request.controller());
    if (!controller.valid() || !(controller == operation.controller)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix commit controller mismatch");
    }
    if (!prefixTransferIdentity(request.transfer()).valid()
        || !(operation.prepare.transfer == prefixTransferIdentity(request.transfer())) || request.epoch().empty()
        || request.epoch() != current_epoch || prefixEpochId(request.epoch()) != operation.epoch
        || request.version() != operation.expected_version || request.lease_id() == 0
        || (source_lease_id != 0 && request.lease_id() != source_lease_id)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix commit identity mismatch");
    }
    if (operation.state != PrefixCacheTransactionManager::State::COMMITTED && request.version() != current_version
        && !(operation.pinned_path_binding && pinned_path_valid)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix commit snapshot mismatch");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::AbortPrefixOperation(grpc::ServerContext*                 context,
                                                  const AbortPrefixOperationRequestPB* request,
                                                  PrefixOperationResponsePB*           response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, false);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto       cache_manager = engine_->getCacheManager();
    const auto operation     = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
    if (operation.id == 0 || !(operation.prepare.transfer == prefixTransferIdentity(request->transfer()))) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix operation transfer mismatch");
    }
    const auto epoch   = prefixEpochId(request->epoch());
    bool       success = prefix_receives_.cancel(request->operation_id(), epoch, [&] {
        return cache_manager->prefixTransactionManager()->abortImport(request->operation_id(), epoch, controller);
    });
    response->set_success(success);
    response->set_version(static_cast<int64_t>(cache_manager->prefixTransactionManager()->mutationVersion()));
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::ReleasePrefixExport(grpc::ServerContext*                context,
                                                 const ReleasePrefixExportRequestPB* request,
                                                 PrefixOperationResponsePB*          response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, true);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto       cache_manager = engine_->getCacheManager();
    const auto lease         = cache_manager->prefixTransactionManager()->getExport(request->lease_id());
    if (!lease || !(lease->prepare.transfer == prefixTransferIdentity(request->transfer()))) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export transfer mismatch");
    }
    bool success = cache_manager->prefixTransactionManager()->releaseExport(
        request->lease_id(), prefixEpochId(request->epoch()), controller);
    response->set_success(success);
    response->set_version(static_cast<int64_t>(cache_manager->prefixTransactionManager()->mutationVersion()));
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetPrefixOperation(grpc::ServerContext*               context,
                                                const GetPrefixOperationRequestPB* request,
                                                GetPrefixOperationResponsePB*      response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto operation = engine_->getCacheManager()->prefixTransactionManager()->getImport(request->operation_id());
    if (operation.id == 0) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "prefix operation not found");
    }
    auto cache_status = getCacheStatusInfo(-1, false);
    return makePrefixOperationResponse(operation, request->epoch(), cache_status.inventory.epoch, response);
}

grpc::Status
LocalRpcServer::makePrefixOperationResponse(const PrefixCacheTransactionManager::ImportOperation& operation,
                                            const std::string&                                    requested_epoch,
                                            const std::string&                                    current_epoch,
                                            GetPrefixOperationResponsePB*                         response) {
    if (operation.id == 0) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "prefix operation not found");
    }
    if (requested_epoch.empty() || requested_epoch != current_epoch
        || prefixEpochId(requested_epoch) != operation.epoch) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix operation epoch mismatch");
    }
    response->Clear();
    response->set_operation_id(operation.id);
    prefixControllerResponse(operation.controller, response->mutable_controller());
    prefixTransferResponse(operation.prepare.transfer, response->mutable_transfer());
    response->set_epoch(requested_epoch);
    response->set_expected_version(static_cast<int64_t>(operation.expected_version));
    response->set_state(static_cast<int32_t>(operation.state));
    response->set_committed_version(operation.committed_version);
    response->set_source_lease_id(operation.source_lease_id ? operation.source_lease_id : operation.local_export_id);
    response->set_has_copy_plan(true);
    for (const auto index : operation.missing_indices) {
        response->add_missing_indices(index);
    }
    for (const auto key : operation.keys) {
        response->add_cache_keys(static_cast<int64_t>(key));
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::MarkPrefixRankReady(grpc::ServerContext*                context,
                                                 const MarkPrefixRankReadyRequestPB* request,
                                                 PrefixOperationResponsePB*          response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, false);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto       cache_manager = engine_->getCacheManager();
    const auto operation     = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
    if (operation.id == 0 || !(operation.prepare.transfer == prefixTransferIdentity(request->transfer()))) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix operation transfer mismatch");
    }
    bool success = cache_manager->prefixTransactionManager()->markImportRankReady(
        request->operation_id(), prefixEpochId(request->epoch()), request->rank(), controller);
    response->set_success(success);
    response->set_version(static_cast<int64_t>(cache_manager->prefixTransactionManager()->mutationVersion()));
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::ReclaimPrefix(grpc::ServerContext*          context,
                                           const ReclaimPrefixRequestPB* request,
                                           PrefixOperationResponsePB*    response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto        cache_manager = engine_->getCacheManager();
    auto        cache_status  = getCacheStatusInfo(-1, true);
    const auto& inventory     = cache_status.inventory;
    if (request->namespace_() != controller.namespace_id || request->epoch() != inventory.epoch
        || request->expected_version() != inventory.version || !inventory.device_ready) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix reclaim snapshot is not available");
    }
    bool success = cache_manager->prefixTransactionManager()->reclaim(
        prefixEpochId(request->epoch()), request->expected_version(), prefixKeys(request->cache_keys()), controller);
    response->set_success(success);
    response->set_version(static_cast<int64_t>(cache_manager->prefixTransactionManager()->mutationVersion()));
    if (!success) {
        response->set_message("prefix reclaim rejected");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartPrefixSend(grpc::ServerContext*                   context,
                                             const PrefixTransferEndpointRequestPB* request,
                                             PrefixOperationResponsePB*             response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, true);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto cache_manager = engine_->getCacheManager();
    auto lease         = cache_manager->prefixTransactionManager()->getExport(request->lease_id());
    if (!lease.has_value() || !(lease->prepare.transfer == prefixTransferIdentity(request->transfer()))
        || request->operation_id() == 0 || request->peer_ip().empty() || request->peer_port() == 0
        || request->peer_port() > 65535 || request->timeout_ms() <= 0) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix send resources are unavailable");
    }
    if (!request->has_copy_plan()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix copy plan is required");
    }
    uint64_t previous = 0;
    bool     first    = true;
    for (const auto index : request->missing_indices()) {
        if (index >= lease->keys.size() || (!first && index <= previous)) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix copy indices must be ordered and unique");
        }
        previous = index;
        first    = false;
    }
    if (request->missing_indices().empty()) {
        response->set_success(cache_manager->prefixTransactionManager()
                                  ->pinExport(request->lease_id(), prefixEpochId(request->epoch()), controller)
                                  .has_value());
        return grpc::Status::OK;
    }
    auto coordinator = cache_manager->connectorCoordinator();
    auto backend     = coordinator ? coordinator->prefixTransferBackend() : transfer::TransferBackendPair{};
    if (!backend.sender || !backend.receiver) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix send transport is unavailable");
    }
    auto resource_lease = cache_manager->prefixTransactionManager()->pinExport(
        request->lease_id(), prefixEpochId(request->epoch()), controller);
    if (!resource_lease.has_value()) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix export lease is not active");
    }
    auto adapter = std::make_shared<transfer::PrefixTransferAdapter>(backend.sender, backend.receiver);
    transfer::PrefixTransferDispatcher        dispatcher(std::move(adapter));
    const int64_t                             deadline_ms = currentTimeMs() + request->timeout_ms();
    std::vector<transfer::PrefixTransferPair> pairs;
    pairs.reserve(request->missing_indices_size());
    const auto wire_prefix = lease->prepare.transfer.wirePrefix(lease->controller, lease->id, request->operation_id());
    for (const auto index : request->missing_indices()) {
        auto source       = std::make_shared<transfer::KeyBlockInfo>();
        source->cache_key = lease->keys[index];
        source->blocks    = cache_manager->prefixBlockBuffers(static_cast<int>(lease->blocks[index]));
        if (source->blocks.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix source block is unavailable");
        }
        pairs.push_back({wire_prefix + ":" + std::to_string(static_cast<uint64_t>(lease->keys[index])),
                         request->peer_ip(),
                         request->peer_port(),
                         deadline_ms,
                         source,
                         nullptr});
    }
    const bool success = dispatcher.dispatchSend(pairs, request->timeout_ms(), std::move(*resource_lease));
    response->set_success(success);
    response->set_message(success ? "" : "prefix send failed");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartPrefixReceive(grpc::ServerContext*                   context,
                                                const PrefixTransferEndpointRequestPB* request,
                                                PrefixOperationResponsePB*             response) {
    (void)context;
    const auto controller = prefixControllerIdentity(request->controller());
    if (!controller.valid()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "prefix controller identity is required");
    }
    auto controller_status = validatePrefixController(request->controller(),
                                                      request->epoch(),
                                                      getCacheStatusInfo(-1, false).inventory,
                                                      engine_->getCacheManager()->prefixTransactionManager());
    if (!controller_status.ok()) {
        return controller_status;
    }
    auto transfer_status = validatePrefixTransfer(request->transfer(), getCacheStatusInfo(-1, false).inventory, false);
    if (!transfer_status.ok()) {
        return transfer_status;
    }
    auto cache_manager = engine_->getCacheManager();
    auto operation     = cache_manager->prefixTransactionManager()->getImport(request->operation_id());
    if (operation.id == 0 || !(operation.prepare.transfer == prefixTransferIdentity(request->transfer()))
        || request->lease_id() == 0 || operation.source_lease_id != request->lease_id() || request->peer_ip().empty()
        || request->peer_port() == 0 || request->peer_port() > 65535 || request->timeout_ms() <= 0) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix receive resources are unavailable");
    }
    const std::vector<uint32_t> requested_indices(request->missing_indices().begin(), request->missing_indices().end());
    if (!request->has_copy_plan() || requested_indices != operation.missing_indices) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix receive copy plan mismatch");
    }
    if (operation.missing_indices.empty()) {
        const auto epoch   = prefixEpochId(request->epoch());
        auto       manager = cache_manager->prefixTransactionManager();
        const bool success = manager->beginImportTransfer(operation.id, epoch, controller)
                             && manager->markImportRankReady(operation.id, epoch, 0, controller);
        response->set_success(success);
        return grpc::Status::OK;
    }
    auto coordinator = cache_manager->connectorCoordinator();
    auto backend     = coordinator ? coordinator->prefixTransferBackend() : transfer::TransferBackendPair{};
    if (!backend.sender || !backend.receiver) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix receive transport is unavailable");
    }
    auto resource_lease = cache_manager->prefixTransactionManager()->pinImport(
        request->operation_id(), prefixEpochId(request->epoch()), controller);
    if (!resource_lease.has_value()
        || !cache_manager->prefixTransactionManager()->beginImportTransfer(
            request->operation_id(), prefixEpochId(request->epoch()), controller)) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix import is not ready for receive");
    }
    auto adapter = std::make_shared<transfer::PrefixTransferAdapter>(backend.sender, backend.receiver);
    transfer::PrefixTransferDispatcher        dispatcher(std::move(adapter));
    const int64_t                             deadline_ms = currentTimeMs() + request->timeout_ms();
    std::vector<transfer::PrefixTransferPair> pairs;
    pairs.reserve(operation.missing_indices.size());
    const auto wire_prefix =
        operation.prepare.transfer.wirePrefix(operation.controller, operation.source_lease_id, operation.id);
    for (const auto index : operation.missing_indices) {
        auto destination       = std::make_shared<transfer::KeyBlockInfo>();
        destination->cache_key = operation.keys[index];
        destination->blocks = cache_manager->prefixBlockBuffers(static_cast<int>(operation.destination_blocks[index]));
        if (destination->blocks.empty()) {
            cache_manager->prefixTransactionManager()->abortImport(
                request->operation_id(), prefixEpochId(request->epoch()), controller);
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "prefix destination block is unavailable");
        }
        pairs.push_back({wire_prefix + ":" + std::to_string(static_cast<uint64_t>(operation.keys[index])),
                         "",
                         0,
                         deadline_ms,
                         nullptr,
                         destination});
    }
    auto       handle  = dispatcher.prepareReceive(pairs, request->timeout_ms(), std::move(*resource_lease));
    const auto id      = request->operation_id();
    const auto epoch   = prefixEpochId(request->epoch());
    const bool success = prefix_receives_.store(
        id,
        epoch,
        std::move(handle),
        [cache_manager, id, epoch] {
            const auto current = cache_manager->prefixTransactionManager()->getImport(id);
            return current.epoch == epoch
                   && (current.state == PrefixCacheTransactionManager::State::COPYING
                       || current.state == PrefixCacheTransactionManager::State::DATA_READY);
        },
        [cache_manager, id, epoch, controller] {
            return cache_manager->prefixTransactionManager()->abortImport(id, epoch, controller);
        });
    if (!success) {
        cache_manager->prefixTransactionManager()->abortImport(
            request->operation_id(), prefixEpochId(request->epoch()), controller);
    }
    response->set_success(success);
    response->set_message(success ? "" : "prefix receive registration failed");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetWorkerStatus(grpc::ServerContext*   context,
                                             const StatusVersionPB* request,
                                             WorkerStatusPB*        response) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t request_begin_time_us   = currentTimeUs();
    int64_t latest_finished_version = request->latest_finished_version();
    RTP_LLM_LOG_DEBUG(
        "receive workerStatus rpc request from client: %s, latest_finished_version: %ld, config role_type: %d",
        context->peer().c_str(),
        latest_finished_version,
        maga_init_params_.pd_sep_config.role_type);

    WorkerStatusInfo status_info              = getWorkerStatusInfo(latest_finished_version);
    int64_t          request_after_ws_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("getWorkerStatusInfo took %ld us", request_after_ws_time_us - request_begin_time_us);

    const auto& engine_schedule_info = status_info.engine_schedule_info;
    response->set_role(status_info.role);

    for (const auto& task : engine_schedule_info.running_task_info_list) {
        TaskInfoPB* task_info = response->add_running_task_info();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_batch_id(task.batch_id);
        task_info->set_is_waiting(task.is_waiting);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_execution_time_ms(task.execution_time_ms);
    }

    for (const auto& task : engine_schedule_info.finished_task_info_list) {
        TaskInfoPB* task_info = response->add_finished_task_list();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_batch_id(task.batch_id);
        task_info->set_is_waiting(task.is_waiting);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_execution_time_ms(task.execution_time_ms);
    }
    response->set_dp_size(status_info.dp_size);
    response->set_tp_size(status_info.tp_size);
    response->set_status_version(status_info.status_version);
    response->set_latest_finished_version(status_info.latest_finished_version);
    response->set_alive(status_info.alive);
    response->set_precision(status_info.precision);
    reportWorkerStatusTime(request_begin_time_us, request_after_ws_time_us);
    return grpc::Status::OK;
}

WorkerStatusInfo LocalRpcServer::getWorkerStatusInfo(int64_t latest_finished_version) {
    WorkerStatusInfo status_info;
    status_info.engine_schedule_info = getEngineScheduleInfo(latest_finished_version);
    switch (maga_init_params_.pd_sep_config.role_type) {
        case RoleType::PDFUSION:
            status_info.role = "RoleType.PDFUSION";
            break;
        case RoleType::PREFILL:
            status_info.role = "RoleType.PREFILL";
            break;
        case RoleType::DECODE:
            status_info.role = "RoleType.DECODE";
            break;
        case RoleType::VIT:
            status_info.role = "RoleType.VIT";
            break;
        case RoleType::FRONTEND:
            status_info.role = "RoleType.FRONTEND";
            break;
        default:
            status_info.role = "RoleType.UNKNOWN";
    }
    status_info.dp_size                 = maga_init_params_.parallelism_config.dp_size;
    status_info.tp_size                 = maga_init_params_.parallelism_config.tp_size;
    status_info.dp_rank                 = maga_init_params_.parallelism_config.dp_rank;
    status_info.status_version          = currentTimeUs();
    status_info.latest_finished_version = status_info.engine_schedule_info.latest_finished_version;
    status_info.alive                   = true;
    auto quant_method                   = maga_init_params_.model_config_.quant_algo.getQuantMethod();

    switch (quant_method) {
        case QuantMethod::WeightOnlyPerCol:
            status_info.precision = "WeightOnlyPerCol";
            break;
        case QuantMethod::GptQ:
            status_info.precision = "GptQ";
            break;
        case QuantMethod::Awq:
            status_info.precision = "Awq";
            break;
        case QuantMethod::SmoothQuant:
            status_info.precision = "SmoothQuant";
            break;
        case QuantMethod::OmniQuant:
            status_info.precision = "OmniQuant";
            break;
        case QuantMethod::PerTensorQuant:
            status_info.precision = "PerTensorQuant";
            break;
        case QuantMethod::FP8Quant:
            status_info.precision = "FP8Quant";
            break;
        case QuantMethod::FP8PTPC:
            status_info.precision = "FP8PTPC";
            break;
        case QuantMethod::W4A8INT4PTPC:
            status_info.precision = "W4A8INT4PTPC";
            break;
        case QuantMethod::None:
            status_info.precision = "FP16";
            break;
        default:
            RTP_LLM_LOG_ERROR("unknown quant method: %d", static_cast<int>(quant_method));
            status_info.precision = "UNKNOWN";
    }
    return status_info;
}

KVCacheInfo LocalRpcServer::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    int64_t     request_begin_time_us = currentTimeUs();
    const auto& cache_info            = engine_->getCacheStatusInfo(latest_version, need_cache_keys);
    reportCacheStatusTime(request_begin_time_us);
    return cache_info;
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finished_version) {
    static const bool refresh_running_prefix = []() {
        const char* value   = std::getenv("RTP_LLM_RUNNING_PREFIX_REPORT");
        const bool  enabled = value != nullptr && std::string(value) == "1";
        RTP_LLM_LOG_INFO("running_prefix_report enabled=%d", int(enabled));
        return enabled;
    }();
    EngineScheduleInfo                        info = meta_->getEngineScheduleInfo(latest_finished_version);
    std::vector<EngineScheduleInfo::TaskInfo> running_task_info_list = engine_->getScheduler().runningTaskList();
    info.mergeRunningSnapshot(running_task_info_list, refresh_running_prefix);
    auto last_schedule_time = engine_->getLastScheduleTime();
    // in case last_schedule_delta is negative
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
}

grpc::Status LocalRpcServer::CancelIfReceived(grpc::ServerContext*,
                                              const CancelIfReceivedRequestPB* request,
                                              CancelIfReceivedResponsePB*      response) {
    if (request->request_id() <= 0 || request->operation_epoch() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "request_id and operation_epoch must be positive");
    }
    const auto result = meta_->cancelIfReceived(request->request_id(), request->operation_epoch());
    using Outcome     = RpcServerRuntimeMeta::CancelResult::Outcome;
    switch (result.outcome) {
        case Outcome::CANCELLED:
            response->set_outcome(CANCEL_IF_RECEIVED_CANCELLED);
            response->set_detail("request cancelled before KV allocation");
            break;
        case Outcome::NOT_FOUND:
            response->set_outcome(CANCEL_IF_RECEIVED_NOT_FOUND);
            response->set_detail("request is not present in the engine registry");
            break;
        case Outcome::NOT_RECEIVED:
            response->set_outcome(CANCEL_IF_RECEIVED_NOT_RECEIVED);
            response->set_detail("request has already crossed the execution boundary");
            break;
        case Outcome::ALREADY_CANCELLED:
            response->set_outcome(CANCEL_IF_RECEIVED_ALREADY_CANCELLED);
            response->set_detail("idempotent replay for this operation epoch");
            break;
    }
    response->set_observed_phase(static_cast<::TaskPhase>(result.observed_phase));
    response->set_accepted_operation_epoch(request->operation_epoch());
    response->set_idempotent_replay(result.idempotent_replay);
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::UpdateSchedulerInfo(grpc::ServerContext*                context,
                                                 const UpdateSchedulerInfoRequestPB* request,
                                                 EmptyPB*                            response) {
    const std::string             scheduler_info = request->scheduler_info();
    ParallelTopologySwitchRequest switch_request;
    try {
        autil::legacy::FromJsonString(switch_request, scheduler_info);
    } catch (const autil::legacy::ExceptionBase& e) {
        try {
            std::map<std::string, autil::legacy::Any> fields;
            autil::legacy::FromJsonString(fields, scheduler_info);
            const auto action = fields.find("action");
            if (action != fields.end()
                && autil::legacy::AnyCast<std::string>(action->second) == "switch_parallel_topology") {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "malformed topology plan");
            }
        } catch (const autil::legacy::ExceptionBase&) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "malformed scheduler control payload");
        }
        engine_->getScheduler().updateSchedulerInfo(scheduler_info);
        return grpc::Status::OK;
    }
    if (switch_request.action == "switch_parallel_topology") {
        RTP_LLM_LOG_INFO("receive switch_parallel_topology from %s: tp_size=%ld dp_size=%ld internal=%d phase=%s",
                         context->peer().c_str(),
                         switch_request.tp_size,
                         switch_request.dp_size,
                         int(switch_request.internal),
                         switch_request.phase.c_str());
        if (switch_request.tp_size <= 0 || switch_request.dp_size <= 0) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "tp_size and dp_size must be positive");
        }

        const bool prefix_control      = engine_->getCacheManager()->prefixInventoryEnabled();
        auto*      topology_manager    = engine_->getCacheManager()->prefixTransactionManager();
        const auto topology_controller = switch_request.prefix_controller.identity();
        const auto topology_binding    = switch_request.controllerBinding();
        if (switch_request.resident_cache && switch_request.resident_leader_merge) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "resident modes are mutually exclusive");
        }
        if ((switch_request.resident_cache || switch_request.resident_leader_merge) && !prefix_control) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "resident prefix mode requires prefix inventory control");
        }
        if (prefix_control
            && (!topology_controller.valid() || switch_request.switch_id.empty()
                || switch_request.switch_id.size() > 128 || switch_request.timeout_ms <= 0
                || switch_request.timeout_ms > 300000 || !topology_manager)) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "bounded prefix topology plan and controller are required");
        }
        if (prefix_control && !switch_request.internal && switch_request.phase != "commit") {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "external topology plans cannot select internal phases");
        }

        if (switch_request.internal && engine_->getCacheManager()->prefixInventoryEnabled()) {
            const auto identity = engine_->getCacheManager()->prefixTopologyIdentity();
            if (switch_request.expected_boot_id.empty() || switch_request.expected_epoch.empty()
                || switch_request.expected_world_rank < 0) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                    "prefix topology participant identity is required");
            }
            if (identity.boot_id.empty() || switch_request.expected_boot_id != identity.boot_id
                || switch_request.expected_world_rank != identity.world_rank) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "prefix topology participant incarnation changed");
            }
            if (switch_request.phase == "confirm") {
                return topology_manager->confirmTopologyController(topology_controller, topology_binding) ?
                           grpc::Status::OK :
                           grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "topology release receipt does not match");
            }
            if (switch_request.phase == "drain") {
                const auto inventory = engine_->getCacheManager()->getKVCacheInfo(-1, false).inventory;
                if (inventory.epoch != switch_request.expected_epoch
                    || inventory.namespace_id != topology_controller.namespace_id
                    || !topology_manager->bindTopologyController(topology_controller, topology_binding)) {
                    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "prefix topology controller admission changed");
                }
            } else if (!(switch_request.phase == "quarantine" ?
                             topology_manager->bindTopologyQuarantine(topology_controller, topology_binding) :
                             topology_manager->topologyControllerMatches(topology_controller, topology_binding))) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "prefix topology phase does not match admitted plan");
            }
            if (switch_request.phase != "drain" && switch_request.phase != "flush" && switch_request.phase != "arm"
                && switch_request.phase != "prepare" && switch_request.phase != "prefix_prepare"
                && switch_request.phase != "prefix_resident" && switch_request.phase != "commit"
                && switch_request.phase != "cancel" && switch_request.phase != "resume"
                && switch_request.phase != "settle" && switch_request.phase != "release"
                && switch_request.phase != "quarantine") {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "unknown prefix topology phase");
            }
        }

        if (switch_request.internal && switch_request.phase == "arm") {
            return abslToGrpcStatus(
                engine_->armParallelTopologySwitch(switch_request.pause_at_ms, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "prepare") {
            return abslToGrpcStatus(
                engine_->prepareParallelTopologySwitch(switch_request.timeout_ms, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "cancel") {
            const auto status = engine_->finishParallelTopologySwitch(true, switch_request.switch_id);
            if (status.ok() && prefix_control
                && !topology_manager->releaseTopologyController(topology_controller, topology_binding)) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "cannot release cancelled topology controller");
            }
            return abslToGrpcStatus(status);
        }
        if (switch_request.internal && switch_request.phase == "resume") {
            return abslToGrpcStatus(engine_->finishParallelTopologySwitch(false, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "prefix_prepare") {
            return abslToGrpcStatus(engine_->preparePrefixTopologySwitch(
                switch_request.tp_size, switch_request.dp_size, switch_request.timeout_ms, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "prefix_resident") {
            if (!switch_request.resident_cache && !switch_request.resident_leader_merge) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "resident prefix mode is required");
            }
            return abslToGrpcStatus(engine_->prepareResidentPrefixTopologySwitch(switch_request.tp_size,
                                                                                 switch_request.dp_size,
                                                                                 switch_request.switch_id,
                                                                                 switch_request.resident_leader_merge));
        }
        if (switch_request.internal && switch_request.phase == "quarantine") {
            return abslToGrpcStatus(engine_->quarantineParallelTopologySwitch(switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "drain") {
            return abslToGrpcStatus(engine_->drainParallelTopologySwitch(
                switch_request.timeout_ms, switch_request.switch_id, switch_request.expected_epoch));
        }
        if (switch_request.internal && switch_request.phase == "release") {
            const auto status = engine_->releaseParallelTopologySwitch(switch_request.switch_id);
            if (status.ok() && prefix_control
                && !topology_manager->releaseTopologyController(topology_controller, topology_binding, true)) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "cannot release completed topology controller");
            }
            return abslToGrpcStatus(status);
        }
        if (switch_request.internal && switch_request.phase == "settle") {
            return abslToGrpcStatus(
                engine_->settleParallelTopologySwitch(switch_request.timeout_ms, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase == "flush") {
            return abslToGrpcStatus(
                engine_->flushParallelTopologySwitch(switch_request.timeout_ms, switch_request.switch_id));
        }
        if (switch_request.internal && switch_request.phase != "commit") {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "unknown topology switch phase");
        }
        if (switch_request.internal && switch_request.resident_cache
            && !engine_->getCacheManager()->residentPrefixSplitPrepared(switch_request.tp_size)) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "resident prefix metadata must be prepared before topology commit");
        }
        if (switch_request.internal && switch_request.resident_leader_merge
            && !engine_->getCacheManager()->residentPrefixMergePrepared(switch_request.tp_size)) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "resident merge must be prepared before topology commit");
        }

        std::unique_lock<std::mutex> coordinator_lock(topology_switch_coordinator_mutex_, std::defer_lock);
        if (!switch_request.internal) {
            coordinator_lock.lock();
            const auto& current = maga_init_params_.parallelism_config;
            if (switch_request.resident_cache
                && (!current.prefill_cp_config.is_enabled() || current.tp_size % 2 != 0
                    || switch_request.tp_size != current.tp_size / 2
                    || !engine_->getCacheManager()->supportsPrefixRankReplication())) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "resident prefix mode requires a supported adjacent CP split");
            }
            if (switch_request.resident_leader_merge
                && (!current.prefill_cp_config.is_enabled() || switch_request.tp_size <= 0
                    || switch_request.tp_size % 2 != 0 || switch_request.tp_size / 2 != current.tp_size
                    || !engine_->getCacheManager()->supportsPrefixRankReplication())) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "resident merge requires a supported adjacent CP merge");
            }
            if (prefix_control) {
                const auto identity  = engine_->getCacheManager()->prefixTopologyIdentity();
                const auto inventory = engine_->getCacheManager()->getKVCacheInfo(-1, false).inventory;
                if (identity.world_rank != 0 || identity.world_size <= 0 || identity.world_size > 4096
                    || topology_controller.namespace_id != inventory.namespace_id
                    || !topology_manager->isControllerCurrent(topology_controller)
                    || switch_request.expected_participants.size() != static_cast<size_t>(identity.world_size)) {
                    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "prefix topology coordinator plan is not current");
                }
                std::unordered_set<std::string> expected_boots;
                for (size_t rank = 0; rank < switch_request.expected_participants.size(); ++rank) {
                    const auto& participant = switch_request.expected_participants[rank];
                    if (participant.world_rank != static_cast<int64_t>(rank) || participant.boot_id.empty()
                        || participant.boot_id.size() > 512 || participant.epoch.empty()
                        || participant.epoch.size() > 512 || !expected_boots.insert(participant.boot_id).second) {
                        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                            "invalid expected topology participant set");
                    }
                }
                const auto& expected = switch_request.expected_participants.front();
                if (expected.boot_id != identity.boot_id || expected.epoch != inventory.epoch) {
                    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "prefix topology coordinator incarnation changed");
                }
                if (identity.world_size > 1
                    && (!profile_broadcaster_
                        || profile_broadcaster_->workerNum() != static_cast<size_t>(identity.world_size))) {
                    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "prefix topology plan requires all world ranks");
                }
            }
            try {
                (void)makeSwitchedRpcParallelismConfig(
                    maga_init_params_.parallelism_config, switch_request.tp_size, switch_request.dp_size);
            } catch (const std::exception& error) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error.what());
            }
            if (switch_request.tp_size == maga_init_params_.parallelism_config.tp_size
                && switch_request.dp_size == maga_init_params_.parallelism_config.dp_size
                && (!prefix_control || maga_init_params_.parallelism_config.world_size == 1)) {
                return grpc::Status::OK;
            }
            if (switch_request.switch_id.empty()) {
                std::ifstream identity_source("/proc/sys/kernel/random/uuid");
                std::getline(identity_source, switch_request.switch_id);
            }
            if (switch_request.switch_id.empty() || switch_request.switch_id.size() > 128) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "bounded topology switch identity is required");
            }
            RTP_LLM_LOG_INFO("parallel topology switch identity=%s", switch_request.switch_id.c_str());
        }

        if (!switch_request.internal && maga_init_params_.parallelism_config.world_rank != 0) {
            RTP_LLM_LOG_INFO(
                "skip external switch_parallel_topology on non-coordinator rank, world_rank=%ld target_tp_size=%ld target_dp_size=%ld",
                maga_init_params_.parallelism_config.world_rank,
                switch_request.tp_size,
                switch_request.dp_size);
            return grpc::Status::OK;
        }

        if (!switch_request.internal && maga_init_params_.parallelism_config.world_rank == 0 && profile_broadcaster_
            && profile_broadcaster_->workerNum() > 1) {
            auto&                      scheduler      = engine_->getScheduler();
            const auto                 broadcaster    = profile_broadcaster_;
            const bool                 prefix_managed = engine_->getCacheManager()->prefixInventoryEnabled();
            std::vector<CacheStatusPB> participants;
            if (prefix_managed) {
                if (broadcaster->workerNum() != static_cast<size_t>(maga_init_params_.parallelism_config.world_size)) {
                    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                        "topology discovery must cover every world rank");
                }
                std::vector<CacheVersionPB> requests(broadcaster->workerNum());
                for (auto& request : requests) {
                    request.set_latest_cache_version(-1);
                    request.set_need_cache_keys(false);
                }
                auto discovery = broadcaster->broadcast<CacheVersionPB, CacheStatusPB>(
                    requests,
                    static_cast<int>(switch_request.timeout_ms),
                    [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& client_context,
                       const CacheVersionPB&                       request,
                       grpc::CompletionQueue*                      completion_queue) {
                        return stub->AsyncGetCacheStatus(client_context.get(), request, completion_queue);
                    });
                if (!discovery) {
                    return grpc::Status(grpc::StatusCode::UNAVAILABLE, "cannot discover topology participants");
                }
                try {
                    if (!discovery->waitDone(static_cast<int>(switch_request.timeout_ms)) || !discovery->success()) {
                        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "topology participant discovery failed");
                    }
                } catch (const std::exception& error) {
                    return grpc::Status(grpc::StatusCode::UNAVAILABLE, error.what());
                }
                participants = discovery->responses();
                std::unordered_set<std::string> boot_ids;
                const auto                      local_namespace =
                    engine_->getCacheManager()->getKVCacheInfo(-1, false).inventory.namespace_id;
                const auto local_identity = engine_->getCacheManager()->prefixTopologyIdentity();
                for (size_t rank = 0; rank < participants.size(); ++rank) {
                    const auto& participant = participants[rank];
                    const auto& identity    = participant.topology_identity();
                    const auto& expected    = switch_request.expected_participants[rank];
                    if (!participant.has_topology_identity() || identity.boot_id().empty()
                        || identity.world_rank() != static_cast<int64_t>(rank)
                        || identity.world_size() != static_cast<int64_t>(participants.size())
                        || (rank == 0 && identity.boot_id() != local_identity.boot_id)
                        || !boot_ids.insert(identity.boot_id()).second || !participant.has_inventory()
                        || participant.inventory().epoch().empty()
                        || participant.inventory().namespace_() != local_namespace
                        || identity.boot_id() != expected.boot_id || participant.inventory().epoch() != expected.epoch
                        || identity.world_rank() != expected.world_rank) {
                        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                            "invalid topology participant identity set");
                    }
                }
                if (switch_request.tp_size == maga_init_params_.parallelism_config.tp_size
                    && switch_request.dp_size == maga_init_params_.parallelism_config.dp_size) {
                    return grpc::Status::OK;
                }
            }
            const int64_t begin_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();
            if (!prefix_managed) {
                scheduler.setDrainMode(true);
                while (scheduler.onflightStreams() > 0) {
                    if (switch_request.timeout_ms > 0
                        && autil::TimeUtility::currentTimeInMilliSeconds() - begin_time_ms
                               > switch_request.timeout_ms) {
                        scheduler.setDrainMode(false);
                        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                            "timeout waiting scheduler to drain before topology switch");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
            const int64_t drain_done_ms = autil::TimeUtility::currentTimeInMilliSeconds();
            RTP_LLM_LOG_INFO("parallel topology switch coordinator pre-drain done, cost_ms=%ld",
                             drain_done_ms - begin_time_ms);
            const bool    signal_pause   = useSignalTopologySwitchPause();
            const int64_t pause_grace_ms = signal_pause ? 0 : getTopologySwitchPauseDelayMs();
            switch_request.pause_at_ms   = std::numeric_limits<int64_t>::max();
            RTP_LLM_LOG_INFO("parallel topology switch arm barrier configured, grace_ms=%ld", pause_grace_ms);

            auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                               const std::shared_ptr<grpc::ClientContext>& client_context,
                               const UpdateSchedulerInfoRequestPB&         internal_request,
                               grpc::CompletionQueue*                      completion_queue) {
                return stub->AsyncUpdateSchedulerInfo(client_context.get(), internal_request, completion_queue);
            };
            auto broadcast_phase = [&](const std::string& phase) -> grpc::Status {
                std::vector<UpdateSchedulerInfoRequestPB> requests(broadcaster->workerNum());
                for (size_t rank = 0; rank < requests.size(); ++rank) {
                    auto bound_request = switch_request;
                    if (prefix_managed) {
                        bound_request.expected_boot_id    = participants[rank].topology_identity().boot_id();
                        bound_request.expected_world_rank = participants[rank].topology_identity().world_rank();
                        bound_request.expected_epoch      = participants[rank].inventory().epoch();
                    }
                    requests[rank].set_scheduler_info(makeInternalSwitchSchedulerInfo(bound_request, phase));
                }
                auto broadcast_result = broadcaster->broadcast<UpdateSchedulerInfoRequestPB, EmptyPB>(
                    requests, static_cast<int>(switch_request.timeout_ms), rpc_call);
                if (!broadcast_result) {
                    return grpc::Status(grpc::StatusCode::INTERNAL,
                                        "failed to broadcast switch_parallel_topology to world ranks");
                }
                bool broadcast_finished = false;
                try {
                    broadcast_finished = broadcast_result->waitDone(static_cast<int>(switch_request.timeout_ms));
                } catch (const std::exception& e) {
                    return grpc::Status(grpc::StatusCode::INTERNAL,
                                        std::string("broadcast switch_parallel_topology exception: ") + e.what());
                }
                if (!broadcast_finished) {
                    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                        "timeout broadcasting switch_parallel_topology to world ranks");
                }
                if (!broadcast_result->success()) {
                    return grpc::Status(grpc::StatusCode::INTERNAL,
                                        "broadcast switch_parallel_topology to world ranks failed");
                }
                return grpc::Status::OK;
            };

            const auto drain_status = broadcast_phase("drain");
            if (!drain_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return drain_status;
            }
            const auto flush_status = broadcast_phase("flush");
            if (!flush_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return flush_status;
            }
            const auto arm_status = broadcast_phase("arm");
            if (!arm_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return arm_status;
            }
            switch_request.pause_at_ms =
                signal_pause ? 0 : autil::TimeUtility::currentTimeInMilliSeconds() + pause_grace_ms;
            RTP_LLM_LOG_INFO(
                "parallel topology switch pause signal configured after arm barrier, signal_mode=%d grace_ms=%ld pause_at_ms=%ld",
                int(signal_pause),
                pause_grace_ms,
                switch_request.pause_at_ms);
            const auto pause_signal_status = broadcast_phase("arm");
            if (!pause_signal_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return pause_signal_status;
            }
            const auto now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
            if (!signal_pause && now_ms < switch_request.pause_at_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(switch_request.pause_at_ms - now_ms));
            }
            const auto prepare_status = broadcast_phase("prepare");
            if (!prepare_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return prepare_status;
            }
            const auto prefix_prepare_status = broadcast_phase("prefix_prepare");
            if (!prefix_prepare_status.ok()) {
                const auto cancel_status = broadcast_phase("cancel");
                if (cancel_status.ok() && !prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return prefix_prepare_status;
            }
            if (switch_request.resident_cache || switch_request.resident_leader_merge) {
                const auto resident_status = broadcast_phase("prefix_resident");
                if (!resident_status.ok()) {
                    (void)broadcast_phase("cancel");
                    return resident_status;
                }
            }
            const auto commit_status = broadcast_phase("commit");
            if (!commit_status.ok()) {
                (void)broadcast_phase(prefix_managed ? "quarantine" : "cancel");
                if (!prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return commit_status;
            }
            const auto resume_status = broadcast_phase("resume");
            if (!resume_status.ok()) {
                (void)broadcast_phase(prefix_managed ? "quarantine" : "cancel");
                if (!prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return resume_status;
            }
            const auto    settle_status     = broadcast_phase("settle");
            const int64_t broadcast_done_ms = autil::TimeUtility::currentTimeInMilliSeconds();
            if (!settle_status.ok()) {
                (void)broadcast_phase(prefix_managed ? "quarantine" : "cancel");
                if (!prefix_managed) {
                    scheduler.setDrainMode(false);
                }
                return settle_status;
            }
            const auto release_status = broadcast_phase("release");
            if (!release_status.ok()) {
                (void)broadcast_phase(prefix_managed ? "quarantine" : "cancel");
                return release_status;
            }
            if (!prefix_managed) {
                scheduler.setDrainMode(false);
            }
            RTP_LLM_LOG_INFO("parallel topology switch broadcast done, cost_ms=%ld total_ms=%ld",
                             broadcast_done_ms - drain_done_ms,
                             broadcast_done_ms - begin_time_ms);
            return prefix_managed ? broadcast_phase("confirm") : grpc::Status::OK;
        }

        const int64_t switch_begin_ms = autil::TimeUtility::currentTimeInMilliSeconds();
        const auto    status          = engine_->switchParallelTopology(
            switch_request.tp_size, switch_request.dp_size, switch_request.timeout_ms, switch_request.switch_id);
        if (!status.ok()) {
            return abslToGrpcStatus(status);
        }
        try {
            maga_init_params_.parallelism_config = makeSwitchedRpcParallelismConfig(
                maga_init_params_.parallelism_config, switch_request.tp_size, switch_request.dp_size);
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INTERNAL,
                                std::string("failed to update rpc parallelism config: ") + e.what());
        }
        profile_broadcaster_ = makeTopologySwitchBroadcaster(maga_init_params_);
        if (!switch_request.internal) {
            return abslToGrpcStatus(engine_->finishParallelTopologySwitch(false, switch_request.switch_id));
        }
        RTP_LLM_LOG_INFO(
            "parallel topology switch local rpc state updated, cost_ms=%ld new_tp_rank=%ld broadcaster_size=%zu",
            autil::TimeUtility::currentTimeInMilliSeconds() - switch_begin_ms,
            maga_init_params_.parallelism_config.tp_rank,
            profile_broadcaster_ ? profile_broadcaster_->workerNum() : 0);
        return grpc::Status::OK;
    }
    engine_->getScheduler().updateSchedulerInfo(scheduler_info);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) {
    std::string log_level_str = request->log_level();
    uint32_t    log_level     = alog::LOG_LEVEL_INFO;
    if (log_level_str == "INFO" || log_level_str == "info") {
        log_level = alog::LOG_LEVEL_INFO;
    } else if (log_level_str == "WARNING" || log_level_str == "warning") {
        log_level = alog::LOG_LEVEL_WARN;
    } else if (log_level_str == "DEBUG" || log_level_str == "debug") {
        log_level = alog::LOG_LEVEL_DEBUG;
    } else if (log_level_str == "TRACE" || log_level_str == "trace") {
        log_level = alog::LOG_LEVEL_TRACE1;
    } else {
        RTP_LLM_LOG_WARNING("set log level failed, unknown log level: %s", log_level_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid log_level format");
    }
    auto& logger = rtp_llm::Logger::getEngineLogger();
    logger.setBaseLevel(log_level);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile from %s start_step=%d num_steps=%d enable_all_rank=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps(),
                     int(request->enable_all_rank()));
    if (!request->enable_all_rank()) {
        engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
        return grpc::Status::OK;
    }
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "enable_all_rank start_profile must be sent to tp_rank 0");
    }
    if (!profile_broadcaster_) {
        if (maga_init_params_.parallelism_config.tp_size <= 1) {
            RTP_LLM_LOG_INFO("start_profile enable_all_rank with tp_size=1, fallback to local start");
            engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
            return grpc::Status::OK;
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for enable_all_rank start_profile");
    }

    std::vector<StartProfileInternalRequestPB> requests(profile_broadcaster_->workerNum());
    for (auto& internal_request : requests) {
        internal_request.set_trace_name(request->trace_name());
        internal_request.set_start_step(request->start_step());
        internal_request.set_num_steps(request->num_steps());
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const StartProfileInternalRequestPB&        internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncStartProfileInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result = profile_broadcaster_->broadcast<StartProfileInternalRequestPB, EmptyPB>(
        requests, /*timeout_ms=*/3000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast start_profile_internal to tp group");
    }
    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast start_profile_internal to tp group failed");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartProfileInternal(grpc::ServerContext*                 context,
                                                  const StartProfileInternalRequestPB* request,
                                                  EmptyPB*                             response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile_internal from %s start_step=%d num_steps=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps());
    engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    response->set_health("OK");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::UpdateEplbConfig(grpc::ServerContext*             context,
                                              const UpdateEplbConfigRequestPB* request,
                                              EmptyPB*                         response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    const string mode_str = request->mode();
    EPLBConfig   config;
    if (mode_str == "EPLB" || mode_str == "eplb") {
        config.eplb_mode = EplbMode::EPLB;
    } else if (mode_str == "STATS" || mode_str == "stats") {
        config.eplb_mode = EplbMode::STATS;
    } else if (mode_str == "NONE" || mode_str == "none") {
        config.eplb_mode = EplbMode::NONE;
    } else if (mode_str == "ALL" || mode_str == "all") {
        config.eplb_mode = EplbMode::ALL;
    } else {
        RTP_LLM_LOG_WARNING("set eplb mode failed, unknown mode : %s", mode_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid eplb mode");
    }
    config.eplb_update_time = request->update_time();
    engine_->updateEplbConfig(config);
    return grpc::Status::OK;
}

void LocalRpcServer::reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_ws_time_us) {
    RpcWorkerStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = request_after_ws_time_us - request_begin_time_us;
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcWorkerStatusMetrics, RpcWorkerStatusMetricsCollector>(nullptr, &collector);
    }
}

void LocalRpcServer::reportCacheStatusTime(int64_t request_begin_time_us) {
    RpcCacheStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = (currentTimeUs() - request_begin_time_us);
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcCacheStatusMetrics, RpcCacheStatusMetricsCollector>(nullptr, &collector);
    }
}

::grpc::Status LocalRpcServer::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive execute function request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("execute function failed, request is cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("execute function failed, engine is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("execute function failed, cache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }
    if (!cache_manager->executeFunction(*request, *response)) {
        RTP_LLM_LOG_WARNING("execute function failed, request: [%s]", request->DebugString().c_str());
        const std::string error_msg = "execute function failed, request: [" + request->DebugString() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    engine_->pause();
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s,", context->peer().c_str());
    return abslToGrpcStatus(engine_->restart());
}

grpc::Status
LocalRpcServer::UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("Receive update weights request from: %s", context->peer().c_str());
    try {
        if (request->name().empty() || request->desc().empty() || request->method().empty()) {
            throw std::runtime_error("Missing required field(s) in request");
        }
        {
            py::gil_scoped_acquire acquire;
            py::dict               req;
            req["name"]   = request->name();
            req["desc"]   = request->desc();
            req["method"] = request->method();
            weight_manager_.attr("update")(req);
        }
        return grpc::Status::OK;
    } catch (const py::error_already_set& e) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        std::string err_msg = value ? PyUnicode_AsUTF8(value) : "Unknown Python error";
        return {grpc::StatusCode::INTERNAL, "exception from python: " + err_msg};
    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL, "exception from C++: " + std::string(e.what())};
    }
}
}  // namespace rtp_llm
