#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include "grpc++/grpc++.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferAdapter.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixReceiveRegistry.h"

namespace rtp_llm {
class LocalRpcServer {
public:
    static PrefixReplicatedLayoutPB makePrefixReplicatedLayout(const KVCacheManager& manager, const ModelConfig& model);
    static void                     serializePrefixInventory(const PrefixInventoryReport& inventory,
                                                             const CacheVersionPB&        request,
                                                             CacheReportV2PB*             report);
    LocalRpcServer() {}
    virtual ~LocalRpcServer() {}
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              py::object                                             mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status
    GetWorkerStatus(grpc::ServerContext* context, const ::StatusVersionPB* request, ::WorkerStatusPB* response);

    grpc::Status
    GetCacheStatus(grpc::ServerContext* context, const ::CacheVersionPB* request, ::CacheStatusPB* response);

    grpc::Status ActivatePrefixController(grpc::ServerContext*                     context,
                                          const ActivatePrefixControllerRequestPB* request,
                                          ActivatePrefixControllerResponsePB*      response);

    grpc::Status GetPrefixCacheCapabilities(grpc::ServerContext*                      context,
                                            const ::PrefixCacheCapabilitiesRequestPB* request,
                                            ::PrefixCacheCapabilitiesResponsePB*      response);

    grpc::Status InspectPrefix(grpc::ServerContext*            context,
                               const ::InspectPrefixRequestPB* request,
                               ::InspectPrefixResponsePB*      response);

    grpc::Status PreparePrefixExport(grpc::ServerContext*                  context,
                                     const ::PreparePrefixExportRequestPB* request,
                                     ::PreparePrefixExportResponsePB*      response);

    grpc::Status PreparePrefixImport(grpc::ServerContext*                  context,
                                     const ::PreparePrefixImportRequestPB* request,
                                     ::PreparePrefixImportResponsePB*      response);

    grpc::Status GetPrefixPrepare(grpc::ServerContext*               context,
                                  const ::GetPrefixPrepareRequestPB* request,
                                  ::GetPrefixPrepareResponsePB*      response);

    grpc::Status CommitPrefixImport(grpc::ServerContext*                 context,
                                    const ::CommitPrefixImportRequestPB* request,
                                    ::PrefixOperationResponsePB*         response);

    grpc::Status AbortPrefixOperation(grpc::ServerContext*                   context,
                                      const ::AbortPrefixOperationRequestPB* request,
                                      ::PrefixOperationResponsePB*           response);

    grpc::Status ReleasePrefixExport(grpc::ServerContext*                  context,
                                     const ::ReleasePrefixExportRequestPB* request,
                                     ::PrefixOperationResponsePB*          response);

    grpc::Status GetPrefixOperation(grpc::ServerContext*                 context,
                                    const ::GetPrefixOperationRequestPB* request,
                                    ::GetPrefixOperationResponsePB*      response);

    grpc::Status MarkPrefixRankReady(grpc::ServerContext*                  context,
                                     const ::MarkPrefixRankReadyRequestPB* request,
                                     ::PrefixOperationResponsePB*          response);

    grpc::Status ReclaimPrefix(grpc::ServerContext*            context,
                               const ::ReclaimPrefixRequestPB* request,
                               ::PrefixOperationResponsePB*    response);

    grpc::Status StartPrefixSend(grpc::ServerContext*                     context,
                                 const ::PrefixTransferEndpointRequestPB* request,
                                 ::PrefixOperationResponsePB*             response);

    grpc::Status StartPrefixReceive(grpc::ServerContext*                     context,
                                    const ::PrefixTransferEndpointRequestPB* request,
                                    ::PrefixOperationResponsePB*             response);

    grpc::Status CancelIfReceived(grpc::ServerContext*               context,
                                  const ::CancelIfReceivedRequestPB* request,
                                  ::CancelIfReceivedResponsePB*      response);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response);

    grpc::Status CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response);

    grpc::Status UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateEplbConfig(grpc::ServerContext* context, const UpdateEplbConfigRequestPB* request, EmptyPB* response);

    grpc::Status SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response);

    grpc::Status StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response);

    grpc::Status
    StartProfileInternal(grpc::ServerContext* context, const StartProfileInternalRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateSchedulerInfo(grpc::ServerContext* context, const UpdateSchedulerInfoRequestPB* request, EmptyPB* response);

    KVCacheInfo getCacheStatusInfo(int64_t latest_cache_version, bool need_cache_keys);

    WorkerStatusInfo getWorkerStatusInfo(int64_t latest_finished_version);

    std::shared_ptr<EngineBase> getEngine() const {
        return engine_;
    }
    std::shared_ptr<MultimodalProcessor> getMultimodalProcessor() const {
        return mm_processor_;
    }

    int64_t tpSize() const {
        return maga_init_params_.parallelism_config.tp_size;
    }

    virtual size_t onflightRequestNum();

    void stop() {
        (void)engine_->stop();
    }

    virtual EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finised_version);

    void reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_lb_time_us);

    void reportCacheStatusTime(int64_t request_begin_time_us);

    ::grpc::Status
    ExecuteFunction(::grpc::ServerContext* context, const ::FunctionRequestPB* request, ::FunctionResponsePB* response);

public:
    typedef grpc::internal::WriterInterface<GenerateOutputsPB> WriterInterface;

protected:
    static grpc::Status validatePrefixController(const PrefixControllerPB&            controller,
                                                 const std::string&                   epoch,
                                                 const PrefixInventoryReport&         inventory,
                                                 const PrefixCacheTransactionManager* manager);
    static grpc::Status
    validatePrefixTransfer(const PrefixTransferPB& transfer, const PrefixInventoryReport& inventory, bool source);
    static std::string  prefixTransferKey(const PrefixControllerPB& controller,
                                          const PrefixTransferPB&   transfer,
                                          uint64_t                  lease_id,
                                          uint64_t                  operation_id,
                                          CacheKeyType              key);
    static void         makePrefixExportResponse(const PrefixCacheTransactionManager::ExportLease& lease,
                                                 const std::string&                                epoch,
                                                 PreparePrefixExportResponsePB*                    response);
    static void         makePrefixImportResponse(const PrefixCacheTransactionManager::ImportOperation& operation,
                                                 const std::string&                                    epoch,
                                                 PreparePrefixImportResponsePB*                        response);
    static grpc::Status validatePrefixCommit(const PrefixCacheTransactionManager::ImportOperation& operation,
                                             const CommitPrefixImportRequestPB&                    request,
                                             const std::string&                                    current_epoch,
                                             uint64_t                                              current_version,
                                             bool pinned_path_valid = false);

    static grpc::Status makePrefixOperationResponse(const PrefixCacheTransactionManager::ImportOperation& operation,
                                                    const std::string&            requested_epoch,
                                                    const std::string&            current_epoch,
                                                    GetPrefixOperationResponsePB* response);

    grpc::Status serializeErrorMsg(const std::string& request_key, ErrorInfo error_info);
    grpc::Status pollStreamOutput(grpc::ServerContext*             context,
                                  const std::string&               request_key,
                                  WriterInterface*                 writer,
                                  std::shared_ptr<GenerateStream>& stream);

    // Shared helpers for single and batch paths
    ErrorInfo prepareInput(const GenerateInputPB& input_pb, std::shared_ptr<GenerateInput>& output);
    ErrorInfo collectStreamOutput(grpc::ServerContext*                  context,
                                  std::shared_ptr<GenerateStream>&      stream,
                                  const std::shared_ptr<GenerateInput>& input,
                                  GenerateOutputs&                      last_outputs);

protected:
    std::shared_ptr<EngineBase>           engine_;
    std::shared_ptr<MultimodalProcessor>  mm_processor_;
    EngineInitParams                      maga_init_params_;
    ProposeModelEngineInitParams*         propose_maga_init_params_;
    kmonitor::MetricsReporterPtr          metrics_reporter_;
    std::atomic<size_t>                   onflight_requests_{0};
    std::shared_ptr<RpcServerRuntimeMeta> meta_;
    py::object                            weight_manager_;
    std::shared_ptr<BroadcastManager>     profile_broadcaster_;
    std::mutex                            topology_switch_coordinator_mutex_;
    transfer::PrefixReceiveRegistry       prefix_receives_;
};

}  // namespace rtp_llm
