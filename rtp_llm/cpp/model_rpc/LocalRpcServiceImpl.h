#pragma once

#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {

class LocalRpcServiceImpl: public RpcService::Service {
public:
    LocalRpcServiceImpl() {}
    virtual ~LocalRpcServiceImpl() {}
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              py::object                                             mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
        local_server_ = std::make_shared<LocalRpcServer>();
        return local_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             weight_manager) {
        local_server_ = std::make_shared<LocalRpcServer>();
        return local_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return local_server_->GenerateStreamCall(context, request, writer);
    }

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response) override {
        return local_server_->BatchGenerateCall(context, request, response);
    }

    ::grpc::Status
    GetWorkerStatus(::grpc::ServerContext* context, const StatusVersionPB* request, WorkerStatusPB* response) override {
        return local_server_->GetWorkerStatus(context, request, response);
    }

    ::grpc::Status
    UpdateWeights(::grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) override {
        return local_server_->UpdateWeights(context, request, response);
    }

    ::grpc::Status
    GetCacheStatus(::grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) override {
        return local_server_->GetCacheStatus(context, request, response);
    }

    ::grpc::Status ActivatePrefixController(::grpc::ServerContext*                   context,
                                            const ActivatePrefixControllerRequestPB* request,
                                            ActivatePrefixControllerResponsePB*      response) override {
        return local_server_->ActivatePrefixController(context, request, response);
    }

    ::grpc::Status GetPrefixPrepare(::grpc::ServerContext*           context,
                                    const GetPrefixPrepareRequestPB* request,
                                    GetPrefixPrepareResponsePB*      response) override {
        return local_server_->GetPrefixPrepare(context, request, response);
    }

    ::grpc::Status GetPrefixCacheCapabilities(::grpc::ServerContext*                  context,
                                              const PrefixCacheCapabilitiesRequestPB* request,
                                              PrefixCacheCapabilitiesResponsePB*      response) override {
        return local_server_->GetPrefixCacheCapabilities(context, request, response);
    }

    ::grpc::Status InspectPrefix(::grpc::ServerContext*        context,
                                 const InspectPrefixRequestPB* request,
                                 InspectPrefixResponsePB*      response) override {
        return local_server_->InspectPrefix(context, request, response);
    }

    ::grpc::Status PreparePrefixExport(::grpc::ServerContext*              context,
                                       const PreparePrefixExportRequestPB* request,
                                       PreparePrefixExportResponsePB*      response) override {
        return local_server_->PreparePrefixExport(context, request, response);
    }

    ::grpc::Status PreparePrefixImport(::grpc::ServerContext*              context,
                                       const PreparePrefixImportRequestPB* request,
                                       PreparePrefixImportResponsePB*      response) override {
        return local_server_->PreparePrefixImport(context, request, response);
    }

    ::grpc::Status CommitPrefixImport(::grpc::ServerContext*             context,
                                      const CommitPrefixImportRequestPB* request,
                                      PrefixOperationResponsePB*         response) override {
        return local_server_->CommitPrefixImport(context, request, response);
    }

    ::grpc::Status AbortPrefixOperation(::grpc::ServerContext*               context,
                                        const AbortPrefixOperationRequestPB* request,
                                        PrefixOperationResponsePB*           response) override {
        return local_server_->AbortPrefixOperation(context, request, response);
    }

    ::grpc::Status ReleasePrefixExport(::grpc::ServerContext*              context,
                                       const ReleasePrefixExportRequestPB* request,
                                       PrefixOperationResponsePB*          response) override {
        return local_server_->ReleasePrefixExport(context, request, response);
    }

    ::grpc::Status GetPrefixOperation(::grpc::ServerContext*             context,
                                      const GetPrefixOperationRequestPB* request,
                                      GetPrefixOperationResponsePB*      response) override {
        return local_server_->GetPrefixOperation(context, request, response);
    }

    ::grpc::Status MarkPrefixRankReady(::grpc::ServerContext*              context,
                                       const MarkPrefixRankReadyRequestPB* request,
                                       PrefixOperationResponsePB*          response) override {
        return local_server_->MarkPrefixRankReady(context, request, response);
    }

    ::grpc::Status ReclaimPrefix(::grpc::ServerContext*        context,
                                 const ReclaimPrefixRequestPB* request,
                                 PrefixOperationResponsePB*    response) override {
        return local_server_->ReclaimPrefix(context, request, response);
    }

    ::grpc::Status StartPrefixSend(::grpc::ServerContext*                 context,
                                   const PrefixTransferEndpointRequestPB* request,
                                   PrefixOperationResponsePB*             response) override {
        return local_server_->StartPrefixSend(context, request, response);
    }

    ::grpc::Status StartPrefixReceive(::grpc::ServerContext*                 context,
                                      const PrefixTransferEndpointRequestPB* request,
                                      PrefixOperationResponsePB*             response) override {
        return local_server_->StartPrefixReceive(context, request, response);
    }

    ::grpc::Status CancelIfReceived(::grpc::ServerContext*           context,
                                    const CancelIfReceivedRequestPB* request,
                                    CancelIfReceivedResponsePB*      response) override {
        return local_server_->CancelIfReceived(context, request, response);
    }

    ::grpc::Status UpdateSchedulerInfo(::grpc::ServerContext*              context,
                                       const UpdateSchedulerInfoRequestPB* request,
                                       EmptyPB*                            response) override {
        return local_server_->UpdateSchedulerInfo(context, request, response);
    }

    ::grpc::Status
    SetLogLevel(::grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) override {
        return local_server_->SetLogLevel(context, request, response);
    }

    ::grpc::Status
    StartProfile(::grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) override {
        return local_server_->StartProfile(context, request, response);
    }

    ::grpc::Status StartProfileInternal(::grpc::ServerContext*               context,
                                        const StartProfileInternalRequestPB* request,
                                        EmptyPB*                             response) override {
        return local_server_->StartProfileInternal(context, request, response);
    }

    ::grpc::Status
    CheckHealth(::grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) override {
        return local_server_->CheckHealth(context, request, response);
    }

    ::grpc::Status UpdateEplbConfig(::grpc::ServerContext*           context,
                                    const UpdateEplbConfigRequestPB* request,
                                    EmptyPB*                         response) override {
        return local_server_->UpdateEplbConfig(context, request, response);
    }

    ::grpc::Status SetPause(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        return local_server_->SetPause(context, request, response);
    }

    ::grpc::Status SetRestart(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        return local_server_->SetRestart(context, request, response);
    }

    WorkerStatusInfo getWorkerStatusInfo(int64_t latest_finished_version) {
        return local_server_->getWorkerStatusInfo(latest_finished_version);
    }

    KVCacheInfo getCacheStatusInfo(int64_t latest_cache_version, bool need_cache_keys) {
        return local_server_->getCacheStatusInfo(latest_cache_version, need_cache_keys);
    }

    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finised_version) {
        return local_server_->getEngineScheduleInfo(latest_finised_version);
    }

    std::shared_ptr<EngineBase> getEngine() const {
        return local_server_->getEngine();
    };

    std::shared_ptr<MultimodalProcessor> getMultimodalProcessor() const {
        return local_server_->getMultimodalProcessor();
    };

    virtual size_t onflightRequestNum() {
        return local_server_->onflightRequestNum();
    }

    virtual void stop() {
        if (local_server_) {
            local_server_->stop();
        }
    }

    ::grpc::Status ExecuteFunction(::grpc::ServerContext*     context,
                                   const ::FunctionRequestPB* request,
                                   ::FunctionResponsePB*      response) override {
        return local_server_->ExecuteFunction(context, request, response);
    }

protected:
    std::shared_ptr<LocalRpcServer> local_server_;
};

typedef LocalRpcServiceImpl RpcServiceImpl;

}  // namespace rtp_llm
