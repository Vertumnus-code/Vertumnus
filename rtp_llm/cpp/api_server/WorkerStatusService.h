#pragma once

#include <atomic>
#include <string>

#include "rtp_llm/cpp/engine_base/EngineBase.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

class WorkerStatusResponse: public autil::legacy::Jsonizable {
public:
    ~WorkerStatusResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("available_kv_cache", cache_status.available_kv_cache);
        json.Jsonize("total_kv_cache", cache_status.total_kv_cache);
        json.Jsonize("alive", alive);
        json.Jsonize("profile_name", profile_name);
        json.Jsonize("dp_size", dp_size);
        json.Jsonize("tp_size", tp_size);
        json.Jsonize("cp_size", cp_size);
        json.Jsonize("world_size", world_size);
        json.Jsonize("local_world_size", local_world_size);
        json.Jsonize("cp_rotate_method", cp_rotate_method);
    }

public:
    int         available_concurrency;
    KVCacheInfo cache_status;
    bool        alive;
    std::string profile_name;
    int         dp_size{-1};
    int         tp_size{-1};
    int         cp_size{-1};
    int         world_size{-1};
    int         local_world_size{-1};
    std::string cp_rotate_method;
};

class WorkerStatusService {
public:
    WorkerStatusService(const std::shared_ptr<EngineBase>&            engine,
                        const std::shared_ptr<ConcurrencyController>& controller);
    ~WorkerStatusService() = default;

public:
    void workerStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                      const http_server::HttpRequest&                         request);
    void stop();

private:
    std::atomic_bool                       is_stopped_{false};
    std::shared_ptr<EngineBase>            engine_;
    std::shared_ptr<ConcurrencyController> controller_;
};

}  // namespace rtp_llm
