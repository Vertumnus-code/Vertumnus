#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"

#include <cstdint>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace transfer {

class PrefixReceiveHandle {
public:
    ~PrefixReceiveHandle();
    PrefixReceiveHandle(const PrefixReceiveHandle&)            = delete;
    PrefixReceiveHandle& operator=(const PrefixReceiveHandle&) = delete;

    bool done() const;
    bool success() const;
    bool failed() const;
    bool expired() const;
    bool wait();
    void cancel();

private:
    friend class PrefixTransferAdapter;
    PrefixReceiveHandle(IKVCacheReceiverPtr receiver, int64_t timeout_ms, std::shared_ptr<void> resource_lease);
    void removeRegistrations();

    IKVCacheReceiverPtr                                      receiver_;
    std::vector<std::pair<std::string, IKVCacheRecvTaskPtr>> tasks_;
    const std::chrono::steady_clock::time_point              deadline_;
    std::atomic<bool>                                        cancelled_{false};
    std::once_flag                                           registrations_removed_;
    std::shared_ptr<void>                                    resource_lease_;
};

class PrefixTransferAdapter {
public:
    PrefixTransferAdapter(IKVCacheSenderPtr sender, IKVCacheReceiverPtr receiver);

    bool transfer(const std::vector<RecvRequest>& receives,
                  const std::vector<SendRequest>& sends,
                  int64_t                         timeout_ms,
                  std::shared_ptr<void>           resource_lease = nullptr);

    bool
    send(const std::vector<SendRequest>& sends, int64_t timeout_ms, std::shared_ptr<void> resource_lease = nullptr);
    bool                                 receive(const std::vector<RecvRequest>& receives,
                                                 int64_t                         timeout_ms,
                                                 std::shared_ptr<void>           resource_lease = nullptr);
    std::shared_ptr<PrefixReceiveHandle> prepareReceive(const std::vector<RecvRequest>& receives,
                                                        int64_t                         timeout_ms,
                                                        std::shared_ptr<void>           resource_lease = nullptr);

private:
    IKVCacheSenderPtr   sender_;
    IKVCacheReceiverPtr receiver_;
};

}  // namespace transfer
}  // namespace rtp_llm
