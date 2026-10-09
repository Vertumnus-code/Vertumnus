#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferAdapter.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace rtp_llm::transfer {

class PrefixReceiveRegistry {
public:
    using Handle = std::shared_ptr<PrefixReceiveHandle>;
    using Check  = std::function<bool()>;

    PrefixReceiveRegistry();
    ~PrefixReceiveRegistry();
    PrefixReceiveRegistry(const PrefixReceiveRegistry&)            = delete;
    PrefixReceiveRegistry& operator=(const PrefixReceiveRegistry&) = delete;

    bool   store(uint64_t id, uint64_t epoch, Handle handle, Check active, Check abort);
    Handle get(uint64_t id, uint64_t epoch) const;
    bool   cancel(uint64_t id, uint64_t epoch, const Check& abort);
    void   release(uint64_t id, uint64_t epoch);

private:
    struct Entry {
        uint64_t epoch;
        Handle   handle;
        Check    active;
        Check    abort;
    };

    void run();

    mutable std::mutex                  mutex_;
    std::condition_variable             wake_;
    std::unordered_map<uint64_t, Entry> entries_;
    std::vector<Handle>                 retired_;
    bool                                stopping_ = false;
    std::thread                         worker_;
};

}  // namespace rtp_llm::transfer
