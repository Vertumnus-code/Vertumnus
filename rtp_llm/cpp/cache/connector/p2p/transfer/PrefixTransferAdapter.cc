#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferAdapter.h"

#include <thread>
#include <unordered_set>

namespace rtp_llm {
namespace transfer {
namespace {

struct SendCompletion {
    SendCompletion(size_t count, std::shared_ptr<void> lease): pending(count), resource_lease(std::move(lease)) {}
    std::atomic<size_t>   pending;
    std::atomic<bool>     failed{false};
    std::shared_ptr<void> resource_lease;
};

std::shared_ptr<SendCompletion> startSends(const IKVCacheSenderPtr&        sender,
                                           const std::vector<SendRequest>& sends,
                                           std::shared_ptr<void>           resource_lease = nullptr) {
    auto completion = std::make_shared<SendCompletion>(sends.size(), std::move(resource_lease));
    try {
        for (const auto& request : sends) {
            auto completed = std::make_shared<std::atomic<bool>>(false);
            sender->send(request, [completion, completed](TransferErrorCode code, const std::string&) {
                if (completed->exchange(true)) {
                    return;
                }
                if (code != TransferErrorCode::OK) {
                    completion->failed.store(true);
                }
                completion->pending.fetch_sub(1);
            });
        }
    } catch (...) {
        completion->failed.store(true);
    }
    return completion;
}

}  // namespace

PrefixReceiveHandle::PrefixReceiveHandle(IKVCacheReceiverPtr   receiver,
                                         int64_t               timeout_ms,
                                         std::shared_ptr<void> resource_lease):
    receiver_(std::move(receiver)),
    deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms)),
    resource_lease_(std::move(resource_lease)) {}

PrefixReceiveHandle::~PrefixReceiveHandle() {
    cancel();
    while (!done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool PrefixReceiveHandle::done() const {
    for (const auto& entry : tasks_) {
        if (!entry.second->done()) {
            return false;
        }
    }
    return true;
}

bool PrefixReceiveHandle::failed() const {
    if (cancelled_.load()) {
        return true;
    }
    for (const auto& entry : tasks_) {
        if (entry.second->done() && !entry.second->success()) {
            return true;
        }
    }
    return false;
}

bool PrefixReceiveHandle::success() const {
    return done() && !failed();
}

bool PrefixReceiveHandle::expired() const {
    return std::chrono::steady_clock::now() >= deadline_;
}

void PrefixReceiveHandle::removeRegistrations() {
    std::call_once(registrations_removed_, [this] {
        for (const auto& entry : tasks_) {
            if (receiver_->getTask(entry.first) == entry.second) {
                receiver_->stealTask(entry.first);
            }
        }
    });
}

void PrefixReceiveHandle::cancel() {
    if (cancelled_.exchange(true)) {
        return;
    }
    for (const auto& entry : tasks_) {
        entry.second->cancel();
    }
    removeRegistrations();
}

bool PrefixReceiveHandle::wait() {
    while (std::chrono::steady_clock::now() < deadline_) {
        if (failed()) {
            break;
        }
        if (done()) {
            removeRegistrations();
            return success();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cancel();
    return false;
}

PrefixTransferAdapter::PrefixTransferAdapter(IKVCacheSenderPtr sender, IKVCacheReceiverPtr receiver):
    sender_(std::move(sender)), receiver_(std::move(receiver)) {}

bool PrefixTransferAdapter::send(const std::vector<SendRequest>& sends,
                                 int64_t                         timeout_ms,
                                 std::shared_ptr<void>           resource_lease) {
    if (!sender_ || sends.empty() || timeout_ms <= 0) {
        return false;
    }
    const auto deadline   = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto       completion = startSends(sender_, sends, std::move(resource_lease));
    while (std::chrono::steady_clock::now() < deadline) {
        if (completion->failed.load()) {
            return false;
        }
        if (completion->pending.load() == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

std::shared_ptr<PrefixReceiveHandle> PrefixTransferAdapter::prepareReceive(const std::vector<RecvRequest>& receives,
                                                                           int64_t                         timeout_ms,
                                                                           std::shared_ptr<void> resource_lease) {
    if (!receiver_ || receives.empty() || timeout_ms <= 0) {
        return nullptr;
    }
    std::unordered_set<std::string> keys;
    for (const auto& request : receives) {
        if (request.unique_key.empty() || !keys.emplace(request.unique_key).second) {
            return nullptr;
        }
    }
    auto handle =
        std::shared_ptr<PrefixReceiveHandle>(new PrefixReceiveHandle(receiver_, timeout_ms, std::move(resource_lease)));
    handle->tasks_.reserve(receives.size());
    try {
        for (const auto& request : receives) {
            auto task = receiver_->recv(request);
            if (!task) {
                return nullptr;
            }
            handle->tasks_.emplace_back(request.unique_key, std::move(task));
        }
    } catch (...) {
        return nullptr;
    }
    return handle;
}

bool PrefixTransferAdapter::receive(const std::vector<RecvRequest>& receives,
                                    int64_t                         timeout_ms,
                                    std::shared_ptr<void>           resource_lease) {
    auto handle = prepareReceive(receives, timeout_ms, std::move(resource_lease));
    return handle && handle->wait();
}

bool PrefixTransferAdapter::transfer(const std::vector<RecvRequest>& receives,
                                     const std::vector<SendRequest>& sends,
                                     int64_t                         timeout_ms,
                                     std::shared_ptr<void>           resource_lease) {
    if (!sender_ || receives.size() != sends.size()) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto       handle   = prepareReceive(receives, timeout_ms, resource_lease);
    if (!handle) {
        return false;
    }
    auto completion = startSends(sender_, sends, std::move(resource_lease));
    while (std::chrono::steady_clock::now() < deadline) {
        if (completion->failed.load() || handle->failed()) {
            break;
        }
        if (completion->pending.load() == 0 && handle->success()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    handle->cancel();
    return false;
}

}  // namespace transfer
}  // namespace rtp_llm
