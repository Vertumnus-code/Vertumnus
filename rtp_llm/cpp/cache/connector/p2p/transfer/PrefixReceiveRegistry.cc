#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixReceiveRegistry.h"

#include <algorithm>
#include <iterator>

namespace rtp_llm::transfer {

PrefixReceiveRegistry::PrefixReceiveRegistry(): worker_([this] { run(); }) {}

PrefixReceiveRegistry::~PrefixReceiveRegistry() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    worker_.join();
}

bool PrefixReceiveRegistry::store(uint64_t id, uint64_t epoch, Handle handle, Check active, Check abort) {
    if (!handle || !active || !abort) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || entries_.count(id) != 0 || handle->expired() || !active()) {
        return false;
    }
    entries_.emplace(id, Entry{epoch, std::move(handle), std::move(active), std::move(abort)});
    wake_.notify_one();
    return true;
}

PrefixReceiveRegistry::Handle PrefixReceiveRegistry::get(uint64_t id, uint64_t epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(id);
    return found != entries_.end() && found->second.epoch == epoch ? found->second.handle : nullptr;
}

bool PrefixReceiveRegistry::cancel(uint64_t id, uint64_t epoch, const Check& abort) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(id);
    if (!abort || (found != entries_.end() && found->second.epoch != epoch) || !abort()) {
        return false;
    }
    if (found != entries_.end()) {
        retired_.push_back(std::move(found->second.handle));
        entries_.erase(found);
        wake_.notify_one();
    }
    return true;
}

void PrefixReceiveRegistry::release(uint64_t id, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(id);
    if (found != entries_.end() && found->second.epoch == epoch) {
        retired_.push_back(std::move(found->second.handle));
        entries_.erase(found);
        wake_.notify_one();
    }
}

void PrefixReceiveRegistry::run() {
    std::vector<Handle> draining;
    for (;;) {
        bool stopping;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait_for(lock, std::chrono::milliseconds(10));
            stopping = stopping_;
            for (auto entry = entries_.begin(); entry != entries_.end();) {
                if (stopping || entry->second.handle->expired() || !entry->second.active()) {
                    entry->second.abort();
                    retired_.push_back(std::move(entry->second.handle));
                    entry = entries_.erase(entry);
                } else {
                    ++entry;
                }
            }
            std::move(retired_.begin(), retired_.end(), std::back_inserter(draining));
            retired_.clear();
        }
        for (const auto& handle : draining) {
            if (!handle->done()) {
                handle->cancel();
            }
        }
        draining.erase(
            std::remove_if(draining.begin(), draining.end(), [](const Handle& handle) { return handle->done(); }),
            draining.end());
        if (stopping && draining.empty()) {
            return;
        }
    }
}

}  // namespace rtp_llm::transfer
