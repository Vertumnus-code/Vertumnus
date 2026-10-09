#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferDispatcher.h"

#include <unordered_set>

namespace rtp_llm {
namespace transfer {

PrefixTransferDispatcher::PrefixTransferDispatcher(std::shared_ptr<PrefixTransferAdapter> adapter):
    adapter_(std::move(adapter)) {}

bool PrefixTransferDispatcher::dispatch(const std::vector<PrefixTransferPair>& pairs, int64_t timeout_ms) const {
    if (!adapter_ || pairs.empty() || timeout_ms <= 0) {
        return false;
    }
    std::vector<RecvRequest> receives;
    std::vector<SendRequest> sends;
    receives.reserve(pairs.size());
    sends.reserve(pairs.size());
    std::unordered_set<std::string> unique_keys;
    for (const auto& pair : pairs) {
        if (pair.unique_key.empty() || pair.source_ip.empty() || pair.source_port == 0 || !pair.source || !pair.destination
            || pair.source->cache_key != pair.destination->cache_key || !unique_keys.emplace(pair.unique_key).second) {
            return false;
        }
        RecvRequest receive{pair.unique_key, {{pair.destination->cache_key, pair.destination}}, pair.deadline_ms};
        SendRequest send{pair.source_ip, pair.source_port, pair.unique_key, {{pair.source->cache_key, pair.source}}, pair.deadline_ms};
        receives.push_back(std::move(receive));
        sends.push_back(std::move(send));
    }
    return adapter_->transfer(receives, sends, timeout_ms);
}

bool PrefixTransferDispatcher::dispatchSend(const std::vector<PrefixTransferPair>& pairs,
                                             int64_t timeout_ms,
                                             std::shared_ptr<void> resource_lease) const {
    if (!adapter_ || pairs.empty()) {
        return false;
    }
    std::vector<SendRequest> sends;
    sends.reserve(pairs.size());
    for (const auto& pair : pairs) {
        if (pair.unique_key.empty() || pair.source_ip.empty() || pair.source_port == 0 || !pair.source) {
            return false;
        }
        sends.push_back(
            {pair.source_ip, pair.source_port, pair.unique_key, {{pair.source->cache_key, pair.source}}, pair.deadline_ms});
    }
    return adapter_->send(sends, timeout_ms, std::move(resource_lease));
}

bool PrefixTransferDispatcher::dispatchReceive(const std::vector<PrefixTransferPair>& pairs,
                                                int64_t                          timeout_ms) const {
    if (!adapter_ || pairs.empty()) {
        return false;
    }
    std::vector<RecvRequest> receives;
    receives.reserve(pairs.size());
    for (const auto& pair : pairs) {
        if (pair.unique_key.empty() || !pair.destination) {
            return false;
        }
        receives.push_back({pair.unique_key, {{pair.destination->cache_key, pair.destination}}, pair.deadline_ms});
    }
    return adapter_->receive(receives, timeout_ms);
}

std::shared_ptr<PrefixReceiveHandle> PrefixTransferDispatcher::prepareReceive(
    const std::vector<PrefixTransferPair>& pairs, int64_t timeout_ms, std::shared_ptr<void> resource_lease) const {
    if (!adapter_ || pairs.empty()) {
        return nullptr;
    }
    std::vector<RecvRequest> receives;
    receives.reserve(pairs.size());
    for (const auto& pair : pairs) {
        if (pair.unique_key.empty() || !pair.destination) {
            return nullptr;
        }
        receives.push_back({pair.unique_key, {{pair.destination->cache_key, pair.destination}}, pair.deadline_ms});
    }
    return adapter_->prepareReceive(receives, timeout_ms, std::move(resource_lease));
}

}  // namespace transfer
}  // namespace rtp_llm
