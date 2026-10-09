#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/PrefixTransferAdapter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {
namespace transfer {

struct PrefixTransferPair {
    std::string       unique_key;
    std::string       source_ip;
    uint32_t          source_port = 0;
    int64_t           deadline_ms = 0;
    KeyBlockInfoPtr   source;
    KeyBlockInfoPtr   destination;
};

class PrefixTransferDispatcher {
public:
    explicit PrefixTransferDispatcher(std::shared_ptr<PrefixTransferAdapter> adapter);

    bool dispatch(const std::vector<PrefixTransferPair>& pairs, int64_t timeout_ms) const;
    bool dispatchSend(const std::vector<PrefixTransferPair>& pairs,
                      int64_t timeout_ms,
                      std::shared_ptr<void> resource_lease = nullptr) const;
    bool dispatchReceive(const std::vector<PrefixTransferPair>& pairs, int64_t timeout_ms) const;
    std::shared_ptr<PrefixReceiveHandle> prepareReceive(const std::vector<PrefixTransferPair>& pairs,
                                                        int64_t timeout_ms,
                                                        std::shared_ptr<void> resource_lease = nullptr) const;

private:
    std::shared_ptr<PrefixTransferAdapter> adapter_;
};

}  // namespace transfer
}  // namespace rtp_llm
