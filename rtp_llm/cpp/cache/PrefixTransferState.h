#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>

namespace rtp_llm {

class PrefixTransferState {
public:
    enum class State { PREPARED, COPYING, DATA_READY, COMMITTED, ABORTED };

    PrefixTransferState(uint64_t operation_id, uint64_t epoch, std::string layout_fingerprint, int rank_count);

    bool beginCopy();
    bool markRankReady(int rank);
    bool commit();
    bool abort(uint64_t epoch);
    void fence(uint64_t epoch);

    State state() const;
    uint64_t operationId() const;
    uint64_t epoch() const;
    const std::string& layoutFingerprint() const;
    int rankCount() const;
    int readyRankCount() const;

private:
    mutable std::mutex mu_;
    uint64_t operation_id_;
    uint64_t epoch_;
    std::string layout_fingerprint_;
    int rank_count_;
    State state_ = State::PREPARED;
    std::unordered_set<int> ready_ranks_;
};

}  // namespace rtp_llm
