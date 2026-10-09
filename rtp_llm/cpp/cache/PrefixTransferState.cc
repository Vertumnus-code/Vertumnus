#include "rtp_llm/cpp/cache/PrefixTransferState.h"

namespace rtp_llm {

PrefixTransferState::PrefixTransferState(
    uint64_t operation_id, uint64_t epoch, std::string layout_fingerprint, int rank_count):
    operation_id_(operation_id),
    epoch_(epoch),
    layout_fingerprint_(std::move(layout_fingerprint)),
    rank_count_(rank_count) {}

bool PrefixTransferState::beginCopy() {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ == State::COPYING) {
        return true;
    }
    if (state_ != State::PREPARED || rank_count_ <= 0) {
        return false;
    }
    state_ = State::COPYING;
    return true;
}

bool PrefixTransferState::markRankReady(int rank) {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != State::COPYING || rank < 0 || rank >= rank_count_) {
        return false;
    }
    ready_ranks_.insert(rank);
    if (static_cast<int>(ready_ranks_.size()) == rank_count_) {
        state_ = State::DATA_READY;
    }
    return true;
}

bool PrefixTransferState::commit() {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ == State::COMMITTED) {
        return true;
    }
    if (state_ != State::DATA_READY) {
        return false;
    }
    state_ = State::COMMITTED;
    return true;
}

bool PrefixTransferState::abort(uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mu_);
    if (epoch != epoch_) {
        return false;
    }
    if (state_ == State::ABORTED) {
        return true;
    }
    if (state_ == State::COMMITTED) {
        return false;
    }
    state_ = State::ABORTED;
    return true;
}

void PrefixTransferState::fence(uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mu_);
    if (epoch != epoch_) {
        state_ = State::ABORTED;
    }
}

PrefixTransferState::State PrefixTransferState::state() const {
    std::lock_guard<std::mutex> lock(mu_);
    return state_;
}

uint64_t PrefixTransferState::operationId() const {
    return operation_id_;
}

uint64_t PrefixTransferState::epoch() const {
    return epoch_;
}

const std::string& PrefixTransferState::layoutFingerprint() const {
    return layout_fingerprint_;
}

int PrefixTransferState::rankCount() const {
    return rank_count_;
}

int PrefixTransferState::readyRankCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(ready_ranks_.size());
}

}  // namespace rtp_llm
