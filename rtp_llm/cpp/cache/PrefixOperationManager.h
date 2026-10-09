#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class PrefixOperationManager {
public:
    enum class State { PREPARED, COPYING, DATA_READY, COMMITTED, ABORTED };

    struct Operation {
        uint64_t          id = 0;
        uint64_t          epoch = 0;
        uint64_t          expected_version = 0;
        State             state = State::PREPARED;
        std::vector<CacheKeyType> keys;
        BlockIndicesType  source_blocks;
        BlockIndicesType  destination_blocks;
    };

    explicit PrefixOperationManager(uint64_t epoch = 1): epoch_(epoch) {}

    uint64_t epoch() const {
        std::lock_guard<std::mutex> lock(mu_);
        return epoch_;
    }

    void fence(uint64_t new_epoch) {
        std::lock_guard<std::mutex> lock(mu_);
        if (new_epoch <= epoch_) {
            return;
        }
        epoch_ = new_epoch;
        for (auto& [id, operation] : operations_) {
            if (operation.state != State::COMMITTED && operation.state != State::ABORTED) {
                operation.state = State::ABORTED;
            }
        }
    }

    bool prepare(uint64_t id,
                 uint64_t operation_epoch,
                 uint64_t expected_version,
                 std::vector<CacheKeyType> keys,
                 BlockIndicesType source_blocks,
                 BlockIndicesType destination_blocks) {
        std::lock_guard<std::mutex> lock(mu_);
        if (operation_epoch != epoch_ || id == 0 || keys.empty() || source_blocks.size() != keys.size()
            || destination_blocks.size() != keys.size()) {
            return false;
        }
        auto existing = operations_.find(id);
        if (existing != operations_.end()) {
            return same(existing->second, operation_epoch, expected_version, keys, source_blocks, destination_blocks);
        }
        operations_.emplace(id,
                            Operation{id,
                                      operation_epoch,
                                      expected_version,
                                      State::PREPARED,
                                      std::move(keys),
                                      std::move(source_blocks),
                                      std::move(destination_blocks)});
        return true;
    }

    bool beginCopy(uint64_t id) {
        return transition(id, State::PREPARED, State::COPYING);
    }

    bool markDataReady(uint64_t id) {
        return transition(id, State::COPYING, State::DATA_READY);
    }

    bool commit(uint64_t id, uint64_t operation_epoch, uint64_t current_version) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = operations_.find(id);
        if (it == operations_.end() || it->second.epoch != operation_epoch || operation_epoch != epoch_
            || it->second.expected_version != current_version) {
            return false;
        }
        if (it->second.state == State::COMMITTED) {
            return true;
        }
        if (it->second.state != State::DATA_READY) {
            return false;
        }
        it->second.state = State::COMMITTED;
        return true;
    }

    bool abort(uint64_t id, uint64_t operation_epoch) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = operations_.find(id);
        if (it == operations_.end() || it->second.epoch != operation_epoch) {
            return false;
        }
        if (it->second.state == State::ABORTED) {
            return true;
        }
        if (it->second.state == State::COMMITTED) {
            return false;
        }
        it->second.state = State::ABORTED;
        return true;
    }

    Operation get(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = operations_.find(id);
        return it == operations_.end() ? Operation{} : it->second;
    }

private:
    static bool same(const Operation& operation,
                     uint64_t         epoch,
                     uint64_t         expected_version,
                     const std::vector<CacheKeyType>& keys,
                     const BlockIndicesType& source_blocks,
                     const BlockIndicesType& destination_blocks) {
        return operation.epoch == epoch && operation.expected_version == expected_version
               && operation.keys == keys && operation.source_blocks == source_blocks
               && operation.destination_blocks == destination_blocks;
    }

    bool transition(uint64_t id, State from, State to) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = operations_.find(id);
        if (it == operations_.end() || it->second.state != from) {
            return false;
        }
        it->second.state = to;
        return true;
    }

    mutable std::mutex               mu_;
    uint64_t                         epoch_ = 1;
    std::unordered_map<uint64_t, Operation> operations_;
};

}  // namespace rtp_llm
