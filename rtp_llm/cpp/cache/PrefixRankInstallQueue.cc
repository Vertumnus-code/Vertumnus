#include "rtp_llm/cpp/cache/PrefixRankInstallQueue.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

PrefixRankInstallQueue::PrefixRankInstallQueue(PrefixCacheTransactionManager& manager,
                                               size_t                         max_pending,
                                               size_t                         max_blocks_per_step,
                                               size_t                         max_bytes_per_step,
                                               size_t                         max_chunks_per_boundary):
    manager_(manager),
    max_pending_(max_pending),
    max_blocks_per_step_(max_blocks_per_step),
    max_bytes_per_step_(max_bytes_per_step),
    max_chunks_per_boundary_(max_chunks_per_boundary) {
    if (max_pending == 0 || max_blocks_per_step == 0 || max_blocks_per_step > 4096 || max_bytes_per_step == 0
        || max_chunks_per_boundary == 0 || max_chunks_per_boundary > 128) {
        throw std::invalid_argument("invalid prefix rank install queue limits");
    }
}

PrefixRankInstallQueue::~PrefixRankInstallQueue() {
    close();
}

bool PrefixRankInstallQueue::current(const Entry& entry) const {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return false;
        }
    }
    const auto operation = manager_.getImport(entry.operation.id);
    return manager_.isControllerCurrent(entry.operation.controller) && operation.epoch == entry.operation.epoch
           && operation.expected_version == entry.operation.expected_version
           && operation.controller == entry.operation.controller && operation.prepare == entry.operation.prepare
           && operation.rank_count == entry.operation.rank_count
           && (operation.state == PrefixCacheTransactionManager::State::COPYING
               || operation.state == PrefixCacheTransactionManager::State::DATA_READY)
           && manager_.importSnapshotCurrent(entry.operation.id);
}

std::shared_future<bool> PrefixRankInstallQueue::enqueue(uint64_t                        id,
                                                         uint64_t                        epoch,
                                                         uint64_t                        version,
                                                         const PrefixControllerIdentity& controller) {
    auto entry       = std::make_shared<Entry>();
    entry->operation = manager_.getImport(id);
    if (entry->operation.id == 0 || entry->operation.epoch != epoch || entry->operation.expected_version != version
        || !(entry->operation.controller == controller) || entry->operation.source_lease_id == 0 || !current(*entry)) {
        entry->promise.set_value(false);
        return entry->future;
    }
    auto pin = manager_.pinImport(id, epoch, controller);
    if (!pin) {
        entry->promise.set_value(false);
        return entry->future;
    }
    entry->pin = std::move(*pin);
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(id);
    if (found != entries_.end()) {
        return found->second->future;
    }
    if (closed_ || entries_.size() >= max_pending_) {
        entry->promise.set_value(false);
        return entry->future;
    }
    entries_.emplace(id, entry);
    pending_.push_back(entry);
    return entry->future;
}

bool PrefixRankInstallQueue::pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_.empty();
}

void PrefixRankInstallQueue::complete(const std::shared_ptr<Entry>& entry, bool success) {
    if (!entry) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entry->completed) {
        entry->completed = true;
        entry->promise.set_value(success && !closed_);
    }
    entries_.erase(entry->operation.id);
    pending_.erase(std::remove(pending_.begin(), pending_.end(), entry), pending_.end());
}

void PrefixRankInstallQueue::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    for (const auto& entry : pending_) {
        if (!entry->completed) {
            entry->completed = true;
            entry->promise.set_value(false);
        }
    }
    pending_.clear();
    entries_.clear();
}

void PrefixRankInstallQueue::process(
    int rank, int rank_count, bool supported, const Describe& describe, const Collective& collective) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (processing_ || active_) {
            throw std::runtime_error("prefix rank install already active or quarantined");
        }
        processing_ = true;
        if (rank == 0 && !pending_.empty()) {
            active_ = pending_.front();
        }
    }
    try {
        for (size_t chunk = 0;; ++chunk) {
            processActive(rank, rank_count, supported, describe, collective);
            bool continue_install = chunk + 1 < max_chunks_per_boundary_;
            if (rank == 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                continue_install = continue_install && !closed_ && active_ && !active_->completed;
            }
            if (!collective.all_ready(continue_install)) {
                break;
            }
        }
    } catch (...) {
        close();
        throw;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    active_.reset();
    processing_ = false;
}

bool PrefixRankInstallQueue::drained() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_ && !processing_ && !active_ && pending_.empty() && entries_.empty();
}

bool PrefixRankInstallQueue::reopen() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_ || processing_ || active_ || !pending_.empty() || !entries_.empty()) {
        return false;
    }
    closed_ = false;
    return true;
}

void PrefixRankInstallQueue::processActive(
    int rank, int rank_count, bool supported, const Describe& describe, const Collective& collective) {
    std::shared_ptr<Entry> entry;
    if (rank == 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        entry = active_;
    }
    auto                   header = torch::zeros({2}, torch::kInt64);
    BlockIndicesType       selected;
    std::vector<BlockInfo> first_buffers;
    if (entry) {
        try {
            if (!supported || entry->operation.rank_count != rank_count || !current(*entry)) {
                complete(entry, false);
                entry.reset();
            } else {
                const auto& indices     = entry->operation.missing_indices;
                size_t      block_limit = max_blocks_per_step_;
                if (entry->cursor < indices.size()) {
                    first_buffers      = describe(entry->operation.destination_blocks.at(indices[entry->cursor]));
                    size_t block_bytes = 0;
                    for (const auto& buffer : first_buffers) {
                        if (buffer.size_bytes > max_bytes_per_step_ - block_bytes) {
                            throw std::runtime_error("prefix block exceeds rank install byte limit");
                        }
                        block_bytes += buffer.size_bytes;
                    }
                    if (block_bytes == 0 || first_buffers.size() > 8192) {
                        throw std::runtime_error("invalid prefix block buffers");
                    }
                    block_limit = std::min(block_limit, max_bytes_per_step_ / block_bytes);
                }
                const auto end = std::min(indices.size(), entry->cursor + block_limit);
                for (size_t index = entry->cursor; index < end; ++index) {
                    selected.push_back(entry->operation.destination_blocks.at(indices[index]));
                }
                header.data_ptr<int64_t>()[0] = static_cast<int64_t>(selected.size());
                header.data_ptr<int64_t>()[1] = static_cast<int64_t>(first_buffers.size());
            }
        } catch (const std::exception&) {
            complete(entry, false);
            entry.reset();
            header.zero_();
        }
    }
    collective.broadcast(header);
    const auto block_count  = header.data_ptr<int64_t>()[0];
    const auto buffer_count = header.data_ptr<int64_t>()[1];
    if (block_count < 0 || block_count > 4096 || buffer_count < 0 || buffer_count > 8192) {
        throw std::runtime_error("invalid prefix rank install header");
    }
    auto metadata = torch::zeros({block_count + buffer_count}, torch::kInt64);
    if (rank == 0 && entry) {
        for (size_t index = 0; index < selected.size(); ++index) {
            metadata.data_ptr<int64_t>()[index] = selected[index];
        }
        for (size_t index = 0; index < first_buffers.size(); ++index) {
            metadata.data_ptr<int64_t>()[selected.size() + index] = first_buffers[index].size_bytes;
        }
    }
    if (metadata.numel() != 0) {
        collective.broadcast(metadata);
    }
    bool                       valid       = supported && (block_count == 0 || buffer_count > 0);
    size_t                     total_bytes = 0;
    std::vector<torch::Tensor> buffers;
    try {
        for (int64_t block = 0; valid && block < block_count; ++block) {
            const auto block_id = metadata.data_ptr<int64_t>()[block];
            if (block_id <= 0 || block_id > std::numeric_limits<BlockIdxType>::max()) {
                valid = false;
                break;
            }
            const auto descriptions = describe(static_cast<BlockIdxType>(block_id));
            if (descriptions.size() != static_cast<size_t>(buffer_count)) {
                valid = false;
                break;
            }
            for (int64_t index = 0; index < buffer_count; ++index) {
                const auto& description = descriptions[index];
                if (!description.addr || !description.is_cuda || description.size_bytes == 0
                    || description.size_bytes > max_bytes_per_step_ - total_bytes
                    || description.size_bytes
                           != static_cast<size_t>(metadata.data_ptr<int64_t>()[block_count + index])) {
                    valid = false;
                    break;
                }
                total_bytes += description.size_bytes;
                buffers.push_back(torch::from_blob(
                    description.addr,
                    {static_cast<int64_t>(description.size_bytes)},
                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, description.device_index)));
            }
        }
    } catch (const std::exception&) {
        valid = false;
    }
    if (!collective.all_ready(valid)) {
        complete(entry, false);
        return;
    }
    collective.synchronize();
    if (collective.broadcast_data && !buffers.empty()) {
        collective.broadcast_data(buffers);
    } else {
        for (auto& buffer : buffers) {
            collective.broadcast(buffer);
        }
    }
    collective.synchronize();
    if (!collective.all_ready(true)) {
        complete(entry, false);
        return;
    }
    if (!entry) {
        return;
    }
    if (!current(*entry)) {
        complete(entry, false);
        return;
    }
    entry->cursor += static_cast<size_t>(block_count);
    if (entry->cursor != entry->operation.missing_indices.size()) {
        return;
    }
    bool ready = true;
    if (manager_.getImport(entry->operation.id).state != PrefixCacheTransactionManager::State::DATA_READY) {
        for (int completed_rank = 0; completed_rank < rank_count; ++completed_rank) {
            ready = manager_.markImportRankReady(
                        entry->operation.id, entry->operation.epoch, completed_rank, entry->operation.controller)
                    && ready;
        }
    }
    complete(entry, ready && manager_.isControllerCurrent(entry->operation.controller));
}

}  // namespace rtp_llm
