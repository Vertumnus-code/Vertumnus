#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"

#include <unordered_map>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <openssl/sha.h>

namespace rtp_llm {

BlockIndicesType PrefixCacheTransactionManager::ImportOperation::stagingBlocks() const {
    BlockIndicesType blocks;
    blocks.reserve(missing_indices.size());
    for (const auto index : missing_indices) {
        blocks.push_back(destination_blocks.at(index));
    }
    return blocks;
}

bool PrefixTransferIdentity::valid() const {
    if (transfer_id.size() != 36) {
        return false;
    }
    for (size_t index = 0; index < transfer_id.size(); ++index) {
        const char value = transfer_id[index];
        if (index == 8 || index == 13 || index == 18 || index == 23) {
            if (value != '-') {
                return false;
            }
        } else if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return !source_worker_id.empty() && source_worker_id.size() <= 512 && !destination_worker_id.empty()
           && destination_worker_id.size() <= 512 && !source_epoch.empty() && source_epoch.size() <= 512
           && !destination_epoch.empty() && destination_epoch.size() <= 512 && !source_layout.empty()
           && source_layout.size() <= 65536 && !destination_layout.empty() && destination_layout.size() <= 65536;
}

bool PrefixTransferIdentity::empty() const {
    return transfer_id.empty() && source_worker_id.empty() && destination_worker_id.empty() && source_epoch.empty()
           && destination_epoch.empty() && source_layout.empty() && destination_layout.empty();
}

bool PrefixTransferIdentity::operator==(const PrefixTransferIdentity& other) const {
    return transfer_id == other.transfer_id && source_worker_id == other.source_worker_id
           && destination_worker_id == other.destination_worker_id && source_epoch == other.source_epoch
           && destination_epoch == other.destination_epoch && source_layout == other.source_layout
           && destination_layout == other.destination_layout;
}

std::string PrefixTransferIdentity::wirePrefix(const PrefixControllerIdentity& controller,
                                               uint64_t                        source_lease_id,
                                               uint64_t                        destination_operation_id) const {
    if (!valid() || !controller.valid() || source_lease_id == 0 || destination_operation_id == 0) {
        throw std::invalid_argument("invalid prefix transfer identity");
    }
    std::string canonical = "prefix-transfer-v2";
    for (const auto& field : {controller.namespace_id,
                              controller.controller_id,
                              std::to_string(controller.term),
                              transfer_id,
                              source_worker_id,
                              destination_worker_id,
                              source_epoch,
                              destination_epoch,
                              source_layout,
                              destination_layout,
                              std::to_string(source_lease_id),
                              std::to_string(destination_operation_id)}) {
        canonical += std::to_string(field.size()) + ":" + field;
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(canonical.data()), canonical.size(), digest);
    constexpr char hex[]  = "0123456789abcdef";
    std::string    result = "prefix-v2:";
    for (const auto value : digest) {
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

PrefixCacheTransactionManager::PrefixCacheTransactionManager(BlockPoolPtr  block_pool,
                                                             BlockCachePtr block_cache,
                                                             CopyBlock     copy_block):
    PrefixCacheTransactionManager(
        std::move(block_pool), std::move(block_cache), std::move(copy_block), LifecycleOptions{}) {}

PrefixCacheTransactionManager::PrefixCacheTransactionManager(BlockPoolPtr     block_pool,
                                                             BlockCachePtr    block_cache,
                                                             CopyBlock        copy_block,
                                                             LifecycleOptions options,
                                                             Now              now,
                                                             WallNow          wall_now):
    block_pool_(std::move(block_pool)),
    block_cache_(std::move(block_cache)),
    copy_block_(std::move(copy_block)),
    options_(options),
    now_(std::move(now)),
    wall_now_(std::move(wall_now)) {
    if (!now_ || !wall_now_ || options_.lease_ttl.count() <= 0 || options_.terminal_ttl.count() <= 0
        || options_.sweep_interval.count() <= 0 || options_.max_records == 0) {
        throw std::invalid_argument("invalid prefix transaction lifecycle options");
    }
    wall_anchor_ms_ = wall_now_();
    steady_anchor_  = now_();
}

int64_t PrefixCacheTransactionManager::wallTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::optional<PrefixCacheTransactionManager::Clock::time_point>
PrefixCacheTransactionManager::prepareDeadlineLocked(const std::optional<PrepareIdentity>& prepare,
                                                     Clock::time_point                     now) {
    if (!prepare) {
        return now + options_.lease_ttl;
    }
    auto wall = wall_anchor_ms_ + std::chrono::duration_cast<std::chrono::milliseconds>(now - steady_anchor_).count();
    const auto observed = wall_now_();
    if (observed > wall) {
        wall_anchor_ms_ = observed;
        steady_anchor_  = now;
        wall            = observed;
    }
    if ((!prepare->transfer.empty() && !prepare->transfer.valid()) || prepare->request_id.empty()
        || prepare->request_id.size() > 128 || prepare->namespace_id.empty() || prepare->namespace_id.size() > 512
        || wall <= 0 || prepare->deadline_unix_ms <= wall
        || prepare->deadline_unix_ms - wall > options_.lease_ttl.count()) {
        return std::nullopt;
    }
    return now + std::chrono::milliseconds(prepare->deadline_unix_ms - wall);
}

PrefixCacheTransactionManager::~PrefixCacheTransactionManager() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
        wake_.notify_all();
    }
    if (reaper_.joinable()) {
        reaper_.join();
    }
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [id, record] : imports_) {
        if (record.operation.state != State::COMMITTED && !record.released) {
            block_pool_->requestFree(record.operation.stagingBlocks());
        }
    }
    for (const auto& [id, source] : exports_) {
        if (!source.released) {
            block_pool_->connectorFree(source.lease.blocks);
        }
    }
}

bool PrefixCacheTransactionManager::authorizedLocked(const ControllerCommand&  controller,
                                                     const ControllerIdentity* owner) const {
    if (!controller_) {
        return !controller && (!owner || *owner == ControllerIdentity{});
    }
    return controller && *controller == *controller_ && (!owner || *owner == *controller);
}

bool PrefixCacheTransactionManager::activateController(const ControllerIdentity& controller) {
    if (!controller.valid()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_) {
        return false;
    }
    if (!topology_control_binding_.empty()) {
        return controller_ && *controller_ == controller;
    }
    if (controller_) {
        if (controller == *controller_) {
            return true;
        }
        if (controller.namespace_id != controller_->namespace_id || controller.term <= controller_->term) {
            return false;
        }
    }
    const auto now = now_();
    for (auto& entry : imports_) {
        finishImportLocked(entry.second, std::nullopt, now);
    }
    for (auto& entry : exports_) {
        releaseExportLocked(entry.second, now);
    }
    controller_ = controller;
    released_topology_control_binding_.clear();
    return true;
}

bool PrefixCacheTransactionManager::isControllerCurrent(const ControllerIdentity& controller) const {
    std::lock_guard<std::mutex> lock(mu_);
    return controller_ && *controller_ == controller;
}

bool PrefixCacheTransactionManager::bindTopologyController(const ControllerIdentity& controller,
                                                           const std::string&        binding) {
    std::lock_guard<std::mutex> lock(mu_);
    if (binding.empty() || binding.size() > 65536 || !controller_ || !(*controller_ == controller)
        || (topology_control_binding_.empty() && topology_changing_)) {
        return false;
    }
    if (!topology_control_binding_.empty()) {
        return topology_control_binding_ == binding;
    }
    if (topology_receipt_ && topology_receipt_->binding == binding && topology_receipt_->controller == controller) {
        return false;
    }
    topology_control_binding_ = binding;
    topology_receipt_         = TopologyReceipt{controller, binding, TopologyState::ACTIVE};
    return true;
}

std::optional<PrefixCacheTransactionManager::TopologyReceipt> PrefixCacheTransactionManager::topologyReceipt() const {
    std::lock_guard<std::mutex> lock(mu_);
    return topology_receipt_;
}

bool PrefixCacheTransactionManager::topologyControllerMatches(const ControllerIdentity& controller,
                                                              const std::string&        binding) const {
    std::lock_guard<std::mutex> lock(mu_);
    return !binding.empty() && controller_ && *controller_ == controller && topology_control_binding_ == binding;
}

bool PrefixCacheTransactionManager::releaseTopologyController(const ControllerIdentity& controller,
                                                              const std::string&        binding,
                                                              bool                      completed) {
    std::lock_guard<std::mutex> lock(mu_);
    if (binding.empty() || !controller_ || !(*controller_ == controller) || topology_control_binding_ != binding
        || topology_changing_) {
        return false;
    }
    released_topology_control_binding_ = topology_control_binding_;
    topology_control_binding_.clear();
    topology_receipt_ =
        TopologyReceipt{controller, binding, completed ? TopologyState::RELEASED : TopologyState::CANCELLED};
    return true;
}

bool PrefixCacheTransactionManager::bindTopologyQuarantine(const ControllerIdentity& controller,
                                                           const std::string&        binding) {
    std::lock_guard<std::mutex> lock(mu_);
    if (binding.empty() || !controller_ || !(*controller_ == controller)
        || (topology_receipt_ && topology_receipt_->state == TopologyState::CONFIRMED)
        || (!topology_control_binding_.empty() && topology_control_binding_ != binding)
        || (topology_control_binding_.empty() && released_topology_control_binding_ != binding)) {
        return false;
    }
    topology_control_binding_ = binding;
    topology_receipt_         = TopologyReceipt{controller, binding, TopologyState::QUARANTINED};
    return true;
}

bool PrefixCacheTransactionManager::confirmTopologyController(const ControllerIdentity& controller,
                                                              const std::string&        binding) {
    std::lock_guard<std::mutex> lock(mu_);
    if (binding.empty() || !topology_control_binding_.empty() || topology_changing_ || !topology_receipt_
        || !(topology_receipt_->controller == controller) || topology_receipt_->binding != binding
        || (topology_receipt_->state != TopologyState::RELEASED
            && topology_receipt_->state != TopologyState::CONFIRMED)) {
        return false;
    }
    topology_receipt_->state = TopologyState::CONFIRMED;
    return true;
}

uint64_t PrefixCacheTransactionManager::beginTopologyChange() {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_) {
        return topology_generation_;
    }
    if (topology_generation_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("prefix topology generation exhausted");
    }
    topology_changing_ = true;
    ++topology_generation_;
    topology_operation_floor_ = next_id_;
    const auto now            = now_();
    for (auto& entry : imports_) {
        finishImportLocked(entry.second, std::nullopt, now);
    }
    for (auto& entry : exports_) {
        releaseExportLocked(entry.second, now);
    }
    return topology_generation_;
}

bool PrefixCacheTransactionManager::topologyChangeDrainedLocked(uint64_t generation) const {
    return topology_changing_ && generation == topology_generation_ && busy_exports_.empty()
           && active_transfer_leases_->load(std::memory_order_acquire) == 0;
}

bool PrefixCacheTransactionManager::topologyChangeDrained(uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mu_);
    return topologyChangeDrainedLocked(generation);
}

bool PrefixCacheTransactionManager::resumeAfterTopologyChange(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!topologyChangeDrainedLocked(generation)) {
        return false;
    }
    topology_changing_ = false;
    return true;
}

PrefixCacheTransactionManager::ResourceLease
PrefixCacheTransactionManager::pinBlocksLocked(const BlockIndicesType& blocks) {
    auto retained_blocks = blocks;
    auto pool            = block_pool_;
    auto activity        = active_transfer_leases_;
    pool->connectorReference(retained_blocks);
    activity->fetch_add(1, std::memory_order_relaxed);
    return ResourceLease(nullptr, [pool, activity, blocks = std::move(retained_blocks)](void*) {
        pool->connectorFree(blocks);
        activity->fetch_sub(1, std::memory_order_release);
    });
}

void PrefixCacheTransactionManager::finishImportLocked(ImportRecord&           record,
                                                       std::optional<uint64_t> committed_version,
                                                       Clock::time_point       now) {
    if (record.released) {
        return;
    }
    if (committed_version) {
        block_pool_->releasePublishedPrefixReservations(record.operation.stagingBlocks());
    }
    block_pool_->requestFree(record.operation.stagingBlocks());
    record.operation.state             = committed_version ? State::COMMITTED : State::ABORTED;
    record.operation.committed_version = committed_version.value_or(0);
    record.released                    = true;
    record.ancestor_lease.reset();
    record.retire_at = now + options_.terminal_ttl;
    if (!record.operation.prepare.request_id.empty()) {
        record.retire_at = std::max(record.retire_at, record.operation.expires_at);
    }
    transfer_states_.erase(record.operation.id);
}

void PrefixCacheTransactionManager::releaseExportLocked(ExportRecord& record, Clock::time_point now) {
    if (!record.released) {
        block_pool_->connectorFree(record.lease.blocks);
        record.released     = true;
        record.lease.active = false;
        record.retire_at    = now + options_.terminal_ttl;
        if (!record.lease.prepare.request_id.empty()) {
            record.retire_at = std::max(record.retire_at, record.lease.expires_at);
        }
    }
}

void PrefixCacheTransactionManager::expireLocked(Clock::time_point now) {
    for (auto entry = imports_.begin(); entry != imports_.end();) {
        auto& record = entry->second;
        if (!record.released && now >= record.operation.expires_at) {
            finishImportLocked(record, std::nullopt, now);
        }
        if (record.released && !record.copying_locally && now >= record.retire_at) {
            import_requests_.erase(record.operation.prepare.request_id);
            entry = imports_.erase(entry);
        } else {
            ++entry;
        }
    }
    for (auto entry = exports_.begin(); entry != exports_.end();) {
        auto& record = entry->second;
        if (!record.released && now >= record.lease.expires_at) {
            releaseExportLocked(record, now);
        }
        if (record.released && !busy_exports_.count(entry->first) && now >= record.retire_at) {
            export_requests_.erase(record.lease.prepare.request_id);
            entry = exports_.erase(entry);
        } else {
            ++entry;
        }
    }
}

void PrefixCacheTransactionManager::startReaperLocked() {
    if (!reaper_.joinable()) {
        reaper_ = std::thread(&PrefixCacheTransactionManager::runReaper, this);
    }
}

void PrefixCacheTransactionManager::runReaper() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!wake_.wait_for(lock, options_.sweep_interval, [this] { return stopping_; })) {
        expireLocked(now_());
    }
}

std::optional<PrefixCacheTransactionManager::ExportLease>
PrefixCacheTransactionManager::prepareExport(uint64_t                         epoch,
                                             uint64_t                         version,
                                             const std::vector<CacheKeyType>& keys,
                                             std::optional<PrepareIdentity>   prepare,
                                             ControllerCommand                controller) {
    if (keys.empty()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || !topology_control_binding_.empty() || !authorizedLocked(controller)
        || (prepare && controller && prepare->namespace_id != controller->namespace_id)) {
        return std::nullopt;
    }
    if (prepare && !prepare->transfer.empty()
        && (!controller || !prepare->transfer.valid()
            || std::hash<std::string>{}(prepare->transfer.source_epoch) != epoch)) {
        return std::nullopt;
    }
    const auto now = now_();
    expireLocked(now);
    if (prepare) {
        auto request = export_requests_.find(prepare->request_id);
        if (request != export_requests_.end()) {
            const auto& source = exports_.at(request->second).lease;
            return source.id >= topology_operation_floor_ && authorizedLocked(controller, &source.controller)
                           && source.prepare == *prepare && source.epoch == epoch && source.version == version
                           && source.keys == keys ?
                       std::optional<ExportLease>(source) :
                       std::nullopt;
        }
    }
    const auto deadline = prepareDeadlineLocked(prepare, now);
    if (!deadline || epoch == 0 || exports_.size() >= options_.max_records) {
        return std::nullopt;
    }
    startReaperLocked();
    auto blocks = block_cache_->pinPrefix(
        version, keys, [this](const BlockIndicesType& indices) { block_pool_->connectorReference(indices); });
    if (!blocks.has_value()) {
        return std::nullopt;
    }
    ExportLease source;
    source.id         = next_id_++;
    source.epoch      = epoch;
    source.version    = version;
    source.keys       = keys;
    source.blocks     = std::move(*blocks);
    source.expires_at = *deadline;
    source.prepare    = prepare.value_or(PrepareIdentity{});
    source.controller = controller.value_or(ControllerIdentity{});
    exports_.emplace(source.id, ExportRecord{source});
    if (prepare) {
        export_requests_.emplace(prepare->request_id, source.id);
    }
    return source;
}

std::optional<PrefixCacheTransactionManager::ImportOperation>
PrefixCacheTransactionManager::prepareImport(uint64_t                         epoch,
                                             uint64_t                         expected_version,
                                             const std::vector<CacheKeyType>& keys,
                                             int                              rank_count,
                                             uint64_t                         source_lease_id,
                                             std::optional<PrepareIdentity>   prepare,
                                             ControllerCommand                controller) {
    if (keys.empty() || keys.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || !topology_control_binding_.empty() || !authorizedLocked(controller)
        || (prepare && controller && prepare->namespace_id != controller->namespace_id)) {
        return std::nullopt;
    }
    if (prepare && !prepare->transfer.empty()
        && (!controller || !prepare->transfer.valid()
            || std::hash<std::string>{}(prepare->transfer.destination_epoch) != epoch)) {
        return std::nullopt;
    }
    const auto now = now_();
    expireLocked(now);
    if (prepare) {
        auto request = import_requests_.find(prepare->request_id);
        if (request != import_requests_.end()) {
            const auto& operation = imports_.at(request->second).operation;
            return operation.id >= topology_operation_floor_ && authorizedLocked(controller, &operation.controller)
                           && operation.prepare == *prepare && operation.epoch == epoch
                           && operation.expected_version == expected_version && operation.keys == keys
                           && operation.rank_count == rank_count && operation.source_lease_id == source_lease_id ?
                       std::optional<ImportOperation>(operation) :
                       std::nullopt;
        }
    }
    const auto deadline = prepareDeadlineLocked(prepare, now);
    if (!deadline || rank_count <= 0 || epoch == 0 || imports_.size() >= options_.max_records) {
        return std::nullopt;
    }
    startReaperLocked();
    auto plan = block_cache_->pinPrefixForImport(
        expected_version, keys, [this](const BlockIndicesType& blocks) { block_pool_->connectorReference(blocks); });
    if (!plan.has_value()) {
        return std::nullopt;
    }
    ResourceLease ancestor_lease(nullptr, [pool = block_pool_, blocks = std::move(plan->existing_blocks)](void*) {
        pool->connectorFree(blocks);
    });
    auto          destination = plan->missing_indices.empty() ?
                                    BlockIndicesType{} :
                                    block_pool_->mallocPrefixImportBlocks(static_cast<int>(plan->missing_indices.size()));
    if (destination.size() != plan->missing_indices.size()) {
        if (!destination.empty()) {
            block_pool_->requestFree(destination);
        }
        return std::nullopt;
    }
    for (size_t index = 0; index < destination.size(); ++index) {
        plan->blocks_by_key[plan->missing_indices[index]] = destination[index];
    }
    ImportOperation operation{next_id_++,
                              epoch,
                              expected_version,
                              source_lease_id,
                              State::PREPARED,
                              keys,
                              std::move(plan->blocks_by_key),
                              rank_count};
    operation.missing_indices     = std::move(plan->missing_indices);
    operation.pinned_generations  = std::move(plan->generations_by_key);
    operation.pinned_path_binding = options_.pinned_import_commit && source_lease_id != 0;
    operation.expires_at          = *deadline;
    operation.prepare             = prepare.value_or(PrepareIdentity{});
    operation.controller          = controller.value_or(ControllerIdentity{});
    imports_.emplace(operation.id, ImportRecord{operation, false, std::move(ancestor_lease)});
    if (prepare) {
        import_requests_.emplace(prepare->request_id, operation.id);
    }
    transfer_states_.try_emplace(operation.id, operation.id, epoch, "", rank_count);
    return operation;
}

std::optional<PrefixCacheTransactionManager::ExportLease> PrefixCacheTransactionManager::getPreparedExport(
    uint64_t epoch, const std::string& namespace_id, const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        found = export_requests_.find(request_id);
    if (found == export_requests_.end()) {
        return std::nullopt;
    }
    auto source = exports_.at(found->second).lease;
    if (source.epoch != epoch || source.prepare.namespace_id != namespace_id) {
        return std::nullopt;
    }
    source.active = source.active && now_() < source.expires_at;
    return source;
}

std::optional<PrefixCacheTransactionManager::ImportOperation> PrefixCacheTransactionManager::getPreparedImport(
    uint64_t epoch, const std::string& namespace_id, const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        found = import_requests_.find(request_id);
    if (found == import_requests_.end()) {
        return std::nullopt;
    }
    const auto& record    = imports_.at(found->second);
    auto        operation = record.operation;
    if (operation.epoch != epoch || operation.prepare.namespace_id != namespace_id) {
        return std::nullopt;
    }
    if (!record.released && now_() >= operation.expires_at) {
        operation.state = State::ABORTED;
    }
    return operation;
}

bool PrefixCacheTransactionManager::commitImport(uint64_t           id,
                                                 const ExportLease& source,
                                                 uint64_t           current_epoch,
                                                 uint64_t           current_version,
                                                 ControllerCommand  controller) {
    ImportOperation operation;
    ResourceLease   copy_lease;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (topology_changing_ || id < topology_operation_floor_ || source.id < topology_operation_floor_
            || !authorizedLocked(controller, &source.controller)) {
            return false;
        }
        expireLocked(now_());
        auto import_it = imports_.find(id);
        auto export_it = exports_.find(source.id);
        if (import_it == imports_.end() || !authorizedLocked(controller, &import_it->second.operation.controller)
            || !(import_it->second.operation.prepare.transfer == source.prepare.transfer)) {
            return false;
        }
        if (import_it != imports_.end() && import_it->second.operation.state == State::COMMITTED) {
            const auto& record = import_it->second;
            return record.operation.epoch == current_epoch && record.operation.expected_version == current_version
                   && record.local_source && record.local_source->id == source.id
                   && record.local_source->epoch == source.epoch && record.local_source->version == source.version
                   && record.local_source->prepare == source.prepare && record.local_source->keys == source.keys
                   && record.local_source->blocks == source.blocks;
        }
        if (current_version != block_cache_->prefixMutationVersion() || import_it == imports_.end()
            || export_it == exports_.end() || import_it->second.released || export_it->second.released
            || !authorizedLocked(controller, &export_it->second.lease.controller)
            || export_it->second.lease.epoch != current_epoch || source.epoch != current_epoch
            || import_it->second.operation.epoch != current_epoch || export_it->second.lease.version != source.version
            || export_it->second.lease.keys != source.keys || export_it->second.lease.blocks != source.blocks
            || !(export_it->second.lease.prepare == source.prepare) || source.version != current_version
            || import_it->second.operation.expected_version != current_version
            || import_it->second.operation.source_lease_id != 0 || import_it->second.operation.rank_count != 1
            || busy_exports_.count(source.id) != 0
            || (!copy_block_ && !import_it->second.operation.missing_indices.empty())
            || source.keys != import_it->second.operation.keys || source.blocks.size() != source.keys.size()) {
            return false;
        }
        if (import_it->second.operation.state != State::PREPARED) {
            return false;
        }
        auto state_it = transfer_states_.find(id);
        if (state_it == transfer_states_.end() || !state_it->second.beginCopy()) {
            return false;
        }
        import_it->second.operation.state           = State::COPYING;
        import_it->second.copying_locally           = true;
        import_it->second.operation.local_export_id = source.id;
        import_it->second.local_source              = source;
        busy_exports_.insert(source.id);
        operation          = import_it->second.operation;
        auto       blocks  = source.blocks;
        const auto staging = operation.stagingBlocks();
        blocks.insert(blocks.end(), staging.begin(), staging.end());
        copy_lease = pinBlocksLocked(blocks);
    }

    bool copied = true;
    try {
        for (const auto index : operation.missing_indices) {
            copy_block_(source.blocks[index], operation.destination_blocks[index]);
        }
    } catch (...) {
        copied = false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto                  now = now_();
    expireLocked(now);
    busy_exports_.erase(source.id);
    auto       import_it     = imports_.find(id);
    auto       export_it     = exports_.find(source.id);
    const bool source_active = export_it != exports_.end() && !export_it->second.released;
    if (export_it != exports_.end()) {
        releaseExportLocked(export_it->second, now);
    }
    if (import_it == imports_.end()) {
        transfer_states_.erase(id);
        return false;
    }
    auto& record           = import_it->second;
    record.copying_locally = false;
    if (record.released) {
        return false;
    }
    auto                    state_it = transfer_states_.find(id);
    std::optional<uint64_t> published_version;
    if (authorizedLocked(controller, &record.operation.controller) && copied && source_active
        && record.operation.state == State::COPYING && state_it != transfer_states_.end()
        && state_it->second.markRankReady(0)) {
        published_version = block_cache_->publishPrefix(
            current_version,
            operation.keys,
            operation.destination_blocks,
            [this](const BlockIndicesType& indices) { block_pool_->blockCacheReference(indices); },
            operation.id,
            operation.epoch);
    }
    finishImportLocked(record, published_version, now);
    return published_version.has_value();
}

bool PrefixCacheTransactionManager::markImportRankReady(uint64_t          id,
                                                        uint64_t          current_epoch,
                                                        int               rank,
                                                        ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || id < topology_operation_floor_ || !authorizedLocked(controller)) {
        return false;
    }
    expireLocked(now_());
    auto import_it = imports_.find(id);
    auto state_it  = transfer_states_.find(id);
    if (import_it == imports_.end() || state_it == transfer_states_.end() || import_it->second.copying_locally
        || !authorizedLocked(controller, &import_it->second.operation.controller)
        || import_it->second.operation.epoch != current_epoch || import_it->second.operation.state != State::COPYING) {
        return false;
    }
    if (!state_it->second.markRankReady(rank)) {
        return false;
    }
    if (state_it->second.state() == PrefixTransferState::State::DATA_READY) {
        import_it->second.operation.state = State::DATA_READY;
    }
    return true;
}

bool PrefixCacheTransactionManager::beginImportTransfer(uint64_t          id,
                                                        uint64_t          current_epoch,
                                                        ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || id < topology_operation_floor_ || !authorizedLocked(controller)) {
        return false;
    }
    expireLocked(now_());
    auto import_it = imports_.find(id);
    auto state_it  = transfer_states_.find(id);
    if (import_it == imports_.end() || state_it == transfer_states_.end()
        || !authorizedLocked(controller, &import_it->second.operation.controller)
        || import_it->second.operation.epoch != current_epoch || import_it->second.operation.state != State::PREPARED
        || !state_it->second.beginCopy()) {
        return false;
    }
    import_it->second.operation.state = State::COPYING;
    return true;
}

bool PrefixCacheTransactionManager::importSnapshotCurrent(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto                  found = imports_.find(id);
    if (found == imports_.end() || topology_changing_ || id < topology_operation_floor_) {
        return false;
    }
    const auto& operation = found->second.operation;
    if (now_() >= operation.expires_at || (operation.state != State::COPYING && operation.state != State::DATA_READY)) {
        return false;
    }
    return operation.pinned_path_binding ? block_cache_->pinnedPrefixUnchanged(operation.expected_version,
                                                                               operation.keys,
                                                                               operation.destination_blocks,
                                                                               operation.pinned_generations) :
                                           block_cache_->prefixMutationVersion() == operation.expected_version;
}

bool PrefixCacheTransactionManager::commitImportedData(uint64_t          id,
                                                       uint64_t          source_lease_id,
                                                       uint64_t          current_epoch,
                                                       uint64_t          current_version,
                                                       ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || id < topology_operation_floor_ || !authorizedLocked(controller)) {
        return false;
    }
    const auto now = now_();
    expireLocked(now);
    auto import_it = imports_.find(id);
    auto state_it  = transfer_states_.find(id);
    if (import_it == imports_.end() || source_lease_id == 0
        || !authorizedLocked(controller, &import_it->second.operation.controller)
        || import_it->second.operation.epoch != current_epoch
        || import_it->second.operation.source_lease_id != source_lease_id
        || import_it->second.operation.expected_version != current_version) {
        return false;
    }
    if (import_it->second.operation.state == State::COMMITTED) {
        return true;
    }
    if (state_it == transfer_states_.end() || import_it->second.operation.state != State::DATA_READY
        || state_it->second.state() != PrefixTransferState::State::DATA_READY) {
        return false;
    }
    auto published_version = block_cache_->publishPrefix(
        current_version,
        import_it->second.operation.keys,
        import_it->second.operation.destination_blocks,
        [this](const BlockIndicesType& indices) { block_pool_->blockCacheReference(indices); },
        import_it->second.operation.id,
        import_it->second.operation.epoch,
        import_it->second.operation.pinned_path_binding ? &import_it->second.operation.pinned_generations : nullptr);
    if (!published_version.has_value()) {
        return false;
    }
    finishImportLocked(import_it->second, published_version, now);
    return true;
}

bool PrefixCacheTransactionManager::commitImport(
    uint64_t id, uint64_t export_id, uint64_t current_epoch, uint64_t current_version, ControllerCommand controller) {
    std::optional<ExportLease> source;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!authorizedLocked(controller)) {
            return false;
        }
        expireLocked(now_());
        auto found = imports_.find(id);
        if (found != imports_.end() && found->second.operation.state == State::COMMITTED
            && found->second.operation.local_export_id == export_id) {
            source = found->second.local_source;
        }
    }
    if (!source) {
        source = getExport(export_id);
    }
    return source.has_value() && commitImport(id, *source, current_epoch, current_version, controller);
}

bool PrefixCacheTransactionManager::abortImport(uint64_t id, uint64_t current_epoch, ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!authorizedLocked(controller)) {
        return false;
    }
    const auto now = now_();
    expireLocked(now);
    auto it = imports_.find(id);
    if (it == imports_.end() || !authorizedLocked(controller, &it->second.operation.controller)
        || it->second.operation.epoch != current_epoch) {
        return false;
    }
    if (it->second.operation.state == State::ABORTED) {
        return true;
    }
    if (it->second.operation.state == State::COMMITTED || it->second.released) {
        return false;
    }
    finishImportLocked(it->second, std::nullopt, now);
    return true;
}

bool PrefixCacheTransactionManager::releaseExport(uint64_t id, uint64_t current_epoch, ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!authorizedLocked(controller)) {
        return false;
    }
    const auto now = now_();
    expireLocked(now);
    auto it = exports_.find(id);
    if (it == exports_.end() || !authorizedLocked(controller, &it->second.lease.controller)
        || it->second.lease.epoch != current_epoch) {
        return false;
    }
    if (!it->second.released && busy_exports_.count(id)) {
        return false;
    }
    releaseExportLocked(it->second, now);
    return true;
}

bool PrefixCacheTransactionManager::reclaim(uint64_t                         epoch,
                                            uint64_t                         expected_version,
                                            const std::vector<CacheKeyType>& keys,
                                            ControllerCommand                controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || !topology_control_binding_.empty() || !authorizedLocked(controller) || epoch == 0) {
        return false;
    }
    return block_cache_
        ->removePrefix(
            expected_version,
            keys,
            [this](const BlockIndicesType& indices) { block_pool_->blockCacheFree(indices); },
            [this](const BlockIndicesType& indices) { return block_pool_->canReclaimCacheBlocks(indices); })
        .has_value();
}

PrefixCacheTransactionManager::ImportOperation PrefixCacheTransactionManager::getImport(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = imports_.find(id);
    if (it == imports_.end()) {
        return {};
    }
    auto operation = it->second.operation;
    if (!it->second.released && now_() >= operation.expires_at) {
        operation.state = State::ABORTED;
    }
    return operation;
}

std::optional<PrefixCacheTransactionManager::ExportLease> PrefixCacheTransactionManager::getExport(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = exports_.find(id);
    return it == exports_.end() || it->second.released || now_() >= it->second.lease.expires_at ?
               std::nullopt :
               std::optional<ExportLease>(it->second.lease);
}

std::optional<PrefixCacheTransactionManager::ResourceLease>
PrefixCacheTransactionManager::pinExport(uint64_t id, uint64_t current_epoch, ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || id < topology_operation_floor_ || !authorizedLocked(controller)) {
        return std::nullopt;
    }
    expireLocked(now_());
    auto it = exports_.find(id);
    if (it == exports_.end() || !authorizedLocked(controller, &it->second.lease.controller) || it->second.released
        || it->second.lease.epoch != current_epoch) {
        return std::nullopt;
    }
    return pinBlocksLocked(it->second.lease.blocks);
}

std::optional<PrefixCacheTransactionManager::ResourceLease>
PrefixCacheTransactionManager::pinImport(uint64_t id, uint64_t current_epoch, ControllerCommand controller) {
    std::lock_guard<std::mutex> lock(mu_);
    if (topology_changing_ || id < topology_operation_floor_ || !authorizedLocked(controller)) {
        return std::nullopt;
    }
    expireLocked(now_());
    auto it = imports_.find(id);
    if (it == imports_.end() || !authorizedLocked(controller, &it->second.operation.controller)
        || it->second.operation.epoch != current_epoch || it->second.released
        || it->second.operation.state == State::ABORTED || it->second.operation.state == State::COMMITTED) {
        return std::nullopt;
    }
    return pinBlocksLocked(it->second.operation.stagingBlocks());
}

uint64_t PrefixCacheTransactionManager::mutationVersion() const {
    std::lock_guard<std::mutex> lock(mu_);
    return block_cache_->prefixMutationVersion();
}

}  // namespace rtp_llm
