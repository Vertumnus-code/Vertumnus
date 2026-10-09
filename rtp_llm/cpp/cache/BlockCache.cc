#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

BlockCache::MatchResult BlockCache::match(CacheKeyType cache_key, int group_id) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    auto [success, item] = lru_cache_.get(key);
    if (success) {
        return {item.block_index};
    } else {
        return {NULL_BLOCK_IDX};
    }
}

bool BlockCache::contains(CacheKeyType cache_key, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    return lru_cache_.contains(key);
}

size_t BlockCache::touchReadyPrefix(const CacheKeysType& keys, uint64_t ready_generation) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t                      touched = 0;
    auto                        ready   = [ready_generation](const CacheItem& item) {
        return item.inventory_generation > 0 && item.inventory_generation <= ready_generation;
    };
    for (const auto key : keys) {
        if (!lru_cache_.touchIf({key, 0}, ready)) {
            break;
        }
        ++touched;
    }
    return touched;
}

bool BlockCache::put(CacheItem& item) {
    return put(item, {});
}

bool BlockCache::put(CacheItem& item, const std::function<void(BlockIdxType)>& reference) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    CacheKeyGroupPair key{item.cache_key, item.group_id};

    if (lru_cache_.contains(key)) {
        if (recordPrefix(item)) {
            ++prefix_mutation_version_;
        }
        // It already exists; increase its popularity.
        lru_cache_.get(key);
        return false;
    }

    if (reference) {
        reference(item.block_index);
    }
    item.inventory_generation = ++inventory_generation_;
    recordPrefix(item);
    lru_cache_.put(key, item);
    prefix_inventory_snapshot_.reset();
    ++prefix_mutation_version_;
    return true;
}

BlockIndicesType BlockCache::matchWithReference(const std::vector<CacheKeyType>&                    keys,
                                                int                                                 group_id,
                                                const std::function<void(const BlockIndicesType&)>& reference) {
    std::lock_guard<std::mutex> lock(mu_);
    BlockIndicesType            blocks;
    if (!reference) {
        return blocks;
    }
    for (const auto key : keys) {
        auto [found, item] = lru_cache_.get({key, group_id});
        if (!found) {
            break;
        }
        blocks.push_back(item.block_index);
    }
    if (!blocks.empty()) {
        reference(blocks);
    }
    return blocks;
}

BlockIndicesType BlockCache::pop(int nums) {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(nums > 0, "pop nums should > 0, nums = " + std::to_string(nums));
    BlockIndicesType pop_blocks;

    auto cond = [&](const CacheKeyGroupPair& key, const CacheItem& item) { return !item.is_resident; };

    while (nums > 0 && !lru_cache_.empty()) {
        auto [success, item] = lru_cache_.popWithCond(cond);
        if (!success)
            break;
        pop_blocks.push_back(item.block_index);
        forgetPrefix(item);
        ++prefix_mutation_version_;
        nums--;
    }

    return pop_blocks;
}

std::optional<BlockCache::CacheItem> BlockCache::remove(CacheKeyType cache_key, int group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    CacheKeyGroupPair           key{cache_key, group_id};
    CacheItem                   removed_item;
    if (!lru_cache_.remove(key, &removed_item)) {
        return std::nullopt;
    }
    forgetPrefix(removed_item);
    ++prefix_mutation_version_;
    return removed_item;
}

bool BlockCache::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.empty();
}

size_t BlockCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.size();
}

BlockCache::CacheSnapshot BlockCache::cacheSnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.cacheSnapshot(latest_version);
}

int64_t BlockCache::cacheVersion() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_cache_.currentVersion();
}

std::shared_ptr<const BlockCache::CacheSnapshot> BlockCache::prefixInventorySnapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return prefixInventorySnapshotLocked();
}

uint64_t BlockCache::inventoryGeneration() const {
    std::lock_guard<std::mutex> lock(mu_);
    return inventory_generation_;
}

std::shared_ptr<const BlockCache::CacheSnapshot> BlockCache::prefixInventorySnapshotLocked() const {
    if (!prefix_inventory_snapshot_) {
        prefix_inventory_snapshot_ = std::make_shared<const CacheSnapshot>(lru_cache_.cacheSnapshot(-2));
    }
    return prefix_inventory_snapshot_;
}

uint64_t BlockCache::prefixMutationVersion() const {
    std::lock_guard<std::mutex> lock(mu_);
    return prefix_mutation_version_;
}

bool BlockCache::isKnownPrefixPath(const std::vector<CacheKeyType>& keys) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!compatiblePrefixPath(keys)) {
        return false;
    }
    for (const auto key : keys) {
        auto found = prefix_nodes_.find(key);
        if (found == prefix_nodes_.end() || !found->second.present || found->second.depth == 0) {
            return false;
        }
    }
    return true;
}

void BlockCache::invalidatePrefixSnapshot(bool preserves_pinned_paths) {
    std::lock_guard<std::mutex> lock(mu_);
    ++prefix_mutation_version_;
    if (!preserves_pinned_paths) {
        pinned_path_fence_version_ = prefix_mutation_version_;
    }
}

uint64_t BlockCache::observePrefixSnapshot(const std::function<bool(const CacheSnapshot&)>& observer) {
    return observePrefixSnapshotView(
        [&](const std::shared_ptr<const CacheSnapshot>& snapshot) { return observer(*snapshot); });
}

uint64_t
BlockCache::observePrefixSnapshotView(const std::function<bool(const std::shared_ptr<const CacheSnapshot>&)>& observer,
                                      bool preserves_pinned_paths) {
    std::lock_guard<std::mutex> lock(mu_);
    if (observer(prefixInventorySnapshotLocked())) {
        ++prefix_mutation_version_;
        if (!preserves_pinned_paths) {
            pinned_path_fence_version_ = prefix_mutation_version_;
        }
    }
    return prefix_mutation_version_;
}

std::optional<BlockIndicesType> BlockCache::pinPrefix(uint64_t                         expected_version,
                                                      const std::vector<CacheKeyType>& keys,
                                                      const std::function<void(const BlockIndicesType&)>& reference) {
    std::lock_guard<std::mutex> lock(mu_);
    if (keys.empty() || expected_version != prefix_mutation_version_ || !reference) {
        return std::nullopt;
    }
    std::unordered_map<CacheKeyType, BlockIdxType> available;
    for (const auto& entry : lru_cache_.items()) {
        if (entry.second.group_id == 0) {
            available.emplace(entry.second.cache_key, entry.second.block_index);
        }
    }
    BlockIndicesType                 blocks;
    std::unordered_set<CacheKeyType> requested;
    for (const auto key : keys) {
        const auto found = available.find(key);
        if (found == available.end() || !requested.insert(key).second) {
            return std::nullopt;
        }
        blocks.push_back(found->second);
    }
    reference(blocks);
    return blocks;
}

std::optional<BlockIndicesType>
BlockCache::pinExistingPrefix(uint64_t                                            expected_version,
                              const std::vector<CacheKeyType>&                    keys,
                              const std::function<void(const BlockIndicesType&)>& reference) {
    auto plan = pinPrefixForImport(expected_version, keys, reference);
    return plan ? std::optional<BlockIndicesType>(std::move(plan->existing_blocks)) : std::nullopt;
}

std::optional<BlockCache::PinnedPrefixPlan>
BlockCache::pinPrefixForImport(uint64_t                                            expected_version,
                               const std::vector<CacheKeyType>&                    keys,
                               const std::function<void(const BlockIndicesType&)>& reference) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expected_version != prefix_mutation_version_ || !reference || !compatiblePrefixPath(keys)) {
        return std::nullopt;
    }
    std::unordered_map<CacheKeyType, BlockIdxType> available;
    for (const auto& entry : lru_cache_.items()) {
        if (entry.second.group_id == 0) {
            available.emplace(entry.second.cache_key, entry.second.block_index);
        }
    }
    PinnedPrefixPlan plan;
    for (size_t index = 0; index < keys.size(); ++index) {
        auto found = available.find(keys[index]);
        if (found != available.end()) {
            plan.existing_blocks.push_back(found->second);
            plan.blocks_by_key.push_back(found->second);
            const auto [present, item] = lru_cache_.peek({keys[index], 0});
            if (!present || item.inventory_generation == 0) {
                return std::nullopt;
            }
            plan.generations_by_key.push_back(item.inventory_generation);
        } else {
            plan.blocks_by_key.push_back(NULL_BLOCK_IDX);
            plan.generations_by_key.push_back(0);
            plan.missing_indices.push_back(static_cast<uint32_t>(index));
        }
    }
    reference(plan.existing_blocks);
    return plan;
}

bool BlockCache::pinnedPrefixUnchanged(uint64_t                         expected_version,
                                       const std::vector<CacheKeyType>& keys,
                                       const BlockIndicesType&          blocks,
                                       const std::vector<uint64_t>&     generations) const {
    std::lock_guard<std::mutex> lock(mu_);
    return pinnedPrefixUnchangedLocked(expected_version, keys, blocks, generations);
}

bool BlockCache::pinnedPrefixUnchangedLocked(uint64_t                         expected_version,
                                             const std::vector<CacheKeyType>& keys,
                                             const BlockIndicesType&          blocks,
                                             const std::vector<uint64_t>&     generations) const {
    if (keys.empty() || keys.size() != blocks.size() || keys.size() != generations.size()
        || expected_version < pinned_path_fence_version_ || expected_version > prefix_mutation_version_
        || !compatiblePrefixPath(keys)) {
        return false;
    }
    std::unordered_set<BlockIdxType> unique_blocks;
    for (size_t index = 0; index < keys.size(); ++index) {
        if (blocks[index] <= 0 || !unique_blocks.insert(blocks[index]).second) {
            return false;
        }
        const auto [present, item] = lru_cache_.peek({keys[index], 0});
        if (generations[index] == 0 ?
                present :
                (!present || item.block_index != blocks[index] || item.inventory_generation != generations[index])) {
            return false;
        }
    }
    return true;
}

std::optional<uint64_t> BlockCache::publishPrefix(uint64_t                                            expected_version,
                                                  const std::vector<CacheKeyType>&                    keys,
                                                  const BlockIndicesType&                             blocks,
                                                  const std::function<void(const BlockIndicesType&)>& reference,
                                                  uint64_t                                            operation_id,
                                                  uint64_t                                            import_epoch,
                                                  const std::vector<uint64_t>* pinned_generations) {
    std::lock_guard<std::mutex> lock(mu_);
    // Revalidate and publish under the same lock. Existing physical blocks remain leased;
    // every originally missing key must still be absent. Never merge a concurrent mapping.
    const bool binding_valid = pinned_generations ?
                                   pinnedPrefixUnchangedLocked(expected_version, keys, blocks, *pinned_generations) :
                                   expected_version == prefix_mutation_version_;
    if (keys.size() != blocks.size() || !binding_valid || !reference || !compatiblePrefixPath(keys)) {
        return std::nullopt;
    }
    std::vector<CacheItem>           additions;
    BlockIndicesType                 new_blocks;
    std::unordered_set<CacheKeyType> requested;
    std::unordered_set<BlockIdxType> requested_blocks;
    for (size_t index = 0; index < keys.size(); ++index) {
        if (isNullBlockIdx(blocks[index]) || !requested.insert(keys[index]).second
            || !requested_blocks.insert(blocks[index]).second) {
            return std::nullopt;
        }
        if (!lru_cache_.contains({keys[index], 0})) {
            additions.push_back({keys[index], 0, blocks[index], false});
            new_blocks.push_back(blocks[index]);
        }
    }
    reference(new_blocks);
    for (size_t index = 0; index < keys.size(); ++index) {
        CacheItem item{keys[index], 0, blocks[index], false};
        item.prefix_depth = index + 1;
        if (index != 0) {
            item.prefix_parent = keys[index - 1];
        }
        if (recordPrefix(item) && lru_cache_.contains({keys[index], 0})) {
            ++prefix_mutation_version_;
        }
    }
    for (auto& item : additions) {
        item.inventory_generation = ++inventory_generation_;
        lru_cache_.put({item.cache_key, 0}, item);
        prefix_inventory_snapshot_.reset();
        if (operation_id != 0 && import_epoch != 0) {
            copy_ledger_->publish(
                item.inventory_generation, operation_id, import_epoch, item.cache_key, item.block_index);
        }
        ++prefix_mutation_version_;
    }
    return prefix_mutation_version_;
}

std::optional<uint64_t> BlockCache::removePrefix(uint64_t                                            expected_version,
                                                 const std::vector<CacheKeyType>&                    keys,
                                                 const std::function<void(const BlockIndicesType&)>& release,
                                                 const std::function<bool(const BlockIndicesType&)>& eligible) {
    std::lock_guard<std::mutex> lock(mu_);
    if (keys.empty() || expected_version != prefix_mutation_version_ || !release) {
        return std::nullopt;
    }
    std::unordered_map<CacheKeyType, CacheItem> available;
    for (const auto& entry : lru_cache_.items()) {
        if (entry.second.group_id == 0) {
            available.emplace(entry.second.cache_key, entry.second);
        }
    }
    BlockIndicesType                 blocks;
    std::unordered_set<CacheKeyType> requested;
    for (const auto key : keys) {
        const auto found = available.find(key);
        if (found == available.end() || found->second.is_resident || !requested.insert(key).second) {
            return std::nullopt;
        }
        blocks.push_back(found->second.block_index);
    }
    if (!reclaimablePrefixSet(requested) || (eligible && !eligible(blocks))) {
        return std::nullopt;
    }
    for (const auto key : keys) {
        lru_cache_.remove({key, 0});
        forgetPrefix(available.at(key));
        ++prefix_mutation_version_;
    }
    release(blocks);
    return prefix_mutation_version_;
}

BlockCache::EvictResult BlockCache::selectAndEvict(size_t min_blocks) {
    std::lock_guard<std::mutex> lock(mu_);

    EvictResult result;
    if (lru_cache_.empty()) {
        return result;
    }

    // First pass: collect resident keys
    std::unordered_set<CacheKeyType> resident_keys;
    for (const auto& [key, item] : lru_cache_.items()) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }

    // Second pass: group non-resident items by cache_key in LRU order (back = least-recently-used)
    std::unordered_map<CacheKeyType, std::vector<CacheItem>> grouped_items;
    std::vector<CacheKeyType>                                lru_keys;
    for (auto it = lru_cache_.items().rbegin(); it != lru_cache_.items().rend(); ++it) {
        const auto& item = it->second;
        if (item.is_resident) {
            continue;
        }
        auto [iter, inserted] = grouped_items.try_emplace(item.cache_key);
        if (inserted) {
            lru_keys.push_back(item.cache_key);
        }
        iter->second.push_back(item);
    }

    // Select keys until we have enough blocks
    std::vector<CacheKeyType> selected_keys;
    size_t                    selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        if (resident_keys.find(cache_key) != resident_keys.end()) {
            continue;
        }
        const auto group_it = grouped_items.find(cache_key);
        if (group_it == grouped_items.end() || group_it->second.empty()) {
            continue;
        }
        selected_keys.push_back(cache_key);
        selected_blocks += group_it->second.size();
        if (selected_blocks >= min_blocks) {
            break;
        }
    }

    if (selected_keys.empty()) {
        return result;
    }

    // Remove selected items from LRU cache and build result
    for (const auto cache_key : selected_keys) {
        auto&                  items = grouped_items.at(cache_key);
        std::vector<CacheItem> evicted_items;
        for (const auto& item : items) {
            CacheKeyGroupPair key{item.cache_key, item.group_id};
            CacheItem         removed_item;
            if (lru_cache_.remove(key, &removed_item)) {
                forgetPrefix(removed_item);
                ++prefix_mutation_version_;
                evicted_items.push_back(removed_item);
            }
        }
        if (!evicted_items.empty()) {
            result.evicted_keys.push_back(cache_key);
            result.evicted_items[cache_key] = std::move(evicted_items);
        }
    }

    return result;
}

bool BlockCache::recordPrefix(const CacheItem& item) {
    if (item.group_id != 0) {
        return false;
    }
    auto& node   = prefix_nodes_[item.cache_key];
    node.present = true;
    if (item.prefix_depth == 0) {
        return false;
    }
    const auto parent  = prefix_nodes_.find(item.prefix_parent.value_or(item.cache_key));
    bool       invalid = (item.prefix_depth == 1) != !item.prefix_parent.has_value()
                   || item.prefix_parent == std::optional<CacheKeyType>(item.cache_key)
                   || (item.prefix_parent && parent != prefix_nodes_.end() && parent->second.depth != 0
                       && parent->second.depth != item.prefix_depth - 1);
    if (node.depth == 0) {
        for (const auto child_key : node.children) {
            const auto& child = prefix_nodes_.at(child_key);
            if (child.depth == 0 || child.depth - 1 != item.prefix_depth) {
                invalid = true;
                break;
            }
        }
    }
    if (invalid || (node.depth != 0 && (node.depth != item.prefix_depth || node.parent != item.prefix_parent))) {
        const bool changed = !node.ambiguous;
        node.ambiguous     = true;
        return changed;
    }
    if (node.depth != 0) {
        return false;
    }
    node.depth  = item.prefix_depth;
    node.parent = item.prefix_parent;
    if (item.prefix_parent) {
        prefix_nodes_[*item.prefix_parent].children.insert(item.cache_key);
    }
    return true;
}

void BlockCache::forgetPrefix(const CacheItem& item) {
    prefix_inventory_snapshot_.reset();
    if (item.group_id != 0) {
        return;
    }
    copy_ledger_->evict(item.inventory_generation);
    auto found = prefix_nodes_.find(item.cache_key);
    if (found == prefix_nodes_.end()) {
        return;
    }
    found->second.present = false;
    while (!found->second.present && found->second.children.empty()) {
        const auto parent = found->second.parent;
        const auto key    = found->first;
        prefix_nodes_.erase(found);
        if (!parent) {
            return;
        }
        found = prefix_nodes_.find(*parent);
        if (found == prefix_nodes_.end()) {
            return;
        }
        found->second.children.erase(key);
    }
}

bool BlockCache::compatiblePrefixPath(const std::vector<CacheKeyType>& keys) const {
    if (keys.empty()) {
        return false;
    }
    std::unordered_set<CacheKeyType> unique;
    for (size_t index = 0; index < keys.size(); ++index) {
        if (!unique.insert(keys[index]).second) {
            return false;
        }
        const auto found  = prefix_nodes_.find(keys[index]);
        const auto parent = index == 0 ? std::nullopt : std::optional<CacheKeyType>(keys[index - 1]);
        if (found != prefix_nodes_.end()
            && (found->second.ambiguous
                || (found->second.depth != 0
                    && (found->second.depth != index + 1 || found->second.parent != parent)))) {
            return false;
        }
        if (found != prefix_nodes_.end() && found->second.depth == 0) {
            for (const auto child_key : found->second.children) {
                const auto& child = prefix_nodes_.at(child_key);
                if (child.depth == 0 || child.depth - 1 != index + 1) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool BlockCache::reclaimablePrefixSet(const std::unordered_set<CacheKeyType>& keys) const {
    for (const auto& [key, node] : prefix_nodes_) {
        if (node.ambiguous || node.depth == 0 || ((node.depth == 1) != !node.parent.has_value())) {
            return false;
        }
        if (node.parent) {
            const auto parent = prefix_nodes_.find(*node.parent);
            if (parent == prefix_nodes_.end() || parent->second.depth != node.depth - 1) {
                return false;
            }
        }
    }
    std::vector<CacheKeyType>        pending(keys.begin(), keys.end());
    std::unordered_set<CacheKeyType> visited;
    while (!pending.empty()) {
        const auto key = pending.back();
        pending.pop_back();
        if (!visited.insert(key).second) {
            continue;
        }
        const auto found = prefix_nodes_.find(key);
        if (found == prefix_nodes_.end() || (found->second.present && !keys.count(key))) {
            return false;
        }
        pending.insert(pending.end(), found->second.children.begin(), found->second.children.end());
    }
    return true;
}

}  // namespace rtp_llm
