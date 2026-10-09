#pragma once

#include <mutex>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/PrefixCopyLedger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

using CacheKeyGroupPair = std::pair<CacheKeyType, GroupIdType>;  // <cache_key, group_id>

class BlockCache {
public:
    struct CacheItem {
        CacheKeyType                cache_key;
        GroupIdType                 group_id;
        BlockIdxType                block_index;
        bool                        is_resident          = false;
        uint64_t                    inventory_generation = 0;
        size_t                      prefix_depth         = 0;
        std::optional<CacheKeyType> prefix_parent;
    };

    struct BatchMatchResult {
        std::vector<BlockIdxType> matched_indices;
    };

    struct MatchResult {
        BlockIdxType matched_index;
    };

    using LRUCacheType  = LRUCache<CacheKeyGroupPair,
                                   CacheItem,
                                   PairFirstHash<CacheKeyType, GroupIdType>,
                                   PairBothEqual<CacheKeyType, GroupIdType>>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    std::shared_ptr<PrefixCopyLedger> copyLedger() const {
        return copy_ledger_;
    }

    bool put(CacheItem& cache_item);
    bool put(CacheItem& cache_item, const std::function<void(BlockIdxType)>& reference);

    bool contains(CacheKeyType cache_key, int group_id = 0) const;

    MatchResult      match(CacheKeyType cache_key, int group_id = 0);
    size_t           touchReadyPrefix(const CacheKeysType& keys, uint64_t ready_generation);
    BlockIndicesType matchWithReference(const std::vector<CacheKeyType>&                    keys,
                                        int                                                 group_id,
                                        const std::function<void(const BlockIndicesType&)>& reference);

    BlockIndicesType pop(int n);

    std::optional<CacheItem> remove(CacheKeyType cache_key, int group_id = 0);

    // Select and remove LRU cache entries until at least min_blocks are freed.
    // Skips resident entries and keys that have any resident item.
    // Returns evicted items grouped by cache_key (in LRU order).
    struct EvictResult {
        std::vector<CacheKeyType>                                evicted_keys;
        std::unordered_map<CacheKeyType, std::vector<CacheItem>> evicted_items;
    };
    EvictResult selectAndEvict(size_t min_blocks);

    bool empty() const;

    size_t size() const;

    CacheSnapshot                        cacheSnapshot(int64_t latest_version) const;
    int64_t                              cacheVersion() const;
    std::shared_ptr<const CacheSnapshot> prefixInventorySnapshot() const;
    uint64_t                             inventoryGeneration() const;

    uint64_t prefixMutationVersion() const;
    bool     isKnownPrefixPath(const std::vector<CacheKeyType>& keys) const;
    void     invalidatePrefixSnapshot(bool preserves_pinned_paths = false);
    uint64_t observePrefixSnapshot(const std::function<bool(const CacheSnapshot&)>& observer);
    uint64_t observePrefixSnapshotView(const std::function<bool(const std::shared_ptr<const CacheSnapshot>&)>& observer,
                                       bool preserves_pinned_paths = false);
    std::optional<BlockIndicesType> pinPrefix(uint64_t                                            expected_version,
                                              const std::vector<CacheKeyType>&                    keys,
                                              const std::function<void(const BlockIndicesType&)>& reference);
    std::optional<BlockIndicesType> pinExistingPrefix(uint64_t                         expected_version,
                                                      const std::vector<CacheKeyType>& keys,
                                                      const std::function<void(const BlockIndicesType&)>& reference);
    struct PinnedPrefixPlan {
        BlockIndicesType      blocks_by_key;
        BlockIndicesType      existing_blocks;
        std::vector<uint32_t> missing_indices;
        std::vector<uint64_t> generations_by_key;
    };
    std::optional<PinnedPrefixPlan> pinPrefixForImport(uint64_t                         expected_version,
                                                       const std::vector<CacheKeyType>& keys,
                                                       const std::function<void(const BlockIndicesType&)>& reference);
    std::optional<uint64_t>         publishPrefix(uint64_t                                            expected_version,
                                                  const std::vector<CacheKeyType>&                    keys,
                                                  const BlockIndicesType&                             blocks,
                                                  const std::function<void(const BlockIndicesType&)>& reference,
                                                  uint64_t                                            operation_id = 0,
                                                  uint64_t                                            import_epoch = 0,
                                                  const std::vector<uint64_t>* pinned_generations = nullptr);
    bool                            pinnedPrefixUnchanged(uint64_t                         expected_version,
                                                          const std::vector<CacheKeyType>& keys,
                                                          const BlockIndicesType&          blocks,
                                                          const std::vector<uint64_t>&     generations) const;
    std::optional<uint64_t>         removePrefix(uint64_t                                            expected_version,
                                                 const std::vector<CacheKeyType>&                    keys,
                                                 const std::function<void(const BlockIndicesType&)>& release,
                                                 const std::function<bool(const BlockIndicesType&)>& eligible = {});

private:
    struct PrefixNode {
        size_t                           depth = 0;
        std::optional<CacheKeyType>      parent;
        std::unordered_set<CacheKeyType> children;
        bool                             present   = false;
        bool                             ambiguous = false;
    };

    bool                                 pinnedPrefixUnchangedLocked(uint64_t                         expected_version,
                                                                     const std::vector<CacheKeyType>& keys,
                                                                     const BlockIndicesType&          blocks,
                                                                     const std::vector<uint64_t>&     generations) const;
    bool                                 recordPrefix(const CacheItem& item);
    void                                 forgetPrefix(const CacheItem& item);
    bool                                 compatiblePrefixPath(const std::vector<CacheKeyType>& keys) const;
    bool                                 reclaimablePrefixSet(const std::unordered_set<CacheKeyType>& keys) const;
    std::shared_ptr<const CacheSnapshot> prefixInventorySnapshotLocked() const;

    std::unordered_map<CacheKeyType, PrefixNode> prefix_nodes_;
    size_t                                       seq_size_per_block_;
    LRUCacheType                                 lru_cache_;
    // NOTE: BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe (unordered_map + list). Guard all accesses here.
    mutable std::mutex                           mu_;
    uint64_t                                     inventory_generation_      = 0;
    uint64_t                                     prefix_mutation_version_   = 0;
    uint64_t                                     pinned_path_fence_version_ = 0;
    mutable std::shared_ptr<const CacheSnapshot> prefix_inventory_snapshot_;
    std::shared_ptr<PrefixCopyLedger>            copy_ledger_ = std::make_shared<PrefixCopyLedger>();
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
