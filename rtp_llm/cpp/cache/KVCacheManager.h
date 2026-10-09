#pragma once

#include <atomic>
#include <cassert>
#include <functional>
#include <optional>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class PrefixCacheTransactionManager;
class PrefixRankInstallQueue;
class CacheStore;
class KVCacheConnectorCoordinator;
class KVCacheConnectorReadWriteContext;

class KVCacheManager {
public:
    KVCacheManager(const CacheConfig&                 config,
                   bool                               warmup             = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter   = nullptr,
                   const KVCacheConfig&               kv_cache_config    = KVCacheConfig{},
                   const ParallelismConfig&           parallelism_config = ParallelismConfig{},
                   const RuntimeConfig&               runtime_config     = RuntimeConfig{},
                   const SpeculativeExecutionConfig&  sp_config          = SpeculativeExecutionConfig{},
                   const PDSepConfig&                 pd_sep_config      = PDSepConfig{},
                   const CacheStoreConfig&            cache_store_config = CacheStoreConfig{});
    ~KVCacheManager();
    void                                             recordPrefixReuse(int64_t            request_id,
                                                                       int64_t            device_reuse_tokens,
                                                                       int64_t            input_tokens,
                                                                       const std::string& attempt_id);
    bool                                             prefixInventoryEnabled() const;
    size_t                                           touchPrefixAtAdmission(const CompleteTokenIdsPtr& tokens);
    std::unique_ptr<PrefixCopyLedger::Use>           capturePrefixCopyUse(const BatchKVCacheResource& resource,
                                                                          int64_t                     reuse_tokens,
                                                                          int64_t                     request_id,
                                                                          const std::string&          attempt_id);
    PrefixTopologyIdentity                           prefixTopologyIdentity() const;
    BlockCache::CacheSnapshot                        prefixInventoryCandidates() const;
    std::shared_ptr<const BlockCache::CacheSnapshot> prefixInventorySnapshot() const;
    struct PrefixInventoryBoundary {
        std::shared_ptr<BlockCache> cache;
        std::string                 epoch;
        uint64_t                    generation = 0;
    };
    PrefixInventoryBoundary prefixInventoryBoundary() const;
    void                    confirmPrefixInventory(const PrefixInventoryBoundary& boundary, bool all_ranks_ready);
    void                    confirmPrefixInventory(const BlockCache::CacheSnapshot& candidates, bool all_ranks_ready);
    void                    confirmPrefixInventory(const std::shared_ptr<const BlockCache::CacheSnapshot>& candidates,
                                                   bool                                                    all_ranks_ready);
    void                    fencePrefixInventory();
    uint64_t                beginPrefixTopologyChange();
    bool                    prefixTopologyChangeDrained(uint64_t generation) const;
    bool                    prepareResidentPrefixSplit(uint64_t generation, const ParallelismConfig& parallelism);
    bool                    residentPrefixSplitPrepared(int64_t tp_size) const;
    bool                    prepareResidentPrefixMerge(uint64_t generation, const ParallelismConfig& parallelism);
    bool                    residentPrefixMergePrepared(int64_t tp_size) const;
    bool                    completePrefixTopologyChange(uint64_t generation, const ParallelismConfig& parallelism);
    bool                    cancelPrefixTopologyChange(uint64_t generation);

    // 初始化和配置相关
    bool init();

    const CacheConfig& cacheConfig() const;
    const CacheConfig& getMTPModuleCacheConfig(int mtp_module_id) const;

    // 显存管理和缓存分配
    MallocResult malloc(const MallocInfo& malloc_info);
    void         free(const FreeInfo& free_info);
    void         insertIntoCache(const InsertInfo& insert_info);

    int
    singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len, int reserve_step) const;

    // 块操作相关
    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const torch::Tensor& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping);

    // Write one KV block (optionally per-layer) from host/device tensors for test
    virtual bool
    setKVBlockValue(int block_index, int layer_id, const torch::Tensor& k_buffer, const torch::Tensor& v_buffer);
    virtual bool setKVBlockValue(int block_index, const torch::Tensor& k_buffer, const torch::Tensor& v_buffer);

    // 地址转换和缓冲区访问
    BlockAddrInfo          convertIndexToAddr(int block_index, int layer_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int block_index, int layer_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const;

    CacheLayerLayout allLayerCacheBase() const;

    // for main model; it's too hack for mtp module, but we need to keep it for now
    CacheLayerLayout getMainModelCacheLayerLayout() const;
    // for mtp module
    CacheLayerLayout getMTPModuleCacheLayerLayout(int mtp_module_id) const;

    // 资源统计和信息查询
    size_t                  freeBlocksNum() const;
    size_t                  availableBlocksNum() const;
    size_t                  notInUseBlocksNum() const;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free);
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource);
    size_t                  availableTokensNum() const;
    size_t                  totalBlocksNum() const;
    size_t                  maxAvailableTokensNum() const;
    KVCacheInfo             getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const;

    // 系统资源管理
    void regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);

    // CacheStore ownership (set by RemoteRpcServer, read during model forward)
    void                        setCacheStore(std::shared_ptr<CacheStore> cache_store);
    std::shared_ptr<CacheStore> getCacheStore() const;

    // 异步连接器操作
    // async load cache from connector to gpu, for all rank
    std::shared_ptr<AsyncContext>
    asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context);

    // async store cache from gpu to connector, for all rank
    std::shared_ptr<AsyncContext>
    asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context);

    // for every single rank
    bool executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);

    // handle read request from decode side (StartLoad RPC), delegate to coordinator
    void handleRead(const P2PConnectorStartLoadRequestPB& request,
                    P2PConnectorStartLoadResponsePB&      response,
                    std::function<bool()>                 is_cancelled = nullptr);

    bool hasActiveConnectors() const;
    bool hasP2PConnector() const;

    std::shared_ptr<KVCacheConnectorCoordinator> connectorCoordinator() const {
        return coordinator_;
    }

    KVCacheAllocatorPtr allocator() const {
        return allocator_;
    }

    std::vector<BlockInfo>  prefixBlockBuffers(int block_index) const;
    bool                    supportsPrefixRankReplication() const;
    bool                    prefixRankInstallPending() const;
    void                    processPrefixRankInstalls();
    PrefixRankInstallQueue* prefixRankInstallQueue() const {
        return prefix_rank_install_queue_.get();
    }

    PrefixCacheTransactionManager* prefixTransactionManager() const {
        return prefix_transaction_manager_.get();
    }

    // Increment KV cache reference count for PD separation (connector refcount)
    std::shared_ptr<KVCacheResource>
    incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector = true);

private:
    int                                                      prefix_admission_touch_blocks_ = 0;
    mutable PrefixInventory                                  prefix_inventory_;
    mutable int64_t                                          inventory_view_version_ = 0;
    mutable std::shared_ptr<const BlockCache::CacheSnapshot> inventory_keys_snapshot_;
    mutable std::optional<uint64_t>                          inventory_keys_ready_generation_;
    mutable std::string                                      inventory_keys_epoch_;
    mutable bool                                             inventory_keys_ready_ = false;
    mutable std::vector<int64_t>                             inventory_keys_;
    std::unique_ptr<PrefixCacheTransactionManager>           prefix_transaction_manager_;
    std::unique_ptr<PrefixRankInstallQueue>                  prefix_rank_install_queue_;
    mutable std::mutex                                       inventory_snapshot_mutex_;
    std::string                                              inventory_epoch_;
    std::string                                              inventory_boot_epoch_;
    uint64_t                                                 inventory_topology_generation_ = 0;
    std::vector<BlockCache::CacheItem>                       topology_source_ready_items_;
    std::vector<BlockCache::CacheItem>                       topology_resident_items_;
    uint64_t                                                 topology_resident_generation_     = 0;
    int64_t                                                  topology_resident_tp_size_        = 0;
    bool                                                     topology_resident_merge_          = false;
    bool                                                     topology_resident_repair_started_ = false;
    std::unordered_map<CacheKeyType, uint64_t>               inventory_ready_items_;
    std::shared_ptr<const BlockCache::CacheSnapshot>         inventory_ready_snapshot_;
    std::optional<uint64_t>                                  inventory_ready_generation_;
    int64_t                                                  inventory_readiness_generation_ = 0;
    bool                                                     inventory_fenced_               = false;
    bool supportsPrefixRankReplication(const ParallelismConfig& parallelism) const;
    void confirmPrefixInventoryLocked(const BlockCache::CacheSnapshot& candidates, bool all_ranks_ready);
    bool prefixInventoryItemReadyLocked(const BlockCache::CacheItem& item) const;
    bool prefixTopologyChangeDrainedLocked(uint64_t generation) const;
    bool residentPrefixTargetValidLocked(uint64_t generation, const ParallelismConfig& parallelism, bool merge) const;
    bool stageResidentPrefixMappingLocked(bool                                       leader,
                                          const std::function<void(torch::Tensor&)>& broadcast,
                                          const std::function<bool(bool)>&           all_ready);
    bool prepareResidentPrefixSplit(uint64_t                                   generation,
                                    const ParallelismConfig&                   parallelism,
                                    const std::function<void(torch::Tensor&)>& broadcast,
                                    const std::function<bool(bool)>&           all_ready);
    bool completePrefixTopologyChange(uint64_t                                   generation,
                                      const ParallelismConfig&                   parallelism,
                                      const std::function<void(torch::Tensor&)>& broadcast,
                                      const std::function<bool(bool)>&           all_ready);
    bool
    finishPrefixTopologyChangeLocked(uint64_t generation, const ParallelismConfig& parallelism, bool discard_mapping);
    void initConnectorCoordinator();
    void allocateAndSync();
    void reportMetricsLoop();

    // 成员变量
    CacheConfig         config_;
    KVCacheAllocatorPtr allocator_;

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const KVCacheConfig                kv_cache_config_;
    ParallelismConfig                  parallelism_config_;
    const RuntimeConfig                runtime_config_;
    const SpeculativeExecutionConfig   sp_config_;
    const PDSepConfig                  pd_sep_config_;
    const CacheStoreConfig             cache_store_config_;

    std::atomic<bool> stop_{false};
    std::thread       metrics_reporter_thread_;

    std::shared_ptr<KVCacheConnectorCoordinator> coordinator_;

    mutable std::mutex          cache_store_mutex_;
    std::shared_ptr<CacheStore> cache_store_;
};

}  // namespace rtp_llm
