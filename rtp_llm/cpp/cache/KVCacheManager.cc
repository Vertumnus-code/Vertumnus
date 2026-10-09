#include "rtp_llm/cpp/cache/KVCacheManager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"
#include "rtp_llm/cpp/cache/PrefixRankInstallQueue.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

KVCacheManager::KVCacheManager(const CacheConfig&                 config,
                               bool                               warmup,
                               const kmonitor::MetricsReporterPtr metrics_reporter,
                               const KVCacheConfig&               kv_cache_config,
                               const ParallelismConfig&           parallelism_config,
                               const RuntimeConfig&               runtime_config,
                               const SpeculativeExecutionConfig&  sp_config,
                               const PDSepConfig&                 pd_sep_config,
                               const CacheStoreConfig&            cache_store_config):
    config_(config),
    metrics_reporter_(metrics_reporter),
    kv_cache_config_(kv_cache_config),
    parallelism_config_(parallelism_config),
    runtime_config_(runtime_config),
    sp_config_(sp_config),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {
    prefix_admission_touch_blocks_ = autil::EnvUtil::getEnv("PREFIX_CACHE_ADMISSION_TOUCH_BLOCKS", 0);
    RTP_LLM_CHECK_WITH_INFO(prefix_admission_touch_blocks_ >= 0 && prefix_admission_touch_blocks_ <= 8192,
                            "PREFIX_CACHE_ADMISSION_TOUCH_BLOCKS must be between 0 and 8192");
    std::ifstream epoch_source("/proc/sys/kernel/random/uuid");
    std::getline(epoch_source, inventory_epoch_);
    inventory_boot_epoch_ = inventory_epoch_;
    if (warmup) {
        config_.block_num = 1;
    } else {
        allocateAndSync();
    }

    RTP_LLM_LOG_INFO("cache config: layer_num=%d, block_num=%d, block_size=%dB, seq_size_per_block=%zu",
                     config_.layer_num,
                     config_.block_num,
                     config_.block_size_bytes,
                     config_.seq_size_per_block);
}

KVCacheManager::~KVCacheManager() {
    stop_.store(true, std::memory_order_relaxed);
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
    prefix_rank_install_queue_.reset();
    prefix_transaction_manager_.reset();
    allocator_.reset();
    coordinator_.reset();
}

std::vector<BlockInfo> KVCacheManager::prefixBlockBuffers(int block_index) const {
    if (!allocator_ || block_index <= 0 || static_cast<size_t>(block_index) >= config_.block_num) {
        return {};
    }
    std::vector<BlockInfo> buffers;
    for (uint32_t layer_id = 0; layer_id < config_.layer_all_num; ++layer_id) {
        auto layer_buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer_id), block_index);
        buffers.insert(buffers.end(), layer_buffers.begin(), layer_buffers.end());
    }
    return buffers;
}

bool KVCacheManager::supportsPrefixRankReplication() const {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    return supportsPrefixRankReplication(parallelism_config_);
}

bool KVCacheManager::supportsPrefixRankReplication(const ParallelismConfig& parallelism) const {
    const auto method = parallelism.prefill_cp_config.method;
    return parallelism.pp_size == 1 && !parallelism.ffn_disaggregate_config.enable_ffn_disaggregate
           && config_.mtp_sub_configs.empty() && config_.cache_specs.size() == 1 && config_.cache_specs[0]
           && config_.cache_specs[0]->type == KVCacheSpecType::MultiHeadAttention
           && dynamic_cast<SingleTypeKVCacheAllocator*>(allocator_.get()) != nullptr
           && (parallelism.tp_size == 1 || method == CPRotateMethod::ALL_GATHER
               || method == CPRotateMethod::ALL_GATHER_WITH_OVERLAP);
}

bool KVCacheManager::prefixRankInstallPending() const {
    return prefix_rank_install_queue_ && prefix_rank_install_queue_->pending();
}

void KVCacheManager::processPrefixRankInstalls() {
    RTP_LLM_PROFILE_SCOPE("prefix.rank_install");
    ParallelismConfig parallelism;
    bool              supported;
    {
        std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
        parallelism = parallelism_config_;
        supported   = supportsPrefixRankReplication(parallelism);
    }
    PrefixRankInstallQueue::Collective collective{
        [](torch::Tensor& tensor) {
            execBroadcast({{tensor}, 0});
            execSyncCommunication(false);
            runtimeSyncAndCheck();
        },
        [ranks = parallelism.tp_size](bool ready) {
            auto value = torch::full({1}, ready ? 1 : 0, torch::kInt32);
            value      = execAllReduce({value, ReduceOp::Sum, false, ParallelMode::TP}).buffer;
            return value.item<int>() == ranks;
        },
        [] { runtimeSyncAndCheck(); },
        [](const std::vector<torch::Tensor>& buffers) {
            execBroadcast({buffers, 0});
            execSyncCommunication(false);
        }};
    prefix_rank_install_queue_->process(
        parallelism.tp_rank,
        parallelism.tp_size,
        supported,
        [this](BlockIdxType block) { return prefixBlockBuffers(block); },
        collective);
}

// 初始化和配置相关

bool KVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "cache specs must not be empty");

    const bool is_hybrid = config_.groupNums() > 1;
    if (is_hybrid) {
        allocator_ = std::make_shared<rtp_llm::HybridTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "HybridTypeKVCacheAllocator init failed");
    } else {
        allocator_ = std::make_shared<rtp_llm::SingleTypeKVCacheAllocator>(
            config_, AllocationType::DEVICE, metrics_reporter_, kv_cache_config_.reserve_block_ratio);
        RTP_LLM_CHECK_WITH_INFO(allocator_->init(), "SingleTypeKVCacheAllocator init failed");
    }

    PrefixCacheTransactionManager::LifecycleOptions prefix_options;
    prefix_options.pinned_import_commit = autil::EnvUtil::getEnv("RTP_LLM_PREFIX_PINNED_IMPORT_COMMIT", false);
    RTP_LLM_LOG_INFO("prefix pinned import commit enabled=%d", prefix_options.pinned_import_commit);
    prefix_transaction_manager_ = std::make_unique<PrefixCacheTransactionManager>(
        allocator_->getBlockPool(),
        allocator_->getBlockPool()->blockCache(),
        [this](BlockIdxType source, BlockIdxType destination) {
            try {
                blockCopy(source, destination);
            } catch (...) {
                runtimeSyncAndCheck();
                throw;
            }
            runtimeSyncAndCheck();
        },
        prefix_options);

    prefix_rank_install_queue_ =
        std::make_unique<PrefixRankInstallQueue>(*prefix_transaction_manager_, 64, 32, 32 * 1024 * 1024, 128);

    if (metrics_reporter_) {
        stop_.store(false, std::memory_order_relaxed);
        metrics_reporter_thread_ = std::thread(&KVCacheManager::reportMetricsLoop, this);
    }

    initConnectorCoordinator();
    return true;
}

const CacheConfig& KVCacheManager::cacheConfig() const {
    return config_;
}

const CacheConfig& KVCacheManager::getMTPModuleCacheConfig(int mtp_module_id) const {
    return *config_.mtp_sub_configs[mtp_module_id];
}

// 显存管理和缓存分配

MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids);

    const int seq_size_per_block = config_.seq_size_per_block;
    if (!malloc_info.batch_kv_cache_resource->curBlocksNum()) {
        initCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    } else {
        updateCacheKeys(malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids, seq_size_per_block);
    }

    return allocator_->malloc(malloc_info);
}

void KVCacheManager::free(const FreeInfo& free_info) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK(free_info.batch_kv_cache_resource && free_info.complete_token_ids);
    allocator_->free(free_info);
}

void KVCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    RTP_LLM_PROFILE_FUNCTION();
    dropLastPartialBlock(insert_info.batch_kv_cache_resource);
    allocator_->insertIntoCache(insert_info);
}

int KVCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                          int                            seq_len,
                                          int                            reserve_step) const {
    return allocator_->singleBatchNeedBlocks(batch_kv_cache_resource, seq_len, reserve_step);
}

// 块操作相关

void KVCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    return allocator_->blockCopy(src_block_index, dest_block_index);
}

void KVCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    return allocator_->blockBatchCopy(copy_mapping);
}

void KVCacheManager::blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) {
    return allocator_->blockBatchCopy(copy_mapping_begin, copy_mapping_end);
}

bool KVCacheManager::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                   const std::vector<int>&        block_src_batch,
                                   bool                           copy_last_block,
                                   std::vector<BlockIdPair>&      block_update_mapping) {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_->updateKVBlock(batch_kv_cache_resource, block_src_batch, copy_last_block, block_update_mapping);
}

// Write one KV block (optionally per-layer) from host/device tensors for test
bool KVCacheManager::setKVBlockValue(int                  block_index,
                                     int                  layer_id,
                                     const torch::Tensor& k_buffer,
                                     const torch::Tensor& v_buffer) {
    // Basic size/type validation to prevent out-of-bounds copy
    auto&  spec             = config_.cache_specs[0];
    size_t expected_k_bytes = spec->k_block_size_bytes();
    size_t expected_v_bytes = spec->v_block_size_bytes();
    size_t src_k_bytes      = k_buffer.nbytes();
    size_t src_v_bytes      = v_buffer.nbytes();
    if (src_k_bytes < expected_k_bytes || src_v_bytes < expected_v_bytes) {
        RTP_LLM_LOG_ERROR("setKVBlockValue src bytes too small: k[%zu]<[%zu] or v[%zu]<[%zu]",
                          src_k_bytes,
                          expected_k_bytes,
                          src_v_bytes,
                          expected_v_bytes);
        return false;
    }

    auto dst = allocator_->convertIndexToBuffer(layer_id, block_index);
    RTP_LLM_CHECK_WITH_INFO(
        !dst.empty(), "convertIndexToBuffer returned empty for layer %d, block %d", layer_id, block_index);
    if (!dst[0].addr) {
        RTP_LLM_LOG_ERROR("convertIndexToBuffer returned null for layer %d, block %d", layer_id, block_index);
        return false;
    }

    auto copyFunc = [&](const torch::Tensor& src_tensor, const BlockInfo& dst_block, size_t dst_byte_offset) -> bool {
        const size_t dst_bytes = dst_block.size_bytes;
        const size_t src_bytes = src_tensor.nbytes();
        if (dst_bytes < dst_byte_offset + src_bytes) {
            RTP_LLM_LOG_ERROR("dst block bytes[%zu] < dst_offset[%zu] + src bytes[%zu] in setKVBlockValue(layer=%d)",
                              dst_bytes,
                              dst_byte_offset,
                              src_bytes,
                              layer_id);
            return false;
        }

        auto* dst_ptr    = static_cast<char*>(dst_block.addr) + dst_byte_offset;
        auto  dst_device = dst_block.is_cuda ? torch::kCUDA : torch::kCPU;
        auto  src_device = src_tensor.is_cuda() ? torch::kCUDA : torch::kCPU;
        auto  dst_t      = torch::from_blob(
            dst_ptr, {(int64_t)src_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dst_device));
        auto src_t = torch::from_blob(src_tensor.data_ptr(),
                                      {(int64_t)src_bytes},
                                      torch::TensorOptions().dtype(torch::kUInt8).device(src_device));
        dst_t.copy_(src_t);
        return true;
    };

    if (!copyFunc(k_buffer, dst[0], 0)) {
        return false;
    }

    if (!copyFunc(v_buffer, dst[0], expected_k_bytes)) {
        return false;
    }

    cudaSyncAndCheck();
    return true;
}

bool KVCacheManager::setKVBlockValue(int block_index, const torch::Tensor& k_buffer, const torch::Tensor& v_buffer) {
    if (block_index < 0 || block_index >= config_.block_num) {
        RTP_LLM_LOG_WARNING("Invalid block_index: %d, valid range: [0, %d)", block_index, config_.block_num);
        return false;
    }

    bool all_success = true;
    for (int layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        all_success = setKVBlockValue(block_index, layer_id, k_buffer, v_buffer) && all_success;
    }
    return all_success;
}

// 地址转换和缓冲区访问

BlockAddrInfo KVCacheManager::convertIndexToAddr(int block_index, int layer_id) const {
    return allocator_->convertIndexToAddr(layer_id, block_index);
}

std::vector<BlockInfo> KVCacheManager::convertIndexToBuffer(int block_index, int layer_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index);
}

std::vector<BlockInfo>
KVCacheManager::convertIndexToBuffer(int block_index, int layer_id, int partition_count, int partition_id) const {
    return allocator_->convertIndexToBuffer(layer_id, block_index, partition_count, partition_id);
}

CacheLayerLayout KVCacheManager::allLayerCacheBase() const {
    return allocator_->allLayerCacheBase();
}

CacheLayerLayout KVCacheManager::getMainModelCacheLayerLayout() const {
    CacheLayerLayout layout;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(config_.layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(config_.layer_num);
    }

    layout.layer_to_groups = config_.layer_to_group_id;
    layout.group_types     = config_.group_types;
    layout.layer_attn_types.resize(config_.layer_num, CacheGroupType::FULL);

    RTP_LLM_CHECK_WITH_INFO(config_.layer_num <= all_layer_tensors.size(),
                            "config_.layer_num[%d] > all_layer_tensors.size()[%ld]",
                            config_.layer_num,
                            all_layer_tensors.size());

    for (int layer_id = 0; layer_id < static_cast<int>(config_.layer_num); ++layer_id) {
        if (static_cast<size_t>(layer_id) < all_layer_tensors.size()) {
            layout.layer_to_groups[layer_id]          = all_layout.layer_to_groups[layer_id];
            layout.layers_to_kv_buffer_ptrs[layer_id] = all_layer_tensors[layer_id];
        } else {
            RTP_LLM_CHECK(false);
        }

        if (!all_scale_tensors.empty()) {
            if (static_cast<size_t>(layer_id) < all_scale_tensors.size()) {
                layout.layers_to_scale_buffer_ptrs[layer_id] = all_scale_tensors[layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }
        }
        if (static_cast<size_t>(layer_id) < config_.layer_attn_types.size()) {
            layout.layer_attn_types[layer_id] = config_.layer_attn_types[static_cast<size_t>(layer_id)];
        }
    }

    return layout;
}

CacheLayerLayout KVCacheManager::getMTPModuleCacheLayerLayout(int mtp_module_id) const {
    CacheLayerLayout layout;

    RTP_LLM_CHECK_WITH_INFO(mtp_module_id >= 0 && static_cast<size_t>(mtp_module_id) < config_.mtp_sub_configs.size(),
                            "Invalid mtp_module_id: %d, must be in range [0, %zu)",
                            mtp_module_id,
                            config_.mtp_sub_configs.size());

    const auto& mtp_sub_config = config_.mtp_sub_configs[mtp_module_id];
    RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%d] is null", mtp_module_id);
    RTP_LLM_CHECK_WITH_INFO(
        !mtp_sub_config->global_layer_ids.empty(), "mtp_sub_configs[%d]->global_layer_ids is empty", mtp_module_id);
    RTP_LLM_CHECK_WITH_INFO(!mtp_sub_config->global_layer_ids[0].empty(),
                            "mtp_sub_configs[%d]->global_layer_ids[0] is empty",
                            mtp_module_id);

    const auto&    mtp_global_layer_ids = mtp_sub_config->global_layer_ids[0];
    const uint32_t mtp_layer_num        = mtp_sub_config->layer_num;

    auto  all_layout        = allocator_->allLayerCacheBase();
    auto& all_layer_tensors = all_layout.layers_to_kv_buffer_ptrs;
    auto& all_scale_tensors = all_layout.layers_to_scale_buffer_ptrs;

    layout.layer_to_groups.resize(mtp_layer_num);
    layout.layers_to_kv_buffer_ptrs.resize(mtp_layer_num);
    if (!all_scale_tensors.empty()) {
        layout.layers_to_scale_buffer_ptrs.resize(mtp_layer_num);
    }
    layout.layer_attn_types.resize(mtp_layer_num, CacheGroupType::FULL);

    for (uint32_t local_layer_id = 0; local_layer_id < mtp_layer_num; ++local_layer_id) {
        if (local_layer_id < mtp_global_layer_ids.size()) {
            const int global_layer_id = mtp_global_layer_ids[local_layer_id];

            if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_layer_tensors.size()) {
                layout.layer_to_groups[local_layer_id]          = all_layout.layer_to_groups[global_layer_id];
                layout.layers_to_kv_buffer_ptrs[local_layer_id] = all_layer_tensors[global_layer_id];
            } else {
                RTP_LLM_CHECK(false);
            }

            if (!all_scale_tensors.empty()) {
                if (global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < all_scale_tensors.size()) {
                    layout.layers_to_scale_buffer_ptrs[local_layer_id] = all_scale_tensors[global_layer_id];
                } else {
                    RTP_LLM_CHECK(false);
                }
            }
            if (local_layer_id < mtp_sub_config->layer_attn_types.size()) {
                layout.layer_attn_types[local_layer_id] = mtp_sub_config->layer_attn_types[local_layer_id];
            }
        } else {
            RTP_LLM_CHECK(false);
        }
    }

    return layout;
}

// 资源统计和信息查询

size_t KVCacheManager::freeBlocksNum() const {
    return allocator_->freeBlocksNum();
}

size_t KVCacheManager::availableBlocksNum() const {
    return allocator_->availableBlocksNum();
}

size_t KVCacheManager::notInUseBlocksNum() const {
    return allocator_->notInUseBlocksNum();
}

BatchKVCacheResourcePtr KVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    return allocator_->popBlocksFromCache(min_blocks_to_free);
}

void KVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    allocator_->blockCacheFree(batch_kv_cache_resource);
}

size_t KVCacheManager::availableTokensNum() const {
    return allocator_->availableTokensNum();
}

size_t KVCacheManager::totalBlocksNum() const {
    return allocator_->totalBlocksNum();
}

size_t KVCacheManager::maxAvailableTokensNum() const {
    return allocator_->maxAvailableTokensNum();
}

KVCacheInfo KVCacheManager::getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const {
    RTP_LLM_PROFILE_SCOPE("cache.status.total");
    KVCacheInfo info;

    if (!allocator_) {
        RTP_LLM_LOG_ERROR("getKVCacheInfo called before KVCacheManager initialized");
        info.version = latest_version;
        return info;
    }

    if (need_cache_keys) {
        std::unordered_set<CacheKeyType> all_keys;
        // device cache keys
        auto block_cache = allocator_->getBlockPool()->blockCache();
        auto snapshot    = block_cache->cacheSnapshot(latest_version);
        for (const auto& cacheItem : snapshot.values) {
            all_keys.insert(cacheItem.cache_key);
        }
        // memory cache keys
        const auto mem_cache_keys = coordinator_->memoryCacheKeys();
        all_keys.insert(mem_cache_keys.begin(), mem_cache_keys.end());

        info.cached_keys.assign(all_keys.begin(), all_keys.end());
        info.version = snapshot.version;
    } else if (prefixInventoryEnabled()) {
        info.version = allocator_->getBlockPool()->blockCache()->cacheVersion();
    }

    const size_t block_size_tokens = config_.seq_size_per_block;
    const size_t total_blocks      = allocator_->totalBlocksNum();
    const size_t available_blocks  = allocator_->availableBlocksNum();

    info.block_size          = block_size_tokens;
    info.total_kv_cache      = total_blocks * block_size_tokens;
    info.available_kv_cache  = available_blocks * block_size_tokens;
    const char* namespace_id = std::getenv("PREFIX_CACHE_NAMESPACE");
    const char* worker_id    = std::getenv("PREFIX_CACHE_WORKER_ID");
    if (prefixInventoryEnabled()) {
        RTP_LLM_PROFILE_SCOPE("prefix.inventory_report");
        std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
        if (inventory_epoch_.empty()) {
            return info;
        }
        PrefixInventoryReport report;
        report.namespace_id = namespace_id;
        report.worker_id    = worker_id;
        report.dp_lane      = parallelism_config_.dp_rank;
        report.epoch        = inventory_epoch_;
        report.cp_degree    = parallelism_config_.prefill_cp_config.is_enabled() ? parallelism_config_.tp_size : 1;
        report.block_size   = config_.seq_size_per_block;
        report.layout_fingerprint        = config_.debugString() + parallelism_config_.to_string();
        report.physical_free_bytes       = freeBlocksNum() * config_.block_size_bytes * parallelism_config_.tp_size;
        report.physical_block_bytes      = config_.block_size_bytes * parallelism_config_.tp_size;
        report.rank_readiness_generation = inventory_readiness_generation_;
        report.device_ready = !inventory_fenced_ && parallelism_config_.tp_rank == 0 && parallelism_config_.pp_size == 1
                              && !parallelism_config_.ffn_disaggregate_config.enable_ffn_disaggregate
                              && config_.mtp_sub_configs.empty() && config_.cache_specs.size() == 1
                              && config_.cache_specs[0]
                              && config_.cache_specs[0]->type == KVCacheSpecType::MultiHeadAttention
                              && dynamic_cast<SingleTypeKVCacheAllocator*>(allocator_.get()) != nullptr;
        auto       block_pool  = allocator_->getBlockPool();
        auto       block_cache = block_pool->blockCache();
        const auto revision    = block_cache->observePrefixSnapshotView(
            [&](const std::shared_ptr<const BlockCache::CacheSnapshot>& snapshot) {
                RTP_LLM_PROFILE_SCOPE("prefix.inventory_report.snapshot");
                if (report.device_ready) {
                    const auto memory  = block_pool->memorySnapshotUnderCacheLock(*snapshot);
                    const auto maximum = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
                    const auto ranks   = static_cast<uint64_t>(parallelism_config_.tp_size);
                    if (ranks > 0 && config_.block_size_bytes > 0 && config_.block_size_bytes <= maximum / ranks) {
                        const auto bytes = config_.block_size_bytes * ranks;
                        if (memory.capacity_blocks <= maximum / bytes) {
                            report.physical_capacity_bytes     = memory.capacity_blocks * bytes;
                            report.physical_free_bytes         = memory.free_blocks * bytes;
                            report.evictable_bytes             = memory.evictable_blocks * bytes;
                            report.active_bytes                = memory.active_blocks * bytes;
                            report.resident_bytes              = memory.resident_blocks * bytes;
                            report.operation_reserved_bytes    = memory.operation_reserved_blocks * bytes;
                            report.physical_memory_stats_valid = true;
                            report.copy_lifetime               = block_cache->copyLedger()->snapshot();
                        }
                    }
                }
                const bool cached_keys = inventory_ready_items_.empty() && inventory_keys_snapshot_ == snapshot
                                         && inventory_keys_ready_generation_ == inventory_ready_generation_
                                         && inventory_keys_epoch_ == inventory_epoch_
                                         && inventory_keys_ready_ == report.device_ready;
                if (cached_keys) {
                    report.device_keys = inventory_keys_;
                } else {
                    if (report.device_ready) {
                        for (const auto& item : snapshot->values) {
                            if (item.group_id == 0 && prefixInventoryItemReadyLocked(item)) {
                                report.device_keys.push_back(static_cast<int64_t>(item.cache_key));
                            }
                        }
                    }
                    std::sort(report.device_keys.begin(), report.device_keys.end());
                    if (inventory_ready_items_.empty()) {
                        inventory_keys_snapshot_         = snapshot;
                        inventory_keys_ready_generation_ = inventory_ready_generation_;
                        inventory_keys_epoch_            = inventory_epoch_;
                        inventory_keys_ready_            = report.device_ready;
                        inventory_keys_                  = report.device_keys;
                    } else {
                        inventory_keys_snapshot_.reset();
                        inventory_keys_.clear();
                    }
                }
                info.inventory          = prefix_inventory_.snapshotSorted(std::move(report));
                const bool changed      = info.inventory.version != inventory_view_version_;
                inventory_view_version_ = info.inventory.version;
                return changed;
            },
            true);  // Reporting is observational; readiness revocation fences at its mutation site.
        info.inventory.version = static_cast<int64_t>(revision);
    }
    // cached_keys left empty for now; can be populated when distributed cache is wired up.

    return info;
}

void KVCacheManager::recordPrefixReuse(int64_t            request_id,
                                       int64_t            device_reuse_tokens,
                                       int64_t            input_tokens,
                                       const std::string& attempt_id) {
    if (prefixInventoryEnabled()) {
        RTP_LLM_PROFILE_SCOPE("prefix.reuse_event");
        prefix_inventory_.record(request_id, device_reuse_tokens, input_tokens, attempt_id);
    }
}

std::unique_ptr<PrefixCopyLedger::Use> KVCacheManager::capturePrefixCopyUse(const BatchKVCacheResource& resource,
                                                                            int64_t                     reuse_tokens,
                                                                            int64_t                     request_id,
                                                                            const std::string&          attempt_id) {
    if (!prefixInventoryEnabled() || reuse_tokens <= 0) {
        return nullptr;
    }
    RTP_LLM_PROFILE_SCOPE("prefix.copy_usage_capture");
    auto ledger = allocator_->getBlockPool()->blockCache()->copyLedger();
    if (config_.seq_size_per_block == 0 || resource.batchSize() == 0 || resource.groupNums() != 1) {
        ledger->invalidate();
        return nullptr;
    }
    const auto       count = static_cast<size_t>(reuse_tokens / config_.seq_size_per_block);
    BlockIndicesType blocks;
    for (int batch = 0; batch < resource.batchSize(); ++batch) {
        const auto& indices = resource.blocks(batch);
        if (count > indices.size()) {
            ledger->invalidate();
            return nullptr;
        }
        blocks.insert(blocks.end(), indices.begin(), indices.begin() + count);
    }
    return ledger->capture(blocks, request_id, attempt_id);
}

PrefixTopologyIdentity KVCacheManager::prefixTopologyIdentity() const {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    return {inventory_boot_epoch_, parallelism_config_.world_rank, parallelism_config_.world_size};
}

bool KVCacheManager::prefixInventoryEnabled() const {
    const char* namespace_id = std::getenv("PREFIX_CACHE_NAMESPACE");
    const char* worker_id    = std::getenv("PREFIX_CACHE_WORKER_ID");
    return namespace_id && namespace_id[0] != '\0' && worker_id && worker_id[0] != '\0';
}

size_t KVCacheManager::touchPrefixAtAdmission(const CompleteTokenIdsPtr& tokens) {
    if (prefix_admission_touch_blocks_ == 0 || !prefixInventoryEnabled() || !tokens || tokens->batchSize() != 1
        || tokens->seqLength() <= 1) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (!allocator_ || inventory_fenced_ || inventory_topology_generation_ != 0 || !inventory_ready_generation_
        || parallelism_config_.dp_size != 1 || parallelism_config_.tp_rank != 0
        || pd_sep_config_.role_type == RoleType::DECODE || !supportsPrefixRankReplication(parallelism_config_)
        || config_.seq_size_per_block == 0) {
        return 0;
    }
    const size_t  block_size = config_.seq_size_per_block;
    const size_t  blocks     = std::min(static_cast<size_t>(prefix_admission_touch_blocks_),
                                   (static_cast<size_t>(tokens->seqLength()) - 1) / block_size);
    CacheKeysType keys;
    keys.reserve(blocks);
    int64_t rolling_hash = 0;
    auto*   token_ids    = tokens->data(0);
    for (size_t index = 0; index < blocks; ++index) {
        const size_t offset = index * block_size;
        rolling_hash        = hashInt64Array(rolling_hash, token_ids + offset, token_ids + offset + block_size);
        keys.push_back(rolling_hash);
    }
    return allocator_->getBlockPool()->blockCache()->touchReadyPrefix(keys, *inventory_ready_generation_);
}

BlockCache::CacheSnapshot KVCacheManager::prefixInventoryCandidates() const {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_candidates");
    return allocator_->getBlockPool()->blockCache()->cacheSnapshot(-1);
}

std::shared_ptr<const BlockCache::CacheSnapshot> KVCacheManager::prefixInventorySnapshot() const {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_candidates");
    return allocator_->getBlockPool()->blockCache()->prefixInventorySnapshot();
}

KVCacheManager::PrefixInventoryBoundary KVCacheManager::prefixInventoryBoundary() const {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_candidates");
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    auto                        cache = allocator_->getBlockPool()->blockCache();
    return {cache, inventory_epoch_, cache->inventoryGeneration()};
}

void KVCacheManager::confirmPrefixInventory(const PrefixInventoryBoundary& boundary, bool all_ranks_ready) {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_confirm");
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    const auto                  cache = allocator_->getBlockPool()->blockCache();
    if (boundary.cache != cache || boundary.epoch != inventory_epoch_) {
        return;
    }
    if (boundary.generation > cache->inventoryGeneration()) {
        throw std::invalid_argument("prefix inventory boundary exceeds current generation");
    }
    std::optional<uint64_t> ready_generation;
    if (all_ranks_ready && !inventory_fenced_) {
        ready_generation = boundary.generation;
        ++inventory_readiness_generation_;
    }
    if (inventory_ready_generation_.value_or(0) != ready_generation.value_or(0) || !inventory_ready_items_.empty()) {
        const bool monotonic_promotion = inventory_ready_generation_ && ready_generation
                                         && *ready_generation >= *inventory_ready_generation_
                                         && inventory_ready_items_.empty();
        cache->invalidatePrefixSnapshot(monotonic_promotion);
    }
    inventory_ready_generation_ = ready_generation;
    inventory_ready_items_.clear();
    inventory_ready_snapshot_.reset();
}

bool KVCacheManager::prefixInventoryItemReadyLocked(const BlockCache::CacheItem& item) const {
    if (inventory_ready_generation_) {
        return item.inventory_generation > 0 && item.inventory_generation <= *inventory_ready_generation_;
    }
    const auto confirmed = inventory_ready_items_.find(item.cache_key);
    return confirmed != inventory_ready_items_.end() && confirmed->second == item.inventory_generation;
}

void KVCacheManager::confirmPrefixInventory(const std::shared_ptr<const BlockCache::CacheSnapshot>& candidates,
                                            bool                                                    all_ranks_ready) {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_confirm");
    if (!candidates) {
        throw std::invalid_argument("prefix inventory candidates must not be null");
    }
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (all_ranks_ready && !inventory_fenced_ && inventory_ready_snapshot_ == candidates) {
        ++inventory_readiness_generation_;
        return;
    }
    inventory_ready_snapshot_.reset();
    confirmPrefixInventoryLocked(*candidates, all_ranks_ready);
    if (all_ranks_ready && !inventory_fenced_) {
        inventory_ready_snapshot_ = candidates;
    }
}

void KVCacheManager::confirmPrefixInventory(const BlockCache::CacheSnapshot& candidates, bool all_ranks_ready) {
    RTP_LLM_PROFILE_SCOPE("prefix.inventory_confirm");
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    inventory_ready_snapshot_.reset();
    confirmPrefixInventoryLocked(candidates, all_ranks_ready);
}

void KVCacheManager::confirmPrefixInventoryLocked(const BlockCache::CacheSnapshot& candidates, bool all_ranks_ready) {
    decltype(inventory_ready_items_) ready_items;
    if (all_ranks_ready && !inventory_fenced_) {
        for (const auto& item : candidates.values) {
            if (item.group_id == 0) {
                ready_items[item.cache_key] = item.inventory_generation;
            }
        }
        ++inventory_readiness_generation_;
    }
    if (inventory_ready_generation_ || ready_items != inventory_ready_items_) {
        allocator_->getBlockPool()->blockCache()->observePrefixSnapshot([&](const BlockCache::CacheSnapshot& snapshot) {
            for (const auto& item : snapshot.values) {
                if (item.group_id != 0) {
                    continue;
                }
                const auto next      = ready_items.find(item.cache_key);
                const bool was_ready = prefixInventoryItemReadyLocked(item);
                const bool now_ready = next != ready_items.end() && next->second == item.inventory_generation;
                if (was_ready != now_ready) {
                    return true;
                }
            }
            return false;
        });
        inventory_ready_items_ = std::move(ready_items);
    }
    inventory_ready_generation_.reset();
}

void KVCacheManager::fencePrefixInventory() {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (inventory_fenced_ && inventory_topology_generation_ == 0) {
        return;
    }
    inventory_fenced_ = true;
    allocator_->getBlockPool()->blockCache()->copyLedger()->invalidate();
    inventory_topology_generation_ = 0;
    topology_source_ready_items_.clear();
    topology_resident_items_.clear();
    topology_resident_generation_     = 0;
    topology_resident_tp_size_        = 0;
    topology_resident_merge_          = false;
    topology_resident_repair_started_ = false;
    allocator_->getBlockPool()->blockCache()->invalidatePrefixSnapshot();
    inventory_ready_items_.clear();
    inventory_ready_snapshot_.reset();
    inventory_ready_generation_.reset();
    inventory_epoch_ += ":fenced";
    if (prefix_rank_install_queue_) {
        prefix_rank_install_queue_->close();
    }
}

uint64_t KVCacheManager::beginPrefixTopologyChange() {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (inventory_topology_generation_ != 0) {
        return inventory_topology_generation_;
    }
    if (inventory_fenced_ || !prefix_transaction_manager_ || !prefix_rank_install_queue_
        || inventory_boot_epoch_.empty() || !supportsPrefixRankReplication(parallelism_config_)) {
        throw std::logic_error("prefix topology change requires an initialized supported cache");
    }
    topology_source_ready_items_.clear();
    if (parallelism_config_.tp_rank == 0) {
        for (const auto& item : allocator_->getBlockPool()->blockCache()->cacheSnapshot(-2).values) {
            if (prefixInventoryItemReadyLocked(item)) {
                topology_source_ready_items_.push_back(item);
            }
        }
    }
    inventory_topology_generation_ = prefix_transaction_manager_->beginTopologyChange();
    inventory_fenced_              = true;
    allocator_->getBlockPool()->blockCache()->invalidatePrefixSnapshot();
    inventory_ready_items_.clear();
    inventory_ready_snapshot_.reset();
    inventory_ready_generation_.reset();
    prefix_rank_install_queue_->close();
    return inventory_topology_generation_;
}

bool KVCacheManager::prefixTopologyChangeDrainedLocked(uint64_t generation) const {
    return generation != 0 && inventory_topology_generation_ == generation && inventory_fenced_
           && prefix_transaction_manager_->topologyChangeDrained(generation) && prefix_rank_install_queue_->drained()
           && allocator_->getBlockPool()->requestRefBlocksNum() == 0
           && allocator_->getBlockPool()->connectorRefBlocksNum() == 0;
}

bool KVCacheManager::prefixTopologyChangeDrained(uint64_t generation) const {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    return prefixTopologyChangeDrainedLocked(generation);
}

bool KVCacheManager::prepareResidentPrefixSplit(uint64_t generation, const ParallelismConfig& parallelism) {
    return prepareResidentPrefixSplit(
        generation,
        parallelism,
        [](torch::Tensor& tensor) {
            execBroadcast({{tensor}, 0});
            execSyncCommunication(false);
        },
        [ranks = parallelism_config_.tp_size](bool ready) {
            auto value = torch::full({1}, ready ? 1 : 0, torch::kInt32);
            return execAllReduce({value, ReduceOp::Sum, false, ParallelMode::TP}).buffer.item<int>() == ranks;
        });
}

bool KVCacheManager::residentPrefixSplitPrepared(int64_t tp_size) const {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    return !topology_resident_merge_ && topology_resident_tp_size_ == tp_size && topology_resident_generation_ != 0
           && prefixTopologyChangeDrainedLocked(topology_resident_generation_);
}

bool KVCacheManager::residentPrefixMergePrepared(int64_t tp_size) const {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    return topology_resident_merge_ && topology_resident_tp_size_ == tp_size && topology_resident_generation_ != 0
           && prefixTopologyChangeDrainedLocked(topology_resident_generation_);
}

bool KVCacheManager::prepareResidentPrefixMerge(uint64_t generation, const ParallelismConfig& parallelism) {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (!residentPrefixTargetValidLocked(generation, parallelism, true)) {
        return false;
    }
    // The old lanes still own independent KV. Only stage the mode before commit.
    topology_resident_merge_      = true;
    topology_resident_generation_ = generation;
    topology_resident_tp_size_    = parallelism.tp_size;
    return true;
}

bool KVCacheManager::residentPrefixTargetValidLocked(uint64_t                 generation,
                                                     const ParallelismConfig& parallelism,
                                                     bool                     merge) const {
    const bool adjacent =
        merge ? parallelism.tp_size > 0 && parallelism.tp_size % 2 == 0
                    && parallelism.tp_size / 2 == parallelism_config_.tp_size :
                parallelism_config_.tp_size % 2 == 0 && parallelism.tp_size == parallelism_config_.tp_size / 2;
    return prefixTopologyChangeDrainedLocked(generation) && supportsPrefixRankReplication(parallelism) && adjacent
           && parallelism.tp_size > 0 && parallelism.world_size == parallelism_config_.world_size
           && parallelism.world_rank == parallelism_config_.world_rank && parallelism.world_size > 0
           && parallelism.world_rank >= 0 && parallelism.world_rank < parallelism.world_size
           && parallelism.world_size % parallelism.tp_size == 0
           && parallelism.dp_size == parallelism.world_size / parallelism.tp_size
           && parallelism.tp_rank == parallelism.world_rank % parallelism.tp_size
           && parallelism.dp_rank == parallelism.world_rank / parallelism.tp_size;
}

bool KVCacheManager::prepareResidentPrefixSplit(uint64_t                                   generation,
                                                const ParallelismConfig&                   parallelism,
                                                const std::function<void(torch::Tensor&)>& broadcast,
                                                const std::function<bool(bool)>&           all_ready) {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (!all_ready(residentPrefixTargetValidLocked(generation, parallelism, false))
        || !stageResidentPrefixMappingLocked(parallelism_config_.tp_rank == 0, broadcast, all_ready)) {
        return false;
    }
    topology_resident_merge_      = false;
    topology_resident_generation_ = generation;
    topology_resident_tp_size_    = parallelism.tp_size;
    return true;
}

bool KVCacheManager::stageResidentPrefixMappingLocked(bool                                       leader,
                                                      const std::function<void(torch::Tensor&)>& broadcast,
                                                      const std::function<bool(bool)>&           all_ready) {
    if (leader) {
        const auto snapshot = allocator_->getBlockPool()->blockCache()->cacheSnapshot(-2);
        std::unordered_map<CacheKeyType, BlockCache::CacheItem> current;
        for (const auto& item : snapshot.values) {
            current.emplace(item.cache_key, item);
        }
        topology_source_ready_items_.erase(std::remove_if(topology_source_ready_items_.begin(),
                                                          topology_source_ready_items_.end(),
                                                          [&current](const auto& item) {
                                                              const auto found = current.find(item.cache_key);
                                                              return found == current.end()
                                                                     || found->second.block_index != item.block_index
                                                                     || found->second.inventory_generation
                                                                            != item.inventory_generation;
                                                          }),
                                           topology_source_ready_items_.end());
    }
    // Exchange only the confirmed leader mapping, never a union of lane indices.
    auto count = torch::full({1}, static_cast<int64_t>(topology_source_ready_items_.size()), torch::kInt64);
    broadcast(count);
    const auto size = count.item<int64_t>();
    if (!all_ready(size >= 0 && static_cast<uint64_t>(size) < config_.block_num)) {
        return false;
    }
    auto  metadata = torch::zeros({size, 7}, torch::kInt64);
    auto* data     = metadata.data_ptr<int64_t>();
    if (leader) {
        for (int64_t index = 0; index < size; ++index) {
            const auto& item = topology_source_ready_items_[index];
            auto*       row  = data + index * 7;
            row[0]           = item.cache_key;
            row[1]           = item.group_id;
            row[2]           = item.block_index;
            row[3]           = item.is_resident;
            row[4]           = item.prefix_depth;
            row[5]           = item.prefix_parent.has_value();
            row[6]           = item.prefix_parent.value_or(0);
        }
    }
    if (size > 0) {
        broadcast(metadata);
    }
    std::vector<BlockCache::CacheItem> items;
    std::unordered_set<CacheKeyType>   keys;
    std::unordered_set<BlockIdxType>   blocks;
    bool                               metadata_valid = true;
    for (int64_t index = 0; index < size; ++index) {
        const auto* row = data + index * 7;
        if (row[1] != 0 || row[2] <= 0 || static_cast<uint64_t>(row[2]) >= config_.block_num
            || (row[3] != 0 && row[3] != 1) || row[4] < 0 || (row[5] != 0 && row[5] != 1) || !keys.insert(row[0]).second
            || !blocks.insert(row[2]).second) {
            metadata_valid = false;
            break;
        }
        BlockCache::CacheItem item{row[0], 0, static_cast<BlockIdxType>(row[2]), row[3] != 0};
        item.prefix_depth = static_cast<size_t>(row[4]);
        if (row[5]) {
            item.prefix_parent = row[6];
        }
        items.push_back(item);
    }
    if (!all_ready(metadata_valid)) {
        return false;
    }
    topology_resident_items_ = std::move(items);
    return true;
}

bool KVCacheManager::finishPrefixTopologyChangeLocked(uint64_t                 generation,
                                                      const ParallelismConfig& parallelism,
                                                      bool                     discard_mapping) {
    if (!prefixTopologyChangeDrainedLocked(generation) || !supportsPrefixRankReplication(parallelism)
        || parallelism.world_size != parallelism_config_.world_size
        || parallelism.world_rank != parallelism_config_.world_rank || parallelism.tp_size <= 0
        || parallelism.dp_size <= 0 || parallelism.world_size <= 0 || parallelism.world_rank < 0
        || parallelism.world_rank >= parallelism.world_size || parallelism.world_size % parallelism.tp_size != 0
        || parallelism.dp_size != parallelism.world_size / parallelism.tp_size
        || parallelism.tp_rank != parallelism.world_rank % parallelism.tp_size
        || parallelism.dp_rank != parallelism.world_rank / parallelism.tp_size) {
        return false;
    }
    if (coordinator_ && !coordinator_->preparePrefixTopology(parallelism)) {
        return false;
    }
    const bool retain_mapping = discard_mapping && topology_resident_generation_ == generation;
    if (retain_mapping && parallelism.tp_size != topology_resident_tp_size_) {
        return false;
    }
    auto       next_epoch = inventory_boot_epoch_ + ":topology:" + std::to_string(generation);
    const auto pool       = allocator_->getBlockPool();
    const auto cache      = pool->blockCache();
    if (discard_mapping) {
        for (const auto& item : cache->cacheSnapshot(-2).values) {
            auto removed = cache->remove(item.cache_key, item.group_id);
            if (removed) {
                pool->blockCacheFree(removed->block_index);
            }
        }
        if (retain_mapping) {
            for (auto item : topology_resident_items_) {
                if (!cache->put(item, [pool](BlockIdxType block) { pool->blockCacheReference(block); })) {
                    return false;
                }
            }
        }
    }
    if (!prefix_rank_install_queue_->reopen()) {
        return false;
    }
    parallelism_config_             = parallelism;
    inventory_epoch_                = std::move(next_epoch);
    inventory_readiness_generation_ = 0;
    inventory_ready_items_.clear();
    inventory_ready_snapshot_.reset();
    inventory_ready_generation_.reset();
    cache->invalidatePrefixSnapshot();
    if (!prefix_transaction_manager_->resumeAfterTopologyChange(generation)) {
        prefix_rank_install_queue_->close();
        return false;
    }
    inventory_fenced_              = false;
    inventory_topology_generation_ = 0;
    if (retain_mapping) {
        inventory_ready_generation_     = cache->inventoryGeneration();
        inventory_readiness_generation_ = 1;
    }
    topology_source_ready_items_.clear();
    topology_resident_items_.clear();
    topology_resident_generation_     = 0;
    topology_resident_tp_size_        = 0;
    topology_resident_merge_          = false;
    topology_resident_repair_started_ = false;
    return true;
}

bool KVCacheManager::completePrefixTopologyChange(uint64_t generation, const ParallelismConfig& parallelism) {
    return completePrefixTopologyChange(
        generation,
        parallelism,
        [](torch::Tensor& tensor) {
            execBroadcast({{tensor}, 0});
            execSyncCommunication(false);
            runtimeSyncAndCheck();
        },
        [ranks = parallelism.tp_size](bool ready) {
            auto value = torch::full({1}, ready ? 1 : 0, torch::kInt32);
            return execAllReduce({value, ReduceOp::Sum, false, ParallelMode::TP}).buffer.item<int>() == ranks;
        });
}

bool KVCacheManager::completePrefixTopologyChange(uint64_t                                   generation,
                                                  const ParallelismConfig&                   parallelism,
                                                  const std::function<void(torch::Tensor&)>& broadcast,
                                                  const std::function<bool(bool)>&           all_ready) {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (topology_resident_merge_) {
        if (!all_ready(topology_resident_generation_ == generation && topology_resident_tp_size_ == parallelism.tp_size
                       && residentPrefixTargetValidLocked(generation, parallelism, true))) {
            return false;
        }
        const auto pool   = allocator_->getBlockPool();
        const auto layers = pool->allLayerCacheBase();
        const auto layout = config_.debugString();
        auto       shape  = torch::tensor(
            {static_cast<int64_t>(pool->getTotalSizeBytes()), static_cast<int64_t>(layout.size())}, torch::kInt64);
        const auto expected_shape = shape.clone();
        broadcast(shape);
        if (!all_ready(torch::equal(shape, expected_shape) && !layers.empty() && pool->getBaseAddress()
                       && pool->getTotalSizeBytes() > 0 && layout.size() < 1024 * 1024)) {
            return false;
        }
        auto description =
            torch::from_blob(const_cast<char*>(layout.data()), {static_cast<int64_t>(layout.size())}, torch::kUInt8)
                .clone();
        const auto expected_description = description.clone();
        broadcast(description);
        if (!all_ready(torch::equal(description, expected_description))
            || !stageResidentPrefixMappingLocked(parallelism.tp_rank == 0, broadcast, all_ready)) {
            return false;
        }
        // Commit has installed the new group. Repair conflicting block IDs from
        // its primary leader before publishing any ready inventory or resuming.
        auto bytes                        = torch::from_blob(pool->getBaseAddress(),
                                                             {static_cast<int64_t>(pool->getTotalSizeBytes())},
                                      torch::TensorOptions(torch::kUInt8).device(layers.front().device()));
        topology_resident_repair_started_ = true;
        broadcast(bytes);
        if (!all_ready(true)) {
            return false;
        }
    }
    return finishPrefixTopologyChangeLocked(generation, parallelism, true);
}

bool KVCacheManager::cancelPrefixTopologyChange(uint64_t generation) {
    std::lock_guard<std::mutex> lock(inventory_snapshot_mutex_);
    if (topology_resident_repair_started_) {
        return false;
    }
    return finishPrefixTopologyChangeLocked(generation, parallelism_config_, false);
}

// 系统资源管理

void KVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    allocator_->regUserMr(model_id, std::move(cache_store));
}

void KVCacheManager::setCacheStore(std::shared_ptr<CacheStore> cache_store) {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    cache_store_ = std::move(cache_store);
}

std::shared_ptr<CacheStore> KVCacheManager::getCacheStore() const {
    std::lock_guard<std::mutex> lock(cache_store_mutex_);
    return cache_store_;
}

bool KVCacheManager::hasActiveConnectors() const {
    return coordinator_ && coordinator_->hasActiveConnectors();
}

// PD separation: increment KV cache reference count
std::shared_ptr<KVCacheResource>
KVCacheManager::incrKVCacheRef(const KVCacheResource& resource, const CacheKeysType& cache_keys, bool is_connector) {
    return allocator_->incrKVCacheRef(resource, cache_keys, is_connector);
}

bool KVCacheManager::hasP2PConnector() const {
    return coordinator_ && coordinator_->hasP2PConnector();
}

// 异步连接器操作

std::shared_ptr<AsyncContext>
KVCacheManager::asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncRead(connector_context);
}

std::shared_ptr<AsyncContext>
KVCacheManager::asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    return coordinator_->asyncWrite(connector_context);
}

bool KVCacheManager::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    return coordinator_->executeFunction(request, response);
}

void KVCacheManager::initConnectorCoordinator() {
    RTP_LLM_LOG_INFO(
        "init connector coordinator, cache config: [%s], kv cache config: [%s], runtime config: [%s], parallelism config: [%s], sp config: [%s]",
        config_.debugString().c_str(),
        kv_cache_config_.to_string().c_str(),
        runtime_config_.to_string().c_str(),
        parallelism_config_.to_string().c_str(),
        sp_config_.to_string().c_str());
    coordinator_ = std::make_shared<KVCacheConnectorCoordinator>(config_,
                                                                 kv_cache_config_,
                                                                 runtime_config_,
                                                                 parallelism_config_,
                                                                 sp_config_,
                                                                 allocator_,
                                                                 metrics_reporter_,
                                                                 pd_sep_config_,
                                                                 cache_store_config_);
    RTP_LLM_CHECK_WITH_INFO(coordinator_->init(), "connector coordinator init failed");
}

void KVCacheManager::allocateAndSync() {
    size_t world_size = parallelism_config_.tp_size * parallelism_config_.dp_size;
    if (world_size > 1) {
        size_t local_rank    = parallelism_config_.tp_size * parallelism_config_.dp_rank + parallelism_config_.tp_rank;
        auto   block_num_t   = torch::empty({(int64_t)world_size}, torch::kInt32).pin_memory();
        auto   block_num_ptr = block_num_t.data_ptr<int>();
        block_num_ptr[local_rank] = config_.block_num;
        execAllGather({{block_num_t}, ParallelMode::DP_AND_TP});
        execSyncCommunication(false);
        cudaSyncAndCheck();

        if (parallelism_config_.ffn_disaggregate_config.is_ffn_service()) {
            config_.block_num = 1;
        } else {
            config_.block_num = *std::min_element(block_num_ptr, block_num_ptr + world_size);
        }
    }
    RTP_LLM_LOG_INFO("block_num is %d after tp sync", config_.block_num);
}

void KVCacheManager::reportMetricsLoop() {
    RTP_LLM_PROFILE_FUNCTION();
    kmonitor::MetricsTags tags;
    while (!stop_.load(std::memory_order_relaxed)) {
        if (!metrics_reporter_ || !allocator_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtpLLMCacheMetricsCollector collector;

        auto block_pool  = allocator_->getBlockPool();
        auto block_cache = block_pool ? block_pool->blockCache() : nullptr;

        const auto total_blocks         = allocator_->totalBlocksNum();
        const auto available_blocks     = allocator_->availableBlocksNum();
        const auto request_ref_blocks   = allocator_->requestRefBlocksNum();
        const auto connector_ref_blocks = allocator_->connectorRefBlocksNum();

        collector.kv_cache_item_num             = block_cache ? static_cast<int64_t>(block_cache->size()) : 0;
        collector.kv_cache_left_seq             = static_cast<int64_t>(available_blocks * config_.seq_size_per_block);
        collector.kv_cache_available_blocks     = static_cast<int64_t>(available_blocks);
        collector.kv_cache_request_ref_blocks   = static_cast<int64_t>(request_ref_blocks);
        collector.kv_cache_connector_ref_blocks = static_cast<int64_t>(connector_ref_blocks);
        collector.kv_cache_free_blocks          = static_cast<int64_t>(allocator_->freeBlocksNum());
        collector.kv_cache_used_ratio =
            (total_blocks == 0) ?
                0.0f :
                static_cast<float>(100.0 * (total_blocks - available_blocks) / static_cast<double>(total_blocks));
        collector.mr_cost_time_ms = allocator_->getMrCostTimeMs();

        metrics_reporter_->report<RtpLLMCacheMetrics, RtpLLMCacheMetricsCollector>(&tags, &collector);
        std::this_thread::sleep_for(std::chrono::seconds(1));  // 1s
    }
}

void KVCacheManager::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                P2PConnectorStartLoadResponsePB&      response,
                                std::function<bool()>                 is_cancelled) {
    if (coordinator_) {
        coordinator_->handleRead(request, response, is_cancelled);
    }
}

}  // namespace rtp_llm
