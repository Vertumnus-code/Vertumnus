#pragma once

#include <cstdint>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <string>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/PrefixTransferState.h"

namespace rtp_llm {

struct PrefixControllerIdentity {
    std::string namespace_id;
    std::string controller_id;
    uint64_t    term = 0;

    bool valid() const {
        return !namespace_id.empty() && namespace_id.size() <= 512 && !controller_id.empty()
               && controller_id.size() <= 128 && term != 0;
    }

    bool operator==(const PrefixControllerIdentity& other) const {
        return namespace_id == other.namespace_id && controller_id == other.controller_id && term == other.term;
    }
};

struct PrefixTransferIdentity {
    std::string transfer_id;
    std::string source_worker_id;
    std::string destination_worker_id;
    std::string source_epoch;
    std::string destination_epoch;
    std::string source_layout;
    std::string destination_layout;

    bool        valid() const;
    bool        empty() const;
    bool        operator==(const PrefixTransferIdentity& other) const;
    std::string wirePrefix(const PrefixControllerIdentity& controller,
                           uint64_t                        source_lease_id,
                           uint64_t                        destination_operation_id) const;
};

struct PrefixPrepareIdentity {
    std::string            request_id;
    std::string            namespace_id;
    int64_t                deadline_unix_ms = 0;
    PrefixTransferIdentity transfer;

    bool operator==(const PrefixPrepareIdentity& other) const {
        return request_id == other.request_id && namespace_id == other.namespace_id
               && deadline_unix_ms == other.deadline_unix_ms && transfer == other.transfer;
    }
};

class PrefixCacheTransactionManager {
public:
    enum class State {
        PREPARED,
        COPYING,
        DATA_READY,
        COMMITTED,
        ABORTED
    };
    using Clock   = std::chrono::steady_clock;
    using Now     = std::function<Clock::time_point()>;
    using WallNow = std::function<int64_t()>;

    using PrepareIdentity    = PrefixPrepareIdentity;
    using TransferIdentity   = PrefixTransferIdentity;
    using ControllerIdentity = PrefixControllerIdentity;
    using ControllerCommand  = std::optional<ControllerIdentity>;

    static int64_t wallTimeMs();

    struct LifecycleOptions {
        std::chrono::milliseconds lease_ttl{300000};
        std::chrono::milliseconds terminal_ttl{300000};
        std::chrono::milliseconds sweep_interval{100};
        size_t                    max_records          = 4096;
        bool                      pinned_import_commit = false;
    };

    struct ExportLease {
        uint64_t                  id      = 0;
        uint64_t                  epoch   = 0;
        uint64_t                  version = 0;
        std::vector<CacheKeyType> keys;
        BlockIndicesType          blocks;
        Clock::time_point         expires_at;
        PrepareIdentity           prepare;
        bool                      active = true;
        ControllerIdentity        controller;
    };

    struct ImportOperation {
        uint64_t                  id               = 0;
        uint64_t                  epoch            = 0;
        uint64_t                  expected_version = 0;
        uint64_t                  source_lease_id  = 0;
        State                     state            = State::ABORTED;
        std::vector<CacheKeyType> keys;
        BlockIndicesType          destination_blocks;
        int                       rank_count        = 1;
        uint64_t                  committed_version = 0;
        uint64_t                  local_export_id   = 0;
        Clock::time_point         expires_at;
        PrepareIdentity           prepare;
        ControllerIdentity        controller;
        std::vector<uint32_t>     missing_indices;
        std::vector<uint64_t>     pinned_generations;
        bool                      pinned_path_binding = false;

        BlockIndicesType stagingBlocks() const;
    };

    using CopyBlock     = std::function<void(BlockIdxType, BlockIdxType)>;
    using ResourceLease = std::shared_ptr<void>;

    PrefixCacheTransactionManager(BlockPoolPtr block_pool, BlockCachePtr block_cache, CopyBlock copy_block);
    PrefixCacheTransactionManager(BlockPoolPtr     block_pool,
                                  BlockCachePtr    block_cache,
                                  CopyBlock        copy_block,
                                  LifecycleOptions options,
                                  Now              now      = Clock::now,
                                  WallNow          wall_now = wallTimeMs);
    ~PrefixCacheTransactionManager();

    bool activateController(const ControllerIdentity& controller);
    bool isControllerCurrent(const ControllerIdentity& controller) const;
    bool bindTopologyController(const ControllerIdentity& controller, const std::string& binding);
    bool topologyControllerMatches(const ControllerIdentity& controller, const std::string& binding) const;
    bool bindTopologyQuarantine(const ControllerIdentity& controller, const std::string& binding);
    enum class TopologyState {
        ACTIVE,
        CANCELLED,
        RELEASED,
        QUARANTINED,
        CONFIRMED
    };
    struct TopologyReceipt {
        ControllerIdentity controller;
        std::string        binding;
        TopologyState      state;
    };
    bool
    releaseTopologyController(const ControllerIdentity& controller, const std::string& binding, bool completed = false);
    std::optional<TopologyReceipt> topologyReceipt() const;
    bool     confirmTopologyController(const ControllerIdentity& controller, const std::string& binding);
    uint64_t beginTopologyChange();
    bool     topologyChangeDrained(uint64_t generation) const;
    bool     resumeAfterTopologyChange(uint64_t generation);

    std::optional<ExportLease>     prepareExport(uint64_t                         epoch,
                                                 uint64_t                         version,
                                                 const std::vector<CacheKeyType>& keys,
                                                 std::optional<PrepareIdentity>   prepare    = std::nullopt,
                                                 ControllerCommand                controller = std::nullopt);
    std::optional<ImportOperation> prepareImport(uint64_t                         epoch,
                                                 uint64_t                         expected_version,
                                                 const std::vector<CacheKeyType>& keys,
                                                 int                              rank_count      = 1,
                                                 uint64_t                         source_lease_id = 0,
                                                 std::optional<PrepareIdentity>   prepare         = std::nullopt,
                                                 ControllerCommand                controller      = std::nullopt);
    std::optional<ExportLease>
    getPreparedExport(uint64_t epoch, const std::string& namespace_id, const std::string& request_id) const;
    std::optional<ImportOperation>
    getPreparedImport(uint64_t epoch, const std::string& namespace_id, const std::string& request_id) const;
    bool
    markImportRankReady(uint64_t id, uint64_t current_epoch, int rank, ControllerCommand controller = std::nullopt);
    bool beginImportTransfer(uint64_t id, uint64_t current_epoch, ControllerCommand controller = std::nullopt);
    bool commitImportedData(uint64_t          id,
                            uint64_t          source_lease_id,
                            uint64_t          current_epoch,
                            uint64_t          current_version,
                            ControllerCommand controller = std::nullopt);
    bool commitImport(uint64_t           id,
                      const ExportLease& source,
                      uint64_t           current_epoch,
                      uint64_t           current_version,
                      ControllerCommand  controller = std::nullopt);
    bool commitImport(uint64_t          id,
                      uint64_t          export_id,
                      uint64_t          current_epoch,
                      uint64_t          current_version,
                      ControllerCommand controller = std::nullopt);
    bool abortImport(uint64_t id, uint64_t current_epoch, ControllerCommand controller = std::nullopt);
    bool releaseExport(uint64_t id, uint64_t current_epoch, ControllerCommand controller = std::nullopt);
    bool reclaim(uint64_t                         epoch,
                 uint64_t                         expected_version,
                 const std::vector<CacheKeyType>& keys,
                 ControllerCommand                controller = std::nullopt);
    ImportOperation            getImport(uint64_t id) const;
    std::optional<ExportLease> getExport(uint64_t id) const;
    std::optional<ResourceLease>
    pinExport(uint64_t id, uint64_t current_epoch, ControllerCommand controller = std::nullopt);
    std::optional<ResourceLease>
             pinImport(uint64_t id, uint64_t current_epoch, ControllerCommand controller = std::nullopt);
    bool     importSnapshotCurrent(uint64_t id) const;
    uint64_t mutationVersion() const;

private:
    bool authorizedLocked(const ControllerCommand& controller, const ControllerIdentity* owner = nullptr) const;
    bool topologyChangeDrainedLocked(uint64_t generation) const;
    ResourceLease pinBlocksLocked(const BlockIndicesType& blocks);

    struct ImportRecord {
        ImportOperation            operation;
        bool                       released = false;
        ResourceLease              ancestor_lease;
        bool                       copying_locally = false;
        Clock::time_point          retire_at;
        std::optional<ExportLease> local_source;
    };

    struct ExportRecord {
        ExportLease       lease;
        bool              released = false;
        Clock::time_point retire_at;
    };

    void finishImportLocked(ImportRecord& record, std::optional<uint64_t> committed_version, Clock::time_point now);
    void releaseExportLocked(ExportRecord& record, Clock::time_point now);
    void expireLocked(Clock::time_point now);
    void startReaperLocked();
    void runReaper();
    std::optional<Clock::time_point> prepareDeadlineLocked(const std::optional<PrepareIdentity>& prepare,
                                                           Clock::time_point                     now);

    mutable std::mutex                                mu_;
    BlockPoolPtr                                      block_pool_;
    BlockCachePtr                                     block_cache_;
    CopyBlock                                         copy_block_;
    std::unordered_map<uint64_t, ImportRecord>        imports_;
    std::unordered_map<uint64_t, ExportRecord>        exports_;
    std::unordered_set<uint64_t>                      busy_exports_;
    std::unordered_map<uint64_t, PrefixTransferState> transfer_states_;
    uint64_t                                          next_id_ = 1;
    LifecycleOptions                                  options_;
    Now                                               now_;
    std::condition_variable                           wake_;
    bool                                              stopping_ = false;
    std::thread                                       reaper_;
    WallNow                                           wall_now_;
    int64_t                                           wall_anchor_ms_ = 0;
    Clock::time_point                                 steady_anchor_;
    std::unordered_map<std::string, uint64_t>         export_requests_;
    std::unordered_map<std::string, uint64_t>         import_requests_;
    ControllerCommand                                 controller_;
    std::string                                       topology_control_binding_;
    std::string                                       released_topology_control_binding_;
    std::optional<TopologyReceipt>                    topology_receipt_;
    bool                                              topology_changing_        = false;
    uint64_t                                          topology_generation_      = 0;
    uint64_t                                          topology_operation_floor_ = 1;
    std::shared_ptr<std::atomic<size_t>> active_transfer_leases_ = std::make_shared<std::atomic<size_t>>(0);
};

}  // namespace rtp_llm
