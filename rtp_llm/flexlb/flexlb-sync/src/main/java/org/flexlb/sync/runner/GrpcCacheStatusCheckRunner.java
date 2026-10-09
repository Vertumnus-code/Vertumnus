package org.flexlb.sync.runner;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.List;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcCacheStatusCheckRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final String ip;
    private final int httpPort;
    private final int grpcPort;
    private static final Duration CACHE_STATUS_HTTP_TIMEOUT = Duration.ofSeconds(3);
    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(CACHE_STATUS_HTTP_TIMEOUT)
            .build();
    private final long startTime = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final boolean debug;
    private final long requestTimeoutMs;
    private final LongAdder syncCount;
    private final Long syncEngineStatusInterval;
    private final boolean independentInventoryRefresh;
    private final boolean inventoryOnlyRpc;

    public GrpcCacheStatusCheckRunner(String modelName, String ipPort, String site, RoleType roleType,
                                      WorkerStatus workerStatus,
                                      EngineHealthReporter engineHealthReporter,
                                      EngineGrpcService engineGrpcService,
                                      CacheAwareService cacheAwareService,
                                      long requestTimeoutMs,
                                      LongAdder syncCount,
                                      Long syncEngineStatusInterval) {
        this(modelName, ipPort, site, roleType, workerStatus, engineHealthReporter, engineGrpcService,
                cacheAwareService, requestTimeoutMs, syncCount, syncEngineStatusInterval, false);
    }

    public GrpcCacheStatusCheckRunner(String modelName, String ipPort, String site, RoleType roleType,
                                      WorkerStatus workerStatus, EngineHealthReporter engineHealthReporter,
                                      EngineGrpcService engineGrpcService, CacheAwareService cacheAwareService,
                                      long requestTimeoutMs, LongAdder syncCount, Long syncEngineStatusInterval,
                                      boolean independentInventoryRefresh) {
        this(modelName, ipPort, site, roleType, workerStatus, engineHealthReporter, engineGrpcService,
                cacheAwareService, requestTimeoutMs, syncCount, syncEngineStatusInterval, independentInventoryRefresh,
                Boolean.parseBoolean(System.getenv("FLEXLB_PREFIX_INVENTORY_ONLY_RPC")));
    }

    GrpcCacheStatusCheckRunner(String modelName, String ipPort, String site, RoleType roleType,
                              WorkerStatus workerStatus, EngineHealthReporter engineHealthReporter,
                              EngineGrpcService engineGrpcService, CacheAwareService cacheAwareService,
                              long requestTimeoutMs, LongAdder syncCount, Long syncEngineStatusInterval,
                              boolean independentInventoryRefresh, boolean inventoryOnlyRpc) {
        if (independentInventoryRefresh && (cacheAwareService.prefixObserver() == null
                || !cacheAwareService.prefixObserver().settings().confirmed()
                || (roleType != RoleType.PREFILL && roleType != RoleType.PDFUSION))) {
            throw new IllegalArgumentException("independent inventory refresh requires confirmed prefill metadata");
        }
        this.independentInventoryRefresh = independentInventoryRefresh;
        this.inventoryOnlyRpc = inventoryOnlyRpc;

        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.roleType = roleType;
        this.httpPort = Integer.parseInt(split[1]);
        this.grpcPort = CommonUtils.toGrpcPort(httpPort);
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.site = site;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;
        this.debug = Optional.ofNullable(System.getenv("WHALE_CACHE_DEBUG_MODE"))
                .map(Boolean::parseBoolean)
                .orElse(false);
        this.requestTimeoutMs = requestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
    }

    @Override
    public void run() {
        try {
            logger.info("GrpcCacheStatusCheckRunner run for {}", ipPort);
            long prefillCacheStatusCheckInterval = DynamicCacheIntervalService.getCurrentIntervalMs();
            long roundInterval = prefillCacheStatusCheckInterval / syncEngineStatusInterval;
            roundInterval = Math.max(roundInterval, 1);

            // Skip prefill cache status check if not in 100ms interval
            if (!independentInventoryRefresh && (RoleType.PREFILL.equals(roleType) || RoleType.PDFUSION.equals(roleType))
                        && syncCount.longValue() % roundInterval != 0) {
                logger.info("Skip prefill cache status check for {} because not in {}ms interval", ipPort, prefillCacheStatusCheckInterval);
                return;
            }

            long startTime = System.nanoTime() / 1000;
            long currentCacheVersion = getCurrentCacheVersion();

            // Launch gRPC cache status check
            var topology = workerStatus.getPrefixTopology();
            CacheStatus cacheStatus = launchGrpcCacheStatusCheck(ip, grpcPort, currentCacheVersion);
            Boolean singleLaneUpdated = withCacheReportLocks(() -> {
                if (topology != workerStatus.getPrefixTopology()
                        || !workerStatus.acceptsPrefixInventory(cacheStatus == null ? null : cacheStatus.getInventory())) { return null; }
                handleCacheStatusResponse(cacheStatus, startTime);
                return updateConfirmedSingleLaneCache(cacheStatus);
            });
            if (singleLaneUpdated == null) { return; }
            if (!singleLaneUpdated) { syncDpRankCacheStatuses(); }
        } finally {
            workerStatus.getCacheCheckInProgress().set(false);
        }
    }

    private <T> T withCacheReportLocks(Supplier<T> operation) {
        var observer = cacheAwareService.prefixObserver();
        if (observer != null && observer.settings().confirmed()
                && (roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)) {
            while (true) {
                synchronized (observer) {
                    if (workerStatus.lock.tryLock()) {
                        try { return operation.get(); }
                        finally { workerStatus.lock.unlock(); }
                    }
                }
                workerStatus.lock.lock();
                workerStatus.lock.unlock();
            }
        }
        workerStatus.lock.lock();
        try { return operation.get(); }
        finally { workerStatus.lock.unlock(); }
    }

    private boolean updateConfirmedSingleLaneCache(CacheStatus status) {
        if ((roleType != RoleType.PREFILL && roleType != RoleType.PDFUSION)
                || workerStatus.getDpSize() != 1 || !workerStatus.isAlive()
                || status == null || status.getMessage() != null || status.getBlockSize() <= 0
                || status.getTotalKvCache() <= 0 || status.getAvailableKvCache() < 0
                || status.getAvailableKvCache() > status.getTotalKvCache()) { return false; }
        var observer = cacheAwareService.prefixObserver();
        var report = status.getInventory();
        if (observer == null || !observer.settings().confirmed() || report == null || report.dpLane() != 0
                || !report.deviceReady()
                || (!"FULL".equals(report.snapshotKind()) && !"NO_CHANGE".equals(report.snapshotKind()))
                || !report.memoryStatsConsistent()) { return false; }
        var snapshot = observer.directory().view().get(workerStatus.getDpRankCacheKey(0));
        if (snapshot == null || !snapshot.report().equals(report)) { return false; }
        var current = workerStatus.getDpRankCacheStatusMap().get(0);
        if (debug || current == null || status.getVersion() > current.getVersion()) {
            workerStatus.updateDpRankCacheStatus(0, status);
        }
        return true;
    }

    private void syncDpRankCacheStatuses() {
        if (!RoleType.PREFILL.equals(roleType) && !RoleType.PDFUSION.equals(roleType)) {
            return;
        }
        try {
            var topology = workerStatus.getPrefixTopology();
            URI uri = URI.create("http://" + ip + ":" + httpPort + "/cache_status");
            HttpRequest request = HttpRequest.newBuilder(uri)
                    .timeout(CACHE_STATUS_HTTP_TIMEOUT)
                    .GET()
                    .build();
            HttpResponse<String> response = HTTP_CLIENT.send(
                    request, HttpResponse.BodyHandlers.ofString());
            workerStatus.lock.lock();
            try {
                if (topology != workerStatus.getPrefixTopology()) { return; }
                if (response.statusCode() < 200 || response.statusCode() >= 300) {
                    invalidateInventoryLanes();
                    logger.debug("query DP cache HTTP status failed, ipPort: {}, status: {}",
                            ipPort, response.statusCode());
                    return;
                }
                CacheStatus envelope = JsonUtils.toObjectOrNull(response.body(), CacheStatus.class);
                if (envelope == null) {
                    invalidateInventoryLanes();
                    return;
                }
                List<CacheStatus> results = envelope.getResults();
                if (results == null || results.isEmpty()) {
                    updateDpRankCache(0, envelope);
                    invalidateInventoryLanes();
                    return;
                }
                for (int dpRank = 0; dpRank < results.size(); dpRank++) {
                    CacheStatus result = results.get(dpRank);
                    if (result == null) { continue; }
                    updateDpRankCache(result.getInventory() == null ? dpRank : result.getInventory().dpLane(), result);
                }
                var observer = cacheAwareService.prefixObserver();
                if (observer != null && observer.settings().confirmed() && workerStatus.getDpSize() > 1) {
                    var reportedLanes = results.stream().filter(result -> result != null && result.getMessage() == null
                                    && result.getInventory() != null)
                            .map(result -> result.getInventory().dpLane()).collect(java.util.stream.Collectors.toSet());
                    for (int lane = 0; lane < workerStatus.getDpSize(); lane++) {
                        if (!reportedLanes.contains(lane)) {
                            observer.directory().invalidate(workerStatus.getDpRankCacheKey(lane));
                        }
                    }
                }
            } finally { workerStatus.lock.unlock(); }
        } catch (Throwable throwable) {
            invalidateInventoryLanes();
            logger.debug("sync DP-rank cache status failed, ipPort: {}, msg: {}",
                    ipPort, throwable.getMessage());
        }
    }

    private void updateDpRankCache(int dpRank, CacheStatus cacheStatus) {
        if (cacheStatus == null || cacheStatus.getMessage() != null || !workerStatus.acceptsPrefixInventory(cacheStatus.getInventory())) {
            return;
        }
        acceptInventory(workerStatus.getDpRankCacheKey(dpRank), cacheStatus);
        CacheStatus current = workerStatus.getDpRankCacheStatusMap().get(dpRank);
        if (!debug && current != null && cacheStatus.getVersion() <= current.getVersion()) {
            return;
        }
        workerStatus.updateDpRankCacheStatus(dpRank, cacheStatus);
        if (workerStatus.getDpSize() > 1) {
            // A single-lane worker is already indexed incrementally under its plain
            // address by the gRPC sync above; re-registering the full snapshot here
            // would rewrite every block on every version bump.
            cacheAwareService.updateEngineBlockCache(
                    workerStatus.getDpRankCacheKey(dpRank), workerStatus.getRole(), cacheStatus);
        }
    }

    private CacheStatus launchGrpcCacheStatusCheck(String ip, int grpcPort, long cacheVersion) {
        try {
            org.flexlb.dao.cache.CacheReportV2 acceptedInventory = null;
            var observer = cacheAwareService.prefixObserver();
            boolean inventoryOnly = inventoryOnlyRpc && observer != null && observer.settings().confirmed()
                    && !observer.settings().mainCompatibleRouting()
                    && workerStatus.isAlive() && workerStatus.getDpSize() == 1
                    && (roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION);
            if (observer != null && observer.settings().confirmed() && workerStatus.isAlive()
                    && workerStatus.getDpSize() == 1
                    && (roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)) {
                var snapshot = observer.directory().view().get(workerStatus.getDpRankCacheKey(0));
                if (snapshot != null && snapshot.report().dpLane() == 0 && snapshot.report().deviceReady()
                        && workerStatus.acceptsPrefixInventory(snapshot.report())) {
                    acceptedInventory = snapshot.report();
                }
            }
            EngineRpcService.CacheStatusPB cacheStatus = inventoryOnly
                    ? engineGrpcService.getCacheStatus(
                            ip, grpcPort, workerStatus, cacheVersion, requestTimeoutMs, roleType, acceptedInventory, true)
                    : acceptedInventory == null
                    ? engineGrpcService.getCacheStatus(
                            ip, grpcPort, workerStatus, cacheVersion, requestTimeoutMs, roleType)
                    : engineGrpcService.getCacheStatus(
                            ip, grpcPort, workerStatus, cacheVersion, requestTimeoutMs, roleType, acceptedInventory);
            logger.info("gRPC Cache Status Response - handled for {}, role:{}, cache_key_size:{}, cache_version:{}, "
                            + "available_kv_cache:{}, total_kv_cache:{}, block_size:{}",
                    ipPort, roleType.name(), cacheStatus.getCacheKeysMap().size(), cacheStatus.getVersion(),
                    cacheStatus.getAvailableKvCache(), cacheStatus.getTotalKvCache(), cacheStatus.getBlockSize());

            return EngineStatusConverter.convertToCacheStatus(cacheStatus);
        } catch (Throwable throwable) {
            handleException(throwable);
            // Return a default CacheStatus with error information
            return CacheStatus.builder()
                    .version(-1)
                    .availableKvCache(0)
                    .totalKvCache(0)
                    .blockSize(0)
                    .message("Cache Status gRPC call failed: " + throwable.getMessage())
                    .build();
        }
    }

    private void handleCacheStatusResponse(CacheStatus newCacheStatus, long startTime) {

        try {
            logger.info("gRPC Cache Status - handled for {}, role:{}", ipPort, roleType.name());

            if (newCacheStatus.getMessage() != null) {
                if (cacheAwareService.prefixObserver() != null) {
                    cacheAwareService.prefixObserver().directory().invalidate(workerStatus.getDpRankCacheKey(0));
                }
                logger.error("gRPC Cache Status - {}, role:{}, message:{}", ipPort, roleType.name(), newCacheStatus.getMessage());
                return;
            }

            engineHealthReporter.reportCacheStatusCheckRemoteInfo(modelName, ipPort, roleType.name(), startTime);
            if (newCacheStatus.getInventory() != null) {
                acceptInventory(workerStatus.getDpRankCacheKey(newCacheStatus.getInventory().dpLane()), newCacheStatus);
            }

            // Latest available KvCache tokens
            long latestAvailableKvCacheTokens = newCacheStatus.getAvailableKvCache();
            // Latest used KvCache tokens
            long latestUsedKvCacheTokens = newCacheStatus.getTotalKvCache() - latestAvailableKvCacheTokens;

            // Update KvCache tokens
            workerStatus.updateKvCacheTokens(latestUsedKvCacheTokens, latestAvailableKvCacheTokens);

            if (validateCacheStatusResponse(workerStatus, newCacheStatus)) {

                workerStatus.setCacheStatus(newCacheStatus);
                updateLocalKvCache(workerStatus);
                logCacheStatusUpdate(newCacheStatus, startTime);
            }

            engineHealthReporter.reportCacheStatusCheckerSuccess(modelName, workerStatus);
            workerStatus.getCacheLastUpdateTime().set(System.nanoTime() / 1000);

        } catch (Throwable e) {
            log("engine cache status check via gRPC exception, msg: " + e.getMessage(), e);
            engineHealthReporter.reportCacheStatusCheckerFail(modelName, ipPort, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE);
        }
    }

    private void acceptInventory(String route, CacheStatus status) {
        if (roleType != RoleType.PREFILL && roleType != RoleType.PDFUSION) {
            return;
        }
        var observer = cacheAwareService.prefixObserver();
        if (observer != null && observer.settings().confirmed() && status.getInventory() != null) {
            if (!workerStatus.isAlive()) {
                observer.directory().invalidate(route);
                return;
            }
            var result = observer.report(route, status.getInventory());
            if (result == org.flexlb.cache.prefix.ConfirmedCacheDirectory.Result.INVALID
                    || result == org.flexlb.cache.prefix.ConfirmedCacheDirectory.Result.NEED_FULL) {
                observer.directory().invalidate(route);
            } else if ((result == org.flexlb.cache.prefix.ConfirmedCacheDirectory.Result.APPLIED
                    || result == org.flexlb.cache.prefix.ConfirmedCacheDirectory.Result.NO_CHANGE)
                    && status.getInventory().deviceReady() && workerStatus.getCpSize() <= 0) {
                workerStatus.setCpSize(status.getInventory().cpDegree());
            }
            observer.workload(route, workerStatus.getLocalTaskMap().values().stream()
                    .filter(task -> task != null && task.getDpRank() == status.getInventory().dpLane()).count(),
                    org.flexlb.balance.strategy.CpAwareTtftStrategy.calculateQueueSeconds(
                            workerStatus, status.getInventory().dpLane()));
        }
    }

    private void invalidateInventoryLanes() {
        var observer = cacheAwareService.prefixObserver();
        if (observer != null && observer.settings().confirmed()) {
            for (int lane = 1; lane < workerStatus.getDpSize(); lane++) {
                observer.directory().invalidate(workerStatus.getDpRankCacheKey(lane));
            }
        }
    }

    private boolean validateCacheStatusResponse(WorkerStatus workerStatus, CacheStatus newCacheStatus) {
        if (debug) {
            return true;
        }
        CacheStatus currentCacheStatus = workerStatus.getCacheStatus();
        if (currentCacheStatus != null && newCacheStatus.getVersion() <= currentCacheStatus.getVersion()) {
            logger.info("gRPC Cache Status - {}, role:{}, version not updated, current: {}, response: {}",
                    ipPort, roleType.name(), currentCacheStatus.getVersion(), newCacheStatus.getVersion());
            return false;
        }
        return true;
    }

    private void logCacheStatusUpdate(CacheStatus cacheStatus, long startTime) {

        logger.info("gRPC Cache Status - {}, role:{}, block_size:{}, version:{}, cacheKeySize:{},"
                        + " available_kv_cache:{}, total_kv_cache:{}, cost:{}, syncIntervalMs:{}",
                ipPort,
                roleType.name(),
                cacheStatus.getBlockSize(),
                cacheStatus.getVersion(),
                cacheStatus.getCacheKeySize(),
                cacheStatus.getAvailableKvCache(),
                cacheStatus.getTotalKvCache(),
                (System.nanoTime() / 1000) - startTime,
                DynamicCacheIntervalService.getCurrentIntervalMs());
    }

    private void updateLocalKvCache(WorkerStatus workerStatus) {
        try {
            if (!RoleType.PREFILL.equals(roleType) && !RoleType.PDFUSION.equals(roleType)) {
                return;
            }
            WorkerCacheUpdateResult result = cacheAwareService.updateEngineBlockCache(workerStatus);
            if (!result.isSuccess()) {
                logger.warn("Failed to update worker cache for IP: {}, error: {}", workerStatus.getIp(), result.getErrorMessage());
                engineHealthReporter.reportCacheStatusCheckerFail(modelName, ipPort, BalanceStatusEnum.CACHE_UPDATE_FAILED);
            }
        } catch (Exception e) {
            logger.warn("Exception to update worker cache for IP: {}, error: {}", workerStatus.getIp(), e.getMessage());
            engineHealthReporter.reportCacheStatusCheckerFail(modelName, ipPort, BalanceStatusEnum.CACHE_UPDATE_FAILED);
        }
    }

    private void log(String msg) {
        logger.info("[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                (System.nanoTime() / 1000) - startTime,
                msg);
    }

    private void log(String msg, Throwable e) {
        logger.info("[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                (System.nanoTime() / 1000) - startTime,
                msg,
                e);
    }

    private void handleException(Throwable ex) {
        log("gRPC cache status check failed:ipPort:" + ipPort + ", with exception: " + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            engineHealthReporter.reportCacheStatusCheckerFail(modelName, ipPort, BalanceStatusEnum.CACHE_GRPC_TIMEOUT);
        } else {
            engineHealthReporter.reportCacheStatusCheckerFail(modelName, ipPort, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE);
        }
    }

    private long getCurrentCacheVersion() {
        return debug ? -1L : Optional.ofNullable(workerStatus)
                .map(WorkerStatus::getCacheStatus)
                .map(CacheStatus::getVersion)
                .orElse(-1L);
    }
}
