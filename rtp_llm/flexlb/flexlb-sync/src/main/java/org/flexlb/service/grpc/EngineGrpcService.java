package org.flexlb.service.grpc;

import lombok.Getter;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Engine gRPC Service for worker status queries
 */
@Component
public class EngineGrpcService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    @Getter
    private final EngineGrpcClient engineGrpcClient;

    public EngineGrpcService(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    /**
     * Get worker status via gRPC
     *
     * @param ip                  worker IP
     * @param grpcPort            worker gRPC port
     * @param finishedTaskVersion finished task version
     * @return CompletableFuture of WorkerStatusPB
     */
    public EngineRpcService.WorkerStatusPB getWorkerStatus(String ip, int grpcPort, long finishedTaskVersion, long requestTimeoutMs) {
        if (engineGrpcClient == null) {
            throw new RuntimeException("EngineGrpcService not initialized");
        }

        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestFinishedVersion(finishedTaskVersion)
                .build();

        return engineGrpcClient.getWorkerStatus(ip, grpcPort, request, requestTimeoutMs);
    }

    /**
     * Get worker status via gRPC with RoleType routing
     *
     * @param ip                  worker IP
     * @param grpcPort            worker gRPC port
     * @param finishedTaskVersion finished task version
     * @param roleType            role type to determine which service to use
     * @return CompletableFuture of WorkerStatusPB
     */
    public EngineRpcService.WorkerStatusPB getWorkerStatus(String ip, int grpcPort, long finishedTaskVersion, long requestTimeoutMs, RoleType roleType) {
        if (engineGrpcClient == null) {
            throw new RuntimeException("EngineGrpcService not initialized");
        }

        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestFinishedVersion(finishedTaskVersion)
                .build();

        // Use MultimodalRpcService for VIT role, RpcService for others
        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalWorkerStatus(ip, grpcPort, request, requestTimeoutMs);
        } else {
            return engineGrpcClient.getWorkerStatus(ip, grpcPort, request, requestTimeoutMs);
        }
    }

    public EngineRpcService.CancelIfReceivedResponsePB cancelIfReceived(
            String ip, int grpcPort, long requestId, long operationEpoch, long requestTimeoutMs) {
        if (engineGrpcClient == null) {
            throw new IllegalStateException("EngineGrpcService not initialized");
        }
        var request = EngineRpcService.CancelIfReceivedRequestPB.newBuilder()
                .setRequestId(requestId).setOperationEpoch(operationEpoch).build();
        return engineGrpcClient.cancelIfReceived(ip, grpcPort, request, requestTimeoutMs);
    }

    /**
     * Get cache status via gRPC
     *
     * @param ip           worker IP
     * @param grpcPort     worker gRPC port
     * @param workerStatus worker status
     * @param cacheVersion cache version for status check
     * @return CompletableFuture of CacheStatusPB
     */
    public EngineRpcService.CacheStatusPB getCacheStatus(
            String ip, int grpcPort, WorkerStatus workerStatus, long cacheVersion, long requestTimeoutMs) {

        if (engineGrpcClient == null) {
            throw new RuntimeException("EngineGrpcService not initialized");
        }
        // Only need cacheKeys for Prefill nodes in PD-separated mode and non PD-separated mode
        boolean needCacheKeys = RoleType.PREFILL.matches(workerStatus.getRole()) || RoleType.PDFUSION.matches(workerStatus.getRole());
        EngineRpcService.CacheVersionPB request = EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion((int) cacheVersion)
                .setNeedCacheKeys(needCacheKeys)
                .build();
        logger.info("Get cache status Request: {}, cacheVersion: {}, needCacheKeys: {}", ip, cacheVersion, needCacheKeys);
        return engineGrpcClient.getCacheStatus(ip, grpcPort, request, requestTimeoutMs);
    }

    /**
     * Get cache status via gRPC with RoleType routing
     *
     * @param ip           worker IP
     * @param grpcPort     worker gRPC port
     * @param workerStatus worker status
     * @param cacheVersion cache version for status check
     * @param roleType     role type to determine which service to use
     * @return CompletableFuture of CacheStatusPB
     */
    public EngineRpcService.CacheStatusPB getCacheStatus(
        String ip, int grpcPort, WorkerStatus workerStatus, long cacheVersion, long requestTimeoutMs, RoleType roleType) {
        return getCacheStatus(ip, grpcPort, workerStatus, cacheVersion, requestTimeoutMs, roleType, null);
    }

    public EngineRpcService.CacheStatusPB getCacheStatus(
        String ip, int grpcPort, WorkerStatus workerStatus, long cacheVersion, long requestTimeoutMs, RoleType roleType,
        org.flexlb.dao.cache.CacheReportV2 acceptedInventory) {
        return getCacheStatus(ip, grpcPort, workerStatus, cacheVersion, requestTimeoutMs, roleType,
                acceptedInventory, false);
    }

    public EngineRpcService.CacheStatusPB getCacheStatus(
        String ip, int grpcPort, WorkerStatus workerStatus, long cacheVersion, long requestTimeoutMs, RoleType roleType,
        org.flexlb.dao.cache.CacheReportV2 acceptedInventory, boolean inventoryOnly) {

        if (engineGrpcClient == null) {
            throw new RuntimeException("EngineGrpcService not initialized");
        }
        boolean prefill = RoleType.PREFILL.matches(workerStatus.getRole()) || RoleType.PDFUSION.matches(workerStatus.getRole());
        if (inventoryOnly && (!prefill || workerStatus.getDpSize() != 1
                || (roleType != RoleType.PREFILL && roleType != RoleType.PDFUSION))) {
            throw new IllegalArgumentException("inventory-only RPC requires a single prefill lane");
        }
        boolean needCacheKeys = prefill && !inventoryOnly;
        var builder = EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion((int) cacheVersion)
                .setNeedCacheKeys(needCacheKeys);
        if (acceptedInventory != null && prefill && roleType != RoleType.VIT) {
            builder.setPrefixInventoryAck(EngineRpcService.PrefixInventoryAckPB.newBuilder()
                    .setNamespace(acceptedInventory.namespace()).setWorkerId(acceptedInventory.workerId())
                    .setDpLane(acceptedInventory.dpLane()).setEpoch(acceptedInventory.epoch())
                    .setLayoutFingerprint(acceptedInventory.layoutFingerprint()).setVersion(acceptedInventory.version())
                    .setDeviceReady(acceptedInventory.deviceReady()).setCpDegree(acceptedInventory.cpDegree())
                    .setBlockSize(acceptedInventory.blockSize())
                    .setRankReadinessGeneration(acceptedInventory.rankReadinessGeneration()));
        }
        var request = builder.build();
        logger.info("Get cache status Request: {}, cacheVersion: {}, needCacheKeys: {}", ip, cacheVersion, needCacheKeys);

        // Use MultimodalRpcService for VIT role, RpcService for others
        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalCacheStatus(ip, grpcPort, request, requestTimeoutMs);
        } else {
            return engineGrpcClient.getCacheStatus(ip, grpcPort, request, requestTimeoutMs);
        }
    }
}
