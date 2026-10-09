package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Endpoint;

import java.util.Objects;

public final class GrpcPrefixTransferWorkerClient implements PrefixTransferCoordinator.WorkerClient {
    private final EngineGrpcClient grpcClient;
    private final long requestTimeoutMs;
    private final PrefixRpcTiming timing;

    public GrpcPrefixTransferWorkerClient(EngineGrpcClient grpcClient, long requestTimeoutMs) {
        this(grpcClient, requestTimeoutMs, PrefixRpcTiming.configured());
    }

    GrpcPrefixTransferWorkerClient(EngineGrpcClient grpcClient, long requestTimeoutMs, PrefixRpcTiming timing) {
        this.grpcClient = Objects.requireNonNull(grpcClient);
        this.timing = Objects.requireNonNull(timing);
        if (requestTimeoutMs <= 0) {
            throw new IllegalArgumentException("request timeout must be positive");
        }
        this.requestTimeoutMs = requestTimeoutMs;
    }

    @Override
    public EngineRpcService.ActivatePrefixControllerResponsePB activateController(
            Endpoint endpoint, EngineRpcService.ActivatePrefixControllerRequestPB request) {
        return timing.call("activateController", endpoint, request, requestTimeoutMs,
                () -> grpcClient.activatePrefixController(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PreparePrefixExportResponsePB prepareExport(
            Endpoint endpoint, EngineRpcService.PreparePrefixExportRequestPB request) {
        return timing.call("prepareExport", endpoint, request, requestTimeoutMs,
                () -> grpcClient.preparePrefixExport(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PreparePrefixImportResponsePB prepareImport(
            Endpoint endpoint, EngineRpcService.PreparePrefixImportRequestPB request) {
        return timing.call("prepareImport", endpoint, request, requestTimeoutMs,
                () -> grpcClient.preparePrefixImport(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.GetPrefixPrepareResponsePB getPrepare(
            Endpoint endpoint, EngineRpcService.GetPrefixPrepareRequestPB request) {
        return timing.call("getPrepare", endpoint, request, requestTimeoutMs,
                () -> grpcClient.getPrefixPrepare(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB commitImport(
            Endpoint endpoint, EngineRpcService.CommitPrefixImportRequestPB request) {
        return timing.call("commitImport", endpoint, request, requestTimeoutMs,
                () -> grpcClient.commitPrefixImport(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB abort(
            Endpoint endpoint, EngineRpcService.AbortPrefixOperationRequestPB request) {
        return timing.call("abort", endpoint, request, requestTimeoutMs,
                () -> grpcClient.abortPrefixOperation(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB releaseExport(
            Endpoint endpoint, EngineRpcService.ReleasePrefixExportRequestPB request) {
        return timing.call("releaseExport", endpoint, request, requestTimeoutMs,
                () -> grpcClient.releasePrefixExport(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.GetPrefixOperationResponsePB getOperation(
            Endpoint endpoint, EngineRpcService.GetPrefixOperationRequestPB request) {
        return timing.call("getOperation", endpoint, request, requestTimeoutMs,
                () -> grpcClient.getPrefixOperation(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB startSend(
            Endpoint endpoint, EngineRpcService.PrefixTransferEndpointRequestPB request, long timeoutMs) {
        return timing.call("startSend", endpoint, request, timeoutMs,
                () -> grpcClient.startPrefixSend(endpoint.ip(), endpoint.port(), request, timeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB startReceive(
            Endpoint endpoint, EngineRpcService.PrefixTransferEndpointRequestPB request, long timeoutMs) {
        return timing.call("startReceive", endpoint, request, timeoutMs,
                () -> grpcClient.startPrefixReceive(endpoint.ip(), endpoint.port(), request, timeoutMs));
    }

    @Override
    public EngineRpcService.PrefixOperationResponsePB reclaim(
            Endpoint endpoint, EngineRpcService.ReclaimPrefixRequestPB request) {
        return timing.call("reclaim", endpoint, request, requestTimeoutMs,
                () -> grpcClient.reclaimPrefix(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }

    @Override
    public EngineRpcService.CacheStatusPB inventory(Endpoint endpoint) {
        var request = EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion(-1).setNeedCacheKeys(true).build();
        return timing.call("inventory", endpoint, request, requestTimeoutMs,
                () -> grpcClient.getCacheStatus(endpoint.ip(), endpoint.port(), request, requestTimeoutMs));
    }
}
