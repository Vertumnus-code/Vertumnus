package org.flexlb.engine.grpc;

import com.google.protobuf.MessageLite;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioSocketChannel;
import lombok.Getter;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.cache.prefix.PrefixLatencyMetrics;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Engine gRPC client for worker status queries
 */
@Component
public class EngineGrpcClient extends AbstractGrpcClient<AbstractGrpcClient.GrpcStubWrapper> {

    private final PrefixLatencyMetrics prefixControlLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics cacheStatusLatency = new PrefixLatencyMetrics();

    public java.util.Map<String, PrefixLatencyMetrics.Snapshot> prefixRpcLatency() {
        return java.util.Map.of("prefix_control_logical_rpc", prefixControlLatency.snapshot(),
                "cache_status_logical_rpc", cacheStatusLatency.snapshot());
    }

    @Getter
    private final Executor executor;
    @Getter
    private final EventLoopGroup eventLoopGroup;

    public EngineGrpcClient(CustomNameResolver nameResolver,
                            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
                            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup,
                            EngineLocalView engineLocalView,
                            GlobalCacheIndex globalCacheIndex,
                            GrpcReporter grpcReporter) {
        super(engineLocalView, globalCacheIndex, grpcReporter);
        this.executor = executor;
        this.eventLoopGroup = eventLoopGroup;
        nameResolver.start(this);
    }

    /**
     * Common method to execute gRPC calls with proper channel management and error handling
     *
     * @param requestTimeoutMs request timeout in milliseconds.
     * @param serviceType      the service type for channel selection
     */
    private <R> R executeGrpcCall(String ip, int port,
                                  Function<GrpcStubWrapper, R> grpcCall,
                                  long requestTimeoutMs,
                                  ServiceType serviceType) {

        return executeGrpcCall(ip, port, grpcCall, requestTimeoutMs, serviceType, true);
    }

    private <R> R executeGrpcCall(String ip, int port,
                                  Function<GrpcStubWrapper, R> grpcCall,
                                  long requestTimeoutMs, ServiceType serviceType, boolean retryOnConnectionFailure) {

        PrefixLatencyMetrics metrics = serviceType == ServiceType.PREFIX_INSPECT ? prefixControlLatency
                : serviceType == ServiceType.CACHE_STATUS ? cacheStatusLatency : null;
        if (metrics == null) {
            return executeGrpcCallUnmeasured(ip, port, grpcCall, requestTimeoutMs, serviceType, retryOnConnectionFailure);
        }
        return metrics.measure(() -> executeGrpcCallUnmeasured(ip, port, grpcCall, requestTimeoutMs,
                serviceType, retryOnConnectionFailure));
    }

    private <R> R executeGrpcCallUnmeasured(String ip, int port,
                                  Function<GrpcStubWrapper, R> grpcCall,
                                  long requestTimeoutMs, ServiceType serviceType, boolean retryOnConnectionFailure) {

        String channelKey = createKey(ip, port, serviceType);
        Invoker invoker = getInvoker(channelKey);

        if (invoker == null) {
            Logger.warn("ip:{} {} grpc channel not found, creating and adding to pool", ip, serviceType);
            ManagedChannel newChannel = createChannel(channelKey);
            invoker = putInvokerIfAbsent(channelKey, newChannel);
        } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
            Logger.warn("ip:{} {} grpc channel is shutdown or terminated, recreating and updating pool", ip, serviceType);
            ManagedChannel newChannel = createChannel(channelKey);
            invoker = replaceInvoker(channelKey, invoker, newChannel);
        }

        try {
            invoker.updateLastUsedTime();
            GrpcStubWrapper stubWrapper = invoker.getRpcServiceStub()
                    .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            long startTime = System.nanoTime() / 1000;
            R response = grpcCall.apply(stubWrapper);
            long endTime = System.nanoTime() / 1000;

            // Calculate response body size in bytes
            int responseSize = 0;
            if (response instanceof MessageLite messageLite) {
                responseSize = messageLite.getSerializedSize();
            }

            // Record statistics
            long duration = endTime - startTime;
            grpcReporter.reportCallMetrics(ip, serviceType.getOperationName(), duration, responseSize, false);

            return response;
        } catch (StatusRuntimeException e) {
            if (retryOnConnectionFailure && isConnectionBrokenError(e)) {
                invoker.markExpired();
                long connectionDuration = invoker.getConnectionDuration();
                grpcReporter.reportConnectionDuration(ip, serviceType.getOperationName(), connectionDuration);
                Logger.warn("Connection broken for {}:{} {}, duration: {}μs, recreating channel and retrying once, msh:{}",
                        ip, port, serviceType, connectionDuration, e.getMessage());
                return retryWithNewChannel(channelKey, invoker, grpcCall, requestTimeoutMs, ip, port, serviceType);
            }
            Logger.error("Exception during {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        } catch (Exception e) {
            Logger.error("Exception during {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        }
    }

    private boolean isConnectionBrokenError(StatusRuntimeException e) {
        String message = e.getMessage();
        return message != null &&
               (message.contains("end-of-stream mid-frame") ||
                message.contains("Connection reset") ||
                message.contains("Broken pipe") ||
                message.contains("http2 exception") ||
                message.contains("Incomplete header block fragment"));
    }

    private <R> R retryWithNewChannel(String channelKey,
                                      Invoker staleInvoker,
                                      Function<GrpcStubWrapper, R> grpcCall,
                                      long requestTimeoutMs,
                                      String ip, int port,
                                      ServiceType serviceType) {
        ManagedChannel newChannel = createChannel(channelKey);
        Invoker newInvoker = replaceInvoker(channelKey, staleInvoker, newChannel);

        Logger.info("Retrying gRPC call with new channel for {}:{} {}", ip, port, serviceType);

        GrpcStubWrapper stubWrapper = newInvoker.getRpcServiceStub()
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

        long startTime = System.nanoTime() / 1000;
        R response = grpcCall.apply(stubWrapper);
        long endTime = System.nanoTime() / 1000;

        // Calculate response body size in bytes
        int responseSize = 0;
        if (response instanceof MessageLite messageLite) {
            responseSize = messageLite.getSerializedSize();
        }

        // Record retry statistics
        long duration = endTime - startTime;
        grpcReporter.reportCallMetrics(ip, serviceType.getOperationName(), duration, responseSize, true);

        return response;
    }

    /**
     * Get worker status via gRPC
     */
    public EngineRpcService.WorkerStatusPB getWorkerStatus(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.WORKER_STATUS);
    }

    /**
     * Get cache status via gRPC
     */
    public EngineRpcService.CacheStatusPB getCacheStatus(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getCacheStatus(request), requestTimeoutMs, ServiceType.CACHE_STATUS);
    }

    public EngineRpcService.EmptyPB submitPrefixTopology(
            String ip, int port, EngineRpcService.UpdateSchedulerInfoRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().updateSchedulerInfo(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixCacheCapabilitiesResponsePB getPrefixCacheCapabilities(
            String ip, int port, EngineRpcService.PrefixCacheCapabilitiesRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getPrefixCacheCapabilities(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT);
    }

    public EngineRpcService.InspectPrefixResponsePB inspectPrefix(
            String ip, int port, EngineRpcService.InspectPrefixRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().inspectPrefix(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT);
    }

    public EngineRpcService.ActivatePrefixControllerResponsePB activatePrefixController(
            String ip, int port, EngineRpcService.ActivatePrefixControllerRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().activatePrefixController(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PreparePrefixExportResponsePB preparePrefixExport(
            String ip, int port, EngineRpcService.PreparePrefixExportRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().preparePrefixExport(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PreparePrefixImportResponsePB preparePrefixImport(
            String ip, int port, EngineRpcService.PreparePrefixImportRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().preparePrefixImport(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB commitPrefixImport(
            String ip, int port, EngineRpcService.CommitPrefixImportRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().commitPrefixImport(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB abortPrefixOperation(
            String ip, int port, EngineRpcService.AbortPrefixOperationRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().abortPrefixOperation(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB releasePrefixExport(
            String ip, int port, EngineRpcService.ReleasePrefixExportRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().releasePrefixExport(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.GetPrefixPrepareResponsePB getPrefixPrepare(
            String ip, int port, EngineRpcService.GetPrefixPrepareRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getPrefixPrepare(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT);
    }

    public EngineRpcService.GetPrefixOperationResponsePB getPrefixOperation(
            String ip, int port, EngineRpcService.GetPrefixOperationRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getPrefixOperation(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT);
    }

    public EngineRpcService.PrefixOperationResponsePB reclaimPrefix(
            String ip, int port, EngineRpcService.ReclaimPrefixRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().reclaimPrefix(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB markPrefixRankReady(
            String ip, int port, EngineRpcService.MarkPrefixRankReadyRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().markPrefixRankReady(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB startPrefixSend(
            String ip, int port, EngineRpcService.PrefixTransferEndpointRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().startPrefixSend(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    public EngineRpcService.PrefixOperationResponsePB startPrefixReceive(
            String ip, int port, EngineRpcService.PrefixTransferEndpointRequestPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().startPrefixReceive(request),
                requestTimeoutMs, ServiceType.PREFIX_INSPECT, false);
    }

    /** Cancel a prefill request only while the engine still owns it in RECEIVED state. */
    public EngineRpcService.CancelIfReceivedResponsePB cancelIfReceived(
            String ip, int port, EngineRpcService.CancelIfReceivedRequestPB request,
            long requestTimeoutMs) {
        return executeGrpcCall(ip, port,
                stub -> stub.getRpcServiceStub().cancelIfReceived(request),
                requestTimeoutMs, ServiceType.CP_CANCEL_RECEIVED);
    }

    /**
     * Get multimodal worker status via gRPC
     */
    public EngineRpcService.WorkerStatusPB getMultimodalWorkerStatus(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getMultimodalRpcServiceStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_WORKER_STATUS);
    }

    /**
     * Get multimodal cache status via gRPC
     */
    public EngineRpcService.CacheStatusPB getMultimodalCacheStatus(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getMultimodalRpcServiceStub().getCacheStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_CACHE_STATUS);
    }

    @Override
    protected ManagedChannel createChannel(String channelKey) {
        String[] parts = parseServiceKey(channelKey);
        String ip = parts[0];
        int port = Integer.parseInt(parts[1]);
        Logger.info("Creating new channel for ip: {}, port: {}", ip, port);
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                // Connection timeout in milliseconds
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 20)
                // Write buffer water mark: prevents memory accumulation and pendingTasks buildup
                .withOption(ChannelOption.WRITE_BUFFER_WATER_MARK, new WriteBufferWaterMark(64 * 1024, 128 * 1024))
                // Receive/send buffer size
                .withOption(ChannelOption.SO_RCVBUF, 512 * 1024)
                .withOption(ChannelOption.SO_SNDBUF, 512 * 1024)
                // Maximum message size limit (8MB)
                .maxInboundMessageSize(8 * 1024 * 1024)
                // HTTP/2 initial flow control window: prevents transmission issues due to flow control
                .initialFlowControlWindow(2 * 1024 * 1024)
                // gRPC keepalive configuration: keeps connection active, prevents disconnection by intermediate devices
                .keepAliveTime(2, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .keepAliveWithoutCalls(true)
                .executor(executor)
                .eventLoopGroup(eventLoopGroup)
                .usePlaintext()
                .proxyDetector(GrpcUtil.NOOP_PROXY_DETECTOR)
                .disableRetry()
                .build();
    }

    @Override
    protected GrpcStubWrapper createStub(ManagedChannel channel) {
        return new GrpcStubWrapper(
                RpcServiceGrpc.newBlockingStub(channel),
                MultimodalRpcServiceGrpc.newBlockingStub(channel)
        );
    }
}
