package org.flexlb.balance.strategy;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Cancels engine-owned RECEIVED requests so the gateway can reroute them. */
@Component
public class CpReceivedRequestMigrator {

    private static final long DEFAULT_RPC_TIMEOUT_MS = 500L;

    private final EngineWorkerStatus engineWorkerStatus;
    private final EngineGrpcService engineGrpcService;
    private final long rpcTimeoutMs;
    private final Map<RequestEpoch, EngineRpcService.CancelIfReceivedOutcomePB> terminal =
            new ConcurrentHashMap<>();
    private volatile long cachedOperationEpoch = -1L;

    @Autowired
    public CpReceivedRequestMigrator(EngineWorkerStatus engineWorkerStatus,
                                     EngineGrpcService engineGrpcService) {
        this(engineWorkerStatus, engineGrpcService,
                readLongEnv("FLEXLB_CP_CANCEL_RECEIVED_TIMEOUT_MS", DEFAULT_RPC_TIMEOUT_MS));
    }

    CpReceivedRequestMigrator(EngineWorkerStatus engineWorkerStatus,
                              EngineGrpcService engineGrpcService,
                              long rpcTimeoutMs) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineGrpcService = engineGrpcService;
        this.rpcTimeoutMs = Math.max(1L, rpcTimeoutMs);
    }

    public MigrationSummary migrate(Iterable<String> participants, long operationEpoch) {
        if (operationEpoch <= 0L) {
            throw new IllegalArgumentException("operationEpoch must be positive");
        }
        resetTerminalCacheFor(operationEpoch);
        List<String> selected = new java.util.ArrayList<>();
        participants.forEach(selected::add);
        List<EngineWorkerStatus.CpReceivedTask> tasks =
                engineWorkerStatus.cpReceivedTasks(selected);

        int cancelled = 0;
        int alreadyCancelled = 0;
        int becameActive = 0;
        int notFound = 0;
        int skipped = 0;
        int errors = 0;
        // Cancellation acquires engine stream-registry locks. Keep control RPCs
        // ordered per conversion instead of bursting all RECEIVED tasks at once.
        for (EngineWorkerStatus.CpReceivedTask task : tasks) {
            Attempt attempt = cancel(task, operationEpoch);
            switch (attempt.kind()) {
                case CANCELLED -> cancelled++;
                case ALREADY_CANCELLED -> alreadyCancelled++;
                case BECAME_ACTIVE -> becameActive++;
                case NOT_FOUND -> notFound++;
                case SKIPPED -> skipped++;
                case ERROR -> errors++;
            }
        }
        return new MigrationSummary(tasks.size(), cancelled, alreadyCancelled,
                becameActive, notFound, skipped, errors);
    }

    private Attempt cancel(EngineWorkerStatus.CpReceivedTask task, long operationEpoch) {
        RequestEpoch key = new RequestEpoch(task.endpoint(), task.requestId(), operationEpoch);
        EngineRpcService.CancelIfReceivedOutcomePB previous = terminal.get(key);
        if (previous != null) {
            return new Attempt(AttemptKind.SKIPPED, previous);
        }
        try {
            EngineRpcService.CancelIfReceivedResponsePB response =
                    engineGrpcService.cancelIfReceived(task.ip(),
                            CommonUtils.toGrpcPort(task.httpPort()), task.requestId(),
                            operationEpoch, rpcTimeoutMs);
            EngineRpcService.CancelIfReceivedOutcomePB outcome = response.getOutcome();
            return switch (outcome) {
                case CANCEL_IF_RECEIVED_CANCELLED -> {
                    terminal.put(key, outcome);
                    yield new Attempt(AttemptKind.CANCELLED, outcome);
                }
                case CANCEL_IF_RECEIVED_ALREADY_CANCELLED -> {
                    terminal.put(key, outcome);
                    yield new Attempt(AttemptKind.ALREADY_CANCELLED, outcome);
                }
                case CANCEL_IF_RECEIVED_NOT_RECEIVED -> {
                    terminal.put(key, outcome);
                    yield new Attempt(AttemptKind.BECAME_ACTIVE, outcome);
                }
                case CANCEL_IF_RECEIVED_NOT_FOUND ->
                        new Attempt(AttemptKind.NOT_FOUND, outcome);
                case CANCEL_IF_RECEIVED_OUTCOME_UNSPECIFIED, UNRECOGNIZED ->
                        new Attempt(AttemptKind.ERROR, outcome);
            };
        } catch (RuntimeException error) {
            return new Attempt(AttemptKind.ERROR,
                    EngineRpcService.CancelIfReceivedOutcomePB
                            .CANCEL_IF_RECEIVED_OUTCOME_UNSPECIFIED);
        }
    }

    private synchronized void resetTerminalCacheFor(long operationEpoch) {
        if (cachedOperationEpoch != operationEpoch) {
            terminal.clear();
            cachedOperationEpoch = operationEpoch;
        }
    }

    public record MigrationSummary(int candidates,
                                   int cancelled,
                                   int alreadyCancelled,
                                   int becameActive,
                                   int notFound,
                                   int skipped,
                                   int errors) {
        public boolean didWork() {
            return candidates > 0;
        }
    }

    private record RequestEpoch(String endpoint, long requestId, long operationEpoch) {
    }

    private record Attempt(AttemptKind kind,
                           EngineRpcService.CancelIfReceivedOutcomePB outcome) {
    }

    private enum AttemptKind {
        CANCELLED,
        ALREADY_CANCELLED,
        BECAME_ACTIVE,
        NOT_FOUND,
        SKIPPED,
        ERROR
    }

    private static long readLongEnv(String name, long fallback) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
