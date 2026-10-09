package org.flexlb.sync.synchronizer;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.GrpcCacheStatusCheckRunner;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.LongAdder;

public final class PrefixInventoryRefresh implements Runnable {
    private static final Logger LOGGER = LoggerFactory.getLogger("syncLogger");
    private final long intervalMillis;
    private final String modelName;
    private final ModelWorkerStatus workers;
    private final Executor executor;
    private final EngineHealthReporter healthReporter;
    private final EngineGrpcService grpc;
    private final CacheAwareService cache;
    private final long requestTimeoutMillis;
    private final LongAdder syncCount;
    private final long statusIntervalMillis;
    private final LongAdder rounds = new LongAdder();
    private final LongAdder submitted = new LongAdder();
    private final LongAdder completed = new LongAdder();
    private final LongAdder busySkipped = new LongAdder();
    private final LongAdder rejected = new LongAdder();

    public PrefixInventoryRefresh(long intervalMillis, String modelName, ModelWorkerStatus workers,
                                  Executor executor, EngineHealthReporter healthReporter, EngineGrpcService grpc,
                                  CacheAwareService cache, long requestTimeoutMillis, LongAdder syncCount,
                                  long statusIntervalMillis) {
        if (intervalMillis < 0 || (intervalMillis > 0 && (cache.prefixObserver() == null
                || !cache.prefixObserver().settings().confirmed()))) {
            throw new IllegalArgumentException("independent inventory refresh requires confirmed metadata");
        }
        this.intervalMillis = intervalMillis;
        this.modelName = modelName;
        this.workers = workers;
        this.executor = executor;
        this.healthReporter = healthReporter;
        this.grpc = grpc;
        this.cache = cache;
        this.requestTimeoutMillis = requestTimeoutMillis;
        this.syncCount = syncCount;
        this.statusIntervalMillis = statusIntervalMillis;
    }

    public static long configuredInterval(Map<String, String> environment) {
        long interval = Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_INVENTORY_SYNC_MS", "0"));
        if (interval < 0) { throw new IllegalArgumentException("negative inventory refresh interval"); }
        return interval;
    }

    public long intervalMillis() { return intervalMillis; }

    @Override
    public void run() {
        if (intervalMillis == 0) { return; }
        rounds.increment();
        for (var role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            for (var worker : workers.getRoleStatusMap(role).values()) {
                if (!worker.isAlive()) { continue; }
                if (!worker.getCacheCheckInProgress().compareAndSet(false, true)) {
                    busySkipped.increment();
                    continue;
                }
                try {
                    var check = new GrpcCacheStatusCheckRunner(modelName, worker.getIpPort(), worker.getSite(), role,
                            worker, healthReporter, grpc, cache, requestTimeoutMillis, syncCount,
                            statusIntervalMillis, true);
                    executor.execute(() -> {
                        try { check.run(); }
                        finally { completed.increment(); }
                    });
                    submitted.increment();
                } catch (RuntimeException exception) {
                    worker.getCacheCheckInProgress().set(false);
                    rejected.increment();
                    LOGGER.warn("prefix inventory refresh submission failed for {}", worker.getIpPort(), exception);
                }
            }
        }
    }

    public Map<String, Object> status() {
        return Map.of("intervalMillis", intervalMillis, "workerStatusIntervalMillis", statusIntervalMillis,
                "rounds", rounds.sum(), "submitted", submitted.sum(), "completed", completed.sum(),
                "busySkipped", busySkipped.sum(), "rejected", rejected.sum());
    }
}
