package org.flexlb.cache.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * KV cache manager
 * Core functions:
 * 1. Unified management of two-level hash table (per-engine local view + global index)
 * 2. Serve prefix-match lookups for routing
 *
 * <p>The fleet cache index is populated from two sources: engine gRPC snapshots
 * (merge-only — never removes keys the engine failed to re-report) and FlexLB's own
 * routing decisions (authoritative for what each worker has been asked to compute).
 * Under CP/PDFUSION combined-role deployments the engine snapshot alone is unreliable
 * (per-worker KV cache shards thrash faster than the poll interval), so routing
 * decisions become the primary source of truth. Storage bounds live in
 * {@link EngineLocalView}'s per-engine LRU.
 *
 * @author FlexLB
 */
@Slf4j
@Getter
@Component
public class KvCacheManager {
    private final org.flexlb.cache.prefix.PrefixObserver prefixObserver = new org.flexlb.cache.prefix.PrefixObserver(
            org.flexlb.cache.prefix.PrefixCacheSettings.fromEnvironment(), java.time.Clock.systemUTC(),
            Boolean.parseBoolean(System.getenv().getOrDefault("FLEXLB_PREFIX_INFLIGHT_HINTS", "false")),
            Boolean.parseBoolean(System.getenv().getOrDefault("FLEXLB_PREFIX_ROUTING_HISTORY", "false")),
            Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PREFIX_ROUTING_HISTORY_MAX_BLOCKS", "65536")),
            Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PREFIX_OBSERVATION_QUEUE_CAPACITY", "0")));
    private java.util.concurrent.ScheduledExecutorService prefixScanner;
    private final java.util.concurrent.atomic.AtomicReference<java.util.function.Consumer<org.flexlb.cache.prefix.PrefixObserver.Scan>>
            prefixActions = new java.util.concurrent.atomic.AtomicReference<>();

    @Autowired
    private GlobalCacheIndex globalCacheIndex;

    @Autowired
    private EngineLocalView engineLocalView;

    @Autowired
    private WorkerStatusProvider workerStatusProvider;

    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;

    private final LongAdder totalUpdates = new LongAdder();

    @PostConstruct
    public void init() {
        if (prefixObserver.settings().observe()) {
            prefixScanner = java.util.concurrent.Executors.newSingleThreadScheduledExecutor(task -> {
                Thread thread = new Thread(task, "prefix-cache-observe");
                thread.setDaemon(true);
                return thread;
            });
            long period = prefixObserver.settings().scanMillis();
            prefixScanner.scheduleAtFixedRate(() -> {
                try {
                    var scan = prefixObserver.scanSnapshot();
                    var consumer = prefixActions.get();
                    if (consumer != null) { consumer.accept(scan); }
                } catch (RuntimeException exception) {
                    log.error("Prefix scan or asynchronous action submission failed", exception);
                }
            }, period, period, java.util.concurrent.TimeUnit.MILLISECONDS);
        }
        log.info("KvCacheManager initialized successfully");
    }

    public Runnable registerPrefixActions(java.util.function.Consumer<org.flexlb.cache.prefix.PrefixObserver.Scan> consumer) {
        java.util.Objects.requireNonNull(consumer);
        if (!prefixObserver.settings().adaptive() || !prefixActions.compareAndSet(null, consumer)) {
            throw new IllegalStateException("prefix actions require adaptive policy and one lifecycle owner");
        }
        return () -> prefixActions.compareAndSet(consumer, null);
    }

    @org.springframework.context.event.EventListener(org.springframework.context.event.ContextRefreshedEvent.class)
    public void requireAdaptiveLifecycle() {
        if (prefixObserver.settings().adaptive() && prefixActions.get() == null) {
            throw new IllegalStateException("adaptive policy requires a configured prefix recovery/action service");
        }
    }

    @PreDestroy
    public void destroy() {
        prefixActions.set(null);
        if (prefixScanner != null) { prefixScanner.shutdownNow(); }
        log.info("KvCacheManager shutting down...");
        clear();
    }

    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
                                                   RoleType roleType, String group) {
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }
        List<String> enginesIpPorts = workerStatusProvider.getWorkerIpPorts(roleType, group);
        return findMatchingEngines(blockCacheKeys, enginesIpPorts);
    }

    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys, List<String> candidateEngineIpPorts) {
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()
                || candidateEngineIpPorts == null || candidateEngineIpPorts.isEmpty()) {
            return Collections.emptyMap();
        }
        if (prefixObserver.settings().confirmed() && !prefixObserver.settings().mainCompatibleRouting()) {
            return prefixObserver.directory().matches(blockCacheKeys, candidateEngineIpPorts);
        }
        return globalCacheIndex.batchCalculatePrefixMatchLength(candidateEngineIpPorts, blockCacheKeys);
    }

    /**
     * Merge engine-reported cache keys into the local view + global index. Keys the
     * engine failed to report are NOT removed — engine snapshots are unreliable under
     * CP/PDFUSION, so we treat them as additive-only. Aging is handled by the per-engine
     * LRU inside {@link EngineLocalView}.
     */
    public void updateEngineCache(String engineIPort, String role, Set<Long> newCacheBlocks) {
        if (prefixObserver.settings().confirmed() && !prefixObserver.settings().mainCompatibleRouting()) { return; }
        if (engineIPort == null || newCacheBlocks == null || newCacheBlocks.isEmpty()) {
            return;
        }
        applyKeys(engineIPort, role, new java.util.ArrayList<>(newCacheBlocks));
    }

    /**
     * Record that FlexLB routed a request with {@code cacheKeys} to {@code engineIPort}.
     * Populates both the local view and the global index so the next request with the
     * same prefix sees a hit even before the engine acknowledges the blocks.
     */
    public void addCacheBlocks(String engineIPort, String role, List<Long> cacheKeys) {
        if (prefixObserver.settings().confirmed() && !prefixObserver.settings().mainCompatibleRouting()) { return; }
        if (engineIPort == null || cacheKeys == null || cacheKeys.isEmpty()) {
            return;
        }
        applyKeys(engineIPort, role, cacheKeys);
    }

    private void applyKeys(String engineIPort, String role, List<Long> keys) {
        List<Long> evicted = engineLocalView.addOrUpdateCacheBlocks(engineIPort, keys);
        for (int i = 0; i < keys.size(); i++) {
            Long k = keys.get(i);
            if (k == null) {
                continue;
            }
            globalCacheIndex.addCacheBlock(k, engineIPort);
            if (i < evicted.size()) {
                Long e = evicted.get(i);
                if (e != null && !e.equals(k)) {
                    globalCacheIndex.removeCacheBlock(engineIPort, e);
                }
            }
        }
        totalUpdates.increment();
        cacheMetricsReporter.reportEngineLocalMetrics(engineIPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }

    public void clear() {
        prefixObserver.clear();
        globalCacheIndex.clear();
        engineLocalView.clear();
        totalUpdates.reset();
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        log.info("Cleared all cache data");
    }
}
