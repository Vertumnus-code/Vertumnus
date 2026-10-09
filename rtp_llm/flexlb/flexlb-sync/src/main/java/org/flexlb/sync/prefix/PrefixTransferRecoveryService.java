package org.flexlb.sync.prefix;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.time.Clock;
import java.util.concurrent.TimeUnit;

@Component
@ConditionalOnProperty(name = "prefix-cache.recovery.enabled", havingValue = "true")
public final class PrefixTransferRecoveryService implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(PrefixTransferRecoveryService.class);
    private final KvCacheManager cacheManager;
    private final EngineGrpcClient grpc;
    private final Environment environment;
    private CuratorFramework client;
    private ZookeeperPrefixControllerAuthority authority;
    private PrefixTransferRecoveryLoop loop;
    private PrefixTopologyRecoveryLoop topologyLoop;
    private PrefixTopologyPlacement placement;
    private ZookeeperPrefixTopologyBarrier topologyBarrier;
    private PrefixTopologyWriter topologyWriter;
    private PrefixTopologyControl topologyControl;
    private PrefixActionLoop actions;
    private Runnable unregisterActions;
    private boolean closed;

    public PrefixTransferRecoveryService(KvCacheManager cacheManager, EngineGrpcClient grpc, Environment environment) {
        this.cacheManager = cacheManager;
        this.grpc = grpc;
        this.environment = environment;
    }

    @PostConstruct
    public synchronized void start() throws Exception {
        if (closed) { throw new IllegalStateException("prefix recovery service is closed"); }
        if (loop != null) { return; }
        var observer = cacheManager.getPrefixObserver();
        if (!observer.settings().confirmed() || observer.settings().namespace().isBlank()) {
            throw new IllegalStateException("prefix recovery requires confirmed cache metadata and namespace");
        }
        String connection = environment.getRequiredProperty("prefix-cache.zookeeper.connect");
        if (connection.isBlank()) { throw new IllegalArgumentException("missing prefix ZooKeeper connection"); }
        int timeout = environment.getProperty("prefix-cache.recovery.rpc-timeout-ms", Integer.class, 5000);
        if (timeout < 100 || timeout > 300000) { throw new IllegalArgumentException("invalid prefix RPC timeout"); }
        var limits = new PrefixTransferRecoveryLoop.Limits(
                environment.getProperty("prefix-cache.recovery.interval-ms", Long.class, 1000L),
                environment.getProperty("prefix-cache.recovery.max-backoff-ms", Long.class, 60000L),
                environment.getProperty("prefix-cache.recovery.actions-per-tick", Integer.class, 4),
                environment.getProperty("prefix-cache.recovery.retention-ms", Long.class, 60000L));
        String namespace = observer.settings().namespace();
        boolean topologyRecovery = environment.getProperty("prefix-cache.topology.recovery.enabled", Boolean.class, false);
        boolean topologyPlacement = environment.getProperty("prefix-cache.topology.placement.enabled", Boolean.class, false);
        boolean writerEnabled = environment.getProperty("prefix-cache.topology.writer.enabled", Boolean.class, false);
        boolean controlEnabled = environment.getProperty("prefix-cache.topology.control.enabled", Boolean.class, false);
        if (controlEnabled && !writerEnabled) { throw new IllegalArgumentException("topology control requires writer"); }
        if (topologyPlacement && !topologyRecovery) { throw new IllegalArgumentException("topology placement requires durable recovery"); }
        if (writerEnabled && (!topologyPlacement || !observer.settings().observe()
                || !environment.getProperty("prefix-cache.topology.writer.fenced-ingress-only", Boolean.class, false))) {
            throw new IllegalArgumentException("topology writer requires placement and exclusively fenced CP-aware ingress");
        }
        try {
            var mapper = new ObjectMapper();
            PrefixWorkerRegistry registry = null;
            if (observer.settings().adaptive() || topologyPlacement) {
                registry = new PrefixWorkerRegistry(mapper.readValue(environment.getRequiredProperty("prefix-cache.adaptive.workers"),
                        new TypeReference<java.util.List<PrefixWorkerRegistry.Registration>>() { }));
            }
            client = CuratorFrameworkFactory.builder().connectString(connection)
                    .sessionTimeoutMs(15000).connectionTimeoutMs(5000)
                    .retryPolicy(new ExponentialBackoffRetry(100, 3)).build();
            client.start();
            if (!client.blockUntilConnected(10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("prefix recovery ZooKeeper connection not confirmed");
            }
            var journal = new ZookeeperPrefixTransferJournal(client,
                    environment.getProperty("prefix-cache.journal.root", "/prefix-cache/transfer-journal"), namespace,
                    environment.getProperty("prefix-cache.journal.capacity", Integer.class, 1024));
            authority = new ZookeeperPrefixControllerAuthority(client,
                    environment.getProperty("prefix-cache.controller.root", "/prefix-cache/controllers"), namespace);
            authority.start();
            if (topologyRecovery) {
                var topologyJournal = new ZookeeperPrefixTopologyJournal(client,
                        environment.getProperty("prefix-cache.topology.journal.root", "/prefix-cache/topology-journal"),
                        namespace, environment.getProperty("prefix-cache.topology.journal.capacity", Integer.class, 1024), Clock.systemUTC(), writerEnabled);
                var topologyCoordinator = new PrefixTopologyCoordinator(grpc, authority, topologyJournal, Clock.systemUTC(), timeout);
                if (topologyPlacement) {
                    placement = new PrefixTopologyPlacement(topologyJournal, authority, grpc, observer.directory(), registry, timeout,
                            environment.getProperty("prefix-cache.topology.placement.observation-ms", Long.class, 10000L));
                    if (writerEnabled) {
                        placement.attachFrontend(new PrefixTopologyFrontend(timeout));
                        topologyBarrier = new ZookeeperPrefixTopologyBarrier(client,
                                environment.getProperty("prefix-cache.topology.readers.root", "/prefix-cache/topology-readers"), namespace,
                                environment.getRequiredProperty("prefix-cache.topology.reader-id"),
                                mapper.readValue(environment.getRequiredProperty("prefix-cache.topology.required-readers"),
                                        new TypeReference<java.util.Set<String>>() { }), observer, PrefixTransferRecoveryService::frontendDrained,
                                Clock.systemUTC(), environment.getProperty("prefix-cache.topology.reader-ttl-ms", Long.class, 10000L));
                        placement.attachBarrier(topologyBarrier);
                        var settings = observer.settings();
                        int maxPaths = environment.getProperty("prefix-cache.topology.preservation.max-paths", Integer.class, 48);
                        var copyLimits = new PrefixCopyExecutor.Limits(maxPaths, maxPaths,
                                Math.max(1, settings.destinationBudgetBytes()), Math.max(1, settings.wireBudgetBytes()), settings.safetyBytes(),
                                timeout, environment.getProperty("prefix-cache.adaptive.transfer-timeout-ms", Long.class, 30000L));
                        topologyWriter = new PrefixTopologyWriter(observer, registry, grpc, authority, journal, topologyJournal,
                                topologyCoordinator, placement, topologyBarrier, copyLimits,
                                mapper.readValue(environment.getProperty("prefix-cache.adaptive.cross-degree-pairs", "[]"),
                                        new TypeReference<java.util.Set<PrefixCopyExecutor.DegreePair>>() { }), Clock.systemUTC());
                        if (controlEnabled) {
                            topologyControl = new PrefixTopologyControl(mapper.readValue(environment.getRequiredProperty("prefix-cache.topology.worlds"),
                                    new TypeReference<java.util.Map<String, java.util.List<PrefixTopologyControl.Rank>>>() { }), registry, authority,
                                    topologyJournal, topologyCoordinator, topologyWriter, placement, observer.directory(), grpc, namespace, Clock.systemUTC(), timeout);
                        }
                    }
                    try { placement.refresh(); }
                    catch (IllegalStateException unavailable) {
                        observer.directory().suspendTopology();
                        if (topologyBarrier == null || Thread.currentThread().isInterrupted()) { throw unavailable; }
                        // Frontend registration and fresh inventory can arrive only
                        // after application startup. The recovery loop retries while fenced.
                        LOG.warn("Initial topology projection deferred while routing remains fenced: {}", unavailable.getMessage());
                    }
                }
                topologyLoop = new PrefixTopologyRecoveryLoop(topologyJournal, topologyCoordinator, Clock.systemUTC(),
                        new PrefixTopologyRecoveryLoop.Limits(
                                environment.getProperty("prefix-cache.topology.recovery.interval-ms", Long.class, 1000L),
                                environment.getProperty("prefix-cache.topology.recovery.max-backoff-ms", Long.class, 60000L),
                                environment.getProperty("prefix-cache.topology.recovery.actions-per-tick", Integer.class, 4)), placement);
            }
            var coordinator = new PrefixTransferCoordinator(new GrpcPrefixTransferWorkerClient(grpc, timeout), authority, journal);
            loop = new PrefixTransferRecoveryLoop(journal, coordinator, authority, observer.directory(), namespace,
                    Clock.systemUTC(), limits);
            if (observer.settings().adaptive()) {
                var settings = observer.settings();
                java.util.Set<PrefixCopyExecutor.DegreePair> pairs = mapper.readValue(
                        environment.getProperty("prefix-cache.adaptive.cross-degree-pairs", "[]"), new TypeReference<>() { });
                var actionLimits = new PrefixActionLoop.Limits(
                        environment.getProperty("prefix-cache.adaptive.max-candidates", Integer.class, 48),
                        settings.operationsPerList(),
                        environment.getProperty("prefix-cache.adaptive.max-tracked-paths", Integer.class, 1024),
                        environment.getProperty("prefix-cache.adaptive.scan-ttl-ms", Long.class, settings.reportTtlMillis()),
                        environment.getProperty("prefix-cache.adaptive.cooldown-ms", Long.class, 30000L),
                        environment.getProperty("prefix-cache.adaptive.initial-backoff-ms", Long.class, 1000L),
                        environment.getProperty("prefix-cache.adaptive.max-backoff-ms", Long.class, 60000L),
                        environment.getProperty("prefix-cache.adaptive.window-ms", Long.class, 60000L),
                        environment.getProperty("prefix-cache.adaptive.window-wire-bytes", Long.class, settings.wireBudgetBytes()),
                        environment.getProperty("prefix-cache.adaptive.reclaim-bytes", Long.class, settings.destinationBudgetBytes()),
                        environment.getProperty("prefix-cache.adaptive.max-foreground-tasks", Double.class, 1.0),
                        timeout,
                        environment.getProperty("prefix-cache.adaptive.transfer-timeout-ms", Long.class, 30000L),
                        environment.getProperty("prefix-cache.adaptive.round-timeout-ms", Long.class, 60000L));
                if (limits.retentionMillis() < Math.max(actionLimits.windowMillis(), actionLimits.cooldownMillis())) {
                    throw new IllegalArgumentException("durable retention must cover rate-window and cooldown across leadership changes");
                }
                var copyLimits = new PrefixCopyExecutor.Limits(actionLimits.maxCandidates(),
                        Math.min(actionLimits.maxCandidates(), actionLimits.perList() * 2),
                        Math.max(1, settings.destinationBudgetBytes()), Math.max(1, settings.wireBudgetBytes()), settings.safetyBytes(),
                        actionLimits.rpcTimeoutMillis(), actionLimits.transferTimeoutMillis());
                var copies = new PrefixCopyExecutor(grpc, authority, journal, observer.directory(), namespace, copyLimits, pairs);
                var reclaims = new PrefixReclaimCoordinator(new GrpcPrefixTransferWorkerClient(grpc, timeout), authority, journal, observer.directory());
                actions = new PrefixActionLoop(observer, registry, grpc, authority, journal, copies, reclaims, actionLimits, Clock.systemUTC());
                unregisterActions = cacheManager.registerPrefixActions(scan -> actions.submit(scan));
            }
            loop.start();
            if (topologyLoop != null) { topologyLoop.start(); }
        } catch (Exception failure) {
            try {
                close();
            } catch (RuntimeException cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    public synchronized PrefixTransferRecoveryLoop.Status status() {
        return loop == null ? new PrefixTransferRecoveryLoop.Status(0, 0, 0, 0, 0, 0, 0) : loop.status();
    }

    public synchronized PrefixTopologyRecoveryLoop.Status topologyStatus() {
        return topologyLoop == null ? new PrefixTopologyRecoveryLoop.Status(0, 0, 0, 0, 0, 0) : topologyLoop.status();
    }

    public synchronized ZookeeperPrefixTopologyJournal.Entry admitTopology(String switchId,
            java.util.List<PrefixWorkerRegistry.Registration> registrations) {
        if (closed || placement == null) { throw new IllegalStateException("topology placement is not enabled"); }
        return placement.admit(switchId, registrations);
    }

    public synchronized ZookeeperPrefixTopologyJournal.Entry advanceTopology(PrefixTopologyPlan plan) {
        if (closed || topologyWriter == null) { throw new IllegalStateException("topology writer is not enabled"); }
        return topologyWriter.advance(plan);
    }

    public synchronized PrefixTopologyControl topologyControl() {
        if (closed || topologyControl == null) { throw new IllegalStateException("topology control is not enabled"); }
        return topologyControl;
    }

    private static boolean frontendDrained(java.util.Set<String> routes) {
        for (var role : org.flexlb.dao.route.RoleType.values()) {
            var workers = org.flexlb.sync.status.EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role);
            if (workers == null) { continue; }
            if (workers.values().stream().anyMatch(worker -> worker != null && routes.contains(worker.getIpPort())
                    && !worker.getLocalTaskMap().isEmpty())) { return false; }
        }
        return true;
    }

    @Override
    @PreDestroy
    public synchronized void close() {
        if (closed) { return; }
        closed = true;
        RuntimeException failure = null;
        try { if (unregisterActions != null) { unregisterActions.run(); } }
        catch (RuntimeException unregisterFailure) { failure = unregisterFailure; }
        for (AutoCloseable component : java.util.Arrays.<AutoCloseable>asList(topologyLoop, actions, loop, topologyBarrier, authority, client)) {
            if (component == null) { continue; }
            try { component.close(); }
            catch (Exception closeFailure) {
                if (failure == null) { failure = new IllegalStateException("prefix recovery shutdown failed", closeFailure); }
                else { failure.addSuppressed(closeFailure); }
            }
        }
        if (placement != null) {
            try { cacheManager.getPrefixObserver().directory().suspendTopology(); }
            catch (RuntimeException suspendFailure) {
                if (failure == null) { failure = suspendFailure; }
                else { failure.addSuppressed(suspendFailure); }
            }
        }
        if (failure != null) { throw failure; }
    }
}
