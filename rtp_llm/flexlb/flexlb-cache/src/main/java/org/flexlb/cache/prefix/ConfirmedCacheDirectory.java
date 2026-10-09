package org.flexlb.cache.prefix;

import org.flexlb.dao.cache.CacheReportV2;
import java.time.Clock;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ConfirmedCacheDirectory {
    public record TopologyFence(Set<String> workers, Set<String> routes) {
        public TopologyFence {
            workers = Set.copyOf(workers);
            routes = Set.copyOf(routes);
            if (workers.isEmpty() || workers.size() > 4096 || routes.size() > 8192
                    || workers.stream().anyMatch(worker -> worker.isBlank() || worker.length() > 512)
                    || routes.stream().anyMatch(route -> route.isBlank() || route.length() > 1024)) {
                throw new IllegalArgumentException("invalid topology fence");
            }
        }
    }
    public final class OperationPermit implements AutoCloseable {
        private boolean closed;
        private OperationPermit() { }
        @Override
        public void close() {
            if (!closed) { closed = true; topologyLock.readLock().unlock(); }
        }
    }
    public final class MaintenancePermit implements AutoCloseable {
        private boolean closed;
        private MaintenancePermit() { }
        @Override public void close() {
            if (!closed) { topologyLock.writeLock().unlock(); closed = true; }
        }
    }
    public enum Result { APPLIED, NO_CHANGE, STALE, NEED_FULL, INVALID }
    public record Lane(String workerId, int dpLane) { }
    public record Snapshot(CacheReportV2 report, Set<Long> keys, long receivedMillis, Set<Long> unavailableKeys) {
        public Snapshot(CacheReportV2 report, Set<Long> keys, long receivedMillis) {
            this(report, keys, receivedMillis, Set.of());
        }
        public Snapshot {
            keys = Set.copyOf(keys);
            unavailableKeys = Set.copyOf(unavailableKeys);
        }
        public boolean available(List<Long> path) {
            return keys.containsAll(path) && path.stream().noneMatch(unavailableKeys::contains);
        }
    }
    private record Exclusion(String worker, int lane, String epoch, long version, Set<Long> keys) { }
    private final Map<String, Exclusion> exclusions = new HashMap<>();
    private final Map<Lane, Snapshot> lanes = new HashMap<>();
    private final Map<String, Lane> routes = new HashMap<>();
    private final Map<Lane, Set<String>> retiredEpochs = new HashMap<>();
    private final String namespace;
    private final Clock clock;
    private final PrefixLatencyMetrics lookupLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics directoryLookupLatency = new PrefixLatencyMetrics();
    private final long ttlMillis;
    private Integer blockSize;
    private final java.util.concurrent.locks.ReentrantReadWriteLock topologyLock =
            new java.util.concurrent.locks.ReentrantReadWriteLock(true);
    private Map<String, TopologyFence> topologyFences = Map.of();
    private final Map<String, Map<Lane, String>> admittedEpochs = new HashMap<>();
    private record Routing(String worker, String epoch, int degree, int lanes) { }
    private final Map<String, Routing> admittedRoutes = new HashMap<>();
    private final Set<String> topologyRouteBases = new HashSet<>();
    private boolean topologyHealthy = true;
    private boolean cacheOperationsPaused;
    private java.util.function.BooleanSupplier topologyGuard = () -> true;
    private long topologyExpiresMillis = Long.MAX_VALUE;

    public ConfirmedCacheDirectory(String namespace, Clock clock, long ttlMillis) {
        this.namespace = namespace;
        this.clock = clock;
        this.ttlMillis = ttlMillis;
    }

    public synchronized Result apply(String route, CacheReportV2 report) {
        if (report == null || !namespace.equals(report.namespace()) || report.workerId() == null
                || report.workerId().isBlank() || report.epoch() == null || report.epoch().isBlank()
                || report.layoutFingerprint() == null || report.layoutFingerprint().isBlank()
                || report.dpLane() < 0 || report.cpDegree() < 1 || report.blockSize() < 1
                || report.version() < 0 || report.rankReadinessGeneration() < 0
                || (blockSize != null && blockSize != report.blockSize())
                || report.physicalFreeBytes() < 0 || report.physicalBlockBytes() < 0
                || (report.physicalMemoryStatsValid() && !report.memoryStatsConsistent())) {
            return Result.INVALID;
        }
        Lane lane = new Lane(report.workerId(), report.dpLane());
        var admitted = admittedEpochs.get(report.workerId());
        if (admitted != null && !report.epoch().equals(admitted.get(lane))) { return Result.STALE; }
        Snapshot previous = lanes.get(lane);
        boolean sameEpoch = previous != null && previous.report().epoch().equals(report.epoch());
        if (retiredEpochs.getOrDefault(lane, Set.of()).contains(report.epoch())) {
            return Result.STALE;
        }
        if (sameEpoch && (report.version() < previous.report().version()
                || report.rankReadinessGeneration() < previous.report().rankReadinessGeneration())) {
            return Result.STALE;
        }
        if (sameEpoch && (!report.layoutFingerprint().equals(previous.report().layoutFingerprint())
                || report.blockSize() != previous.report().blockSize()
                || report.cpDegree() != previous.report().cpDegree())) {
            return Result.INVALID;
        }
        Set<Long> keys;
        if ("FULL".equals(report.snapshotKind())) {
            if (sameEpoch && report.version() == previous.report().version()
                    && report.deviceReady() == previous.report().deviceReady()
                    && "FULL".equals(previous.report().snapshotKind())
                    && report.deviceKeys().equals(previous.report().deviceKeys())) {
                keys = previous.keys();
            } else {
                keys = report.deviceReady() ? Set.copyOf(report.deviceKeys()) : Set.of();
                if (sameEpoch && report.version() == previous.report().version()
                        && (!keys.equals(previous.keys()) || report.deviceReady() != previous.report().deviceReady())) {
                    return Result.INVALID;
                }
            }
        } else if ("NO_CHANGE".equals(report.snapshotKind())) {
            if (!report.deviceKeys().isEmpty()) { return Result.INVALID; }
            if (!sameEpoch || report.version() != previous.report().version()
                    || report.deviceReady() != previous.report().deviceReady()) {
                return Result.NEED_FULL;
            }
            keys = previous.keys();
        } else {
            return Result.NEED_FULL;
        }
        if (previous != null && !sameEpoch) {
            retiredEpochs.computeIfAbsent(lane, ignored -> new HashSet<>()).add(previous.report().epoch());
        }
        lanes.put(lane, new Snapshot(report, keys, clock.millis()));
        exclusions.values().removeIf(exclusion -> exclusion.worker().equals(report.workerId())
                && exclusion.lane() == report.dpLane() && (!exclusion.epoch().equals(report.epoch())
                || exclusion.version() < report.version()));
        blockSize = report.blockSize();
        routes.entrySet().removeIf(entry -> entry.getValue().equals(lane) && !entry.getKey().equals(route));
        routes.put(route, lane);
        return "FULL".equals(report.snapshotKind()) ? Result.APPLIED : Result.NO_CHANGE;
    }

    private Map<String, Snapshot> rawView() {
        Map<String, Snapshot> result = new HashMap<>();
        routes.forEach((route, lane) -> {
            Snapshot snapshot = lanes.get(lane);
            if (snapshot != null && clock.millis() - snapshot.receivedMillis() < ttlMillis) {
                Set<Long> unavailable = new HashSet<>();
                exclusions.values().stream().filter(exclusion -> exclusion.worker().equals(lane.workerId())
                        && exclusion.lane() == lane.dpLane() && exclusion.epoch().equals(snapshot.report().epoch())
                        && exclusion.version() >= snapshot.report().version()).forEach(exclusion -> unavailable.addAll(exclusion.keys()));
                result.put(route, new Snapshot(snapshot.report(), snapshot.keys(), snapshot.receivedMillis(), unavailable));
            }
        });
        return Map.copyOf(result);
    }

    public synchronized Map<String, Snapshot> view() {
        Map<String, Snapshot> result = new HashMap<>();
        rawView().forEach((route, snapshot) -> {
            if (routeAvailable(route) && workerAvailable(snapshot.report().workerId())) { result.put(route, snapshot); }
        });
        return Map.copyOf(result);
    }

    public synchronized Map<String, Snapshot> preservationView(Set<String> affectedRoutes) {
        var current = rawView();
        var result = new HashMap<String, Snapshot>();
        for (String route : affectedRoutes) {
            var snapshot = current.get(route);
            if (snapshot == null || !snapshot.report().deviceReady()) {
                throw new IllegalStateException("preservation requires current inventory for every affected lane");
            }
            result.put(route, snapshot);
        }
        return Map.copyOf(result);
    }

    public synchronized boolean workerAvailable(String worker) {
        return topologyHealthy && topologyGuard.getAsBoolean() && clock.millis() < topologyExpiresMillis
                && topologyFences.values().stream().noneMatch(fence -> fence.workers().contains(worker));
    }

    public synchronized boolean routeAvailable(String route) {
        if (!topologyHealthy || !topologyGuard.getAsBoolean() || clock.millis() >= topologyExpiresMillis) { return false; }
        var lane = routes.get(route);
        return topologyFences.values().stream().noneMatch(fence ->
                (lane != null && fence.workers().contains(lane.workerId())) || fence.routes().stream().anyMatch(base ->
                        route.equals(base) || route.startsWith(base + "#dp")));
    }

    public synchronized boolean routingAvailable(String base, String route, long degree, long laneCount) {
        if (!routeAvailable(base) || !routeAvailable(route)) { return false; }
        if (!topologyRouteBases.contains(base)) { return true; }
        var expected = admittedRoutes.get(route);
        var lane = routes.get(route);
        var snapshot = lane == null ? null : lanes.get(lane);
        return expected != null && expected.degree() == degree && expected.lanes() == laneCount
                && workerAvailable(expected.worker()) && snapshot != null
                && expected.worker().equals(snapshot.report().workerId()) && expected.epoch().equals(snapshot.report().epoch())
                && snapshot.report().deviceReady() && clock.millis() - snapshot.receivedMillis() < ttlMillis;
    }

    public static String routeBase(String route) { return route.replaceFirst("#dp[0-9]+$", ""); }

    public OperationPermit acquireOperations(Set<String> workers) {
        return acquireReadPermit(workers, false);
    }

    public OperationPermit acquireRouting() { return acquireReadPermit(Set.of(), true); }

    // Existing durable operations must settle before a topology writer can
    // proceed. The maintenance lock still excludes recovery from native commit.
    public OperationPermit acquireRecovery() { return acquireReadPermit(Set.of(), true); }

    private OperationPermit acquireReadPermit(Set<String> workers, boolean routing) {
        try { topologyLock.readLock().lockInterruptibly(); }
        catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted waiting for topology admission", interrupted);
        }
        synchronized (this) {
            if (!topologyHealthy || !topologyGuard.getAsBoolean() || clock.millis() >= topologyExpiresMillis || (!routing && cacheOperationsPaused)
                    || workers.stream().anyMatch(worker -> !workerAvailable(worker))) {
                topologyLock.readLock().unlock();
                throw new IllegalStateException("topology fences new cache operations");
            }
        }
        return new OperationPermit();
    }

    public synchronized void invalidateTopologyObservation() { topologyHealthy = false; }

    public synchronized void topologyGuard(java.util.function.BooleanSupplier guard) {
        topologyGuard = java.util.Objects.requireNonNull(guard);
        topologyHealthy = false;
    }

    public MaintenancePermit acquireMaintenance() {
        invalidateTopologyObservation();
        try { topologyLock.writeLock().lockInterruptibly(); }
        catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted acquiring topology maintenance", interrupted);
        }
        return new MaintenancePermit();
    }

    public void suspendTopology() {
        synchronized (this) { topologyHealthy = false; }
        try { topologyLock.writeLock().lockInterruptibly(); }
        catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted draining local topology operations", interrupted);
        }
        try { synchronized (this) { topologyHealthy = false; } }
        finally { topologyLock.writeLock().unlock(); }
    }

    public void publishTopologyFences(Map<String, TopologyFence> fences, long validForMillis) {
        publishTopologyFences(fences, validForMillis, false);
    }

    public void publishTopologyFences(Map<String, TopologyFence> fences, long validForMillis, boolean pauseCacheOperations) {
        if (fences.size() > 4096 || validForMillis < 1 || validForMillis > 300000) {
            throw new IllegalArgumentException("invalid topology fence observation");
        }
        var copy = Map.copyOf(fences);
        long expires = Math.addExact(clock.millis(), validForMillis);
        topologyLock.writeLock().lock();
        try {
            synchronized (this) {
                topologyFences = copy;
                cacheOperationsPaused = pauseCacheOperations;
                topologyHealthy = true;
                topologyExpiresMillis = expires;
            }
        } finally { topologyLock.writeLock().unlock(); }
    }

    public synchronized void installTopologyReports(Set<String> workers, Map<String, CacheReportV2> reports) {
        if (topologyHealthy) { throw new IllegalStateException("topology placement requires suspended admission"); }
        if (workers.isEmpty() || workers.size() > 4096 || reports.isEmpty() || reports.size() > 4096) {
            throw new IllegalArgumentException("empty topology placement");
        }
        var validated = new ConfirmedCacheDirectory(namespace, clock, ttlMillis);
        var uniqueLanes = new HashSet<Lane>();
        var retained = new HashMap<String, Snapshot>();
        for (var entry : reports.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isBlank() || entry.getKey().length() > 1024
                    || entry.getValue() == null || !workers.contains(entry.getValue().workerId()) || !entry.getValue().deviceReady()
                    || !uniqueLanes.add(new Lane(entry.getValue().workerId(), entry.getValue().dpLane()))
                    || (routes.containsKey(entry.getKey()) && !workers.contains(routes.get(entry.getKey()).workerId()))
                    || !"FULL".equals(entry.getValue().snapshotKind())
                    || validated.apply(entry.getKey(), entry.getValue()) != Result.APPLIED
                    || (blockSize != null && blockSize != entry.getValue().blockSize())) {
                throw new IllegalArgumentException("invalid ready topology FULL");
            }
            var current = lanes.get(new Lane(entry.getValue().workerId(), entry.getValue().dpLane()));
            if (current != null && current.report().epoch().equals(entry.getValue().epoch())
                    && (current.report().version() > entry.getValue().version()
                    || current.report().rankReadinessGeneration() > entry.getValue().rankReadinessGeneration())) {
                var incoming = entry.getValue();
                var newer = current.report();
                // Status polling can finish after the all-rank topology inspection.
                // Keep its newer evidence only when it describes this exact ready lane.
                if (!newer.deviceReady() || newer.version() < incoming.version()
                        || newer.rankReadinessGeneration() < incoming.rankReadinessGeneration()
                        || !newer.layoutFingerprint().equals(incoming.layoutFingerprint())
                        || newer.cpDegree() != incoming.cpDegree() || newer.blockSize() != incoming.blockSize()
                        || newer.physicalBlockBytes() != incoming.physicalBlockBytes()
                        || !new Lane(newer.workerId(), newer.dpLane()).equals(routes.get(entry.getKey()))) {
                    throw new IllegalStateException("topology FULL conflicts with newer current report");
                }
                retained.put(entry.getKey(), current);
            }
        }
        for (var entry : lanes.entrySet()) {
            if (workers.contains(entry.getKey().workerId())) {
                retiredEpochs.computeIfAbsent(entry.getKey(), ignored -> new HashSet<>()).add(entry.getValue().report().epoch());
            }
        }
        lanes.keySet().removeIf(lane -> workers.contains(lane.workerId()));
        routes.forEach((route, lane) -> { if (workers.contains(lane.workerId())) { topologyRouteBases.add(routeBase(route)); } });
        routes.entrySet().removeIf(entry -> workers.contains(entry.getValue().workerId()));
        admittedRoutes.entrySet().removeIf(entry -> workers.contains(entry.getValue().worker()));
        for (String worker : workers) { admittedEpochs.put(worker, new HashMap<>()); }
        for (var entry : reports.entrySet()) {
            var report = entry.getValue();
            var lane = new Lane(report.workerId(), report.dpLane());
            admittedEpochs.get(report.workerId()).put(lane, report.epoch());
            retiredEpochs.computeIfAbsent(lane, ignored -> new HashSet<>()).remove(report.epoch());
            var snapshot = retained.get(entry.getKey());
            lanes.put(lane, snapshot != null ? snapshot : new Snapshot(report, Set.copyOf(report.deviceKeys()), clock.millis()));
            routes.put(entry.getKey(), lane);
            topologyRouteBases.add(routeBase(entry.getKey()));
            admittedRoutes.put(entry.getKey(), new Routing(report.workerId(), report.epoch(), report.cpDegree(), reports.size()));
            blockSize = report.blockSize();
        }
    }

    public synchronized Map<String, Integer> matches(List<Long> keys, List<String> candidates) {
        var current = view();
        return directoryLookupLatency.measure(() -> matches(current, keys, candidates));
    }

    public Map<String, Integer> measuredMatches(Map<String, Snapshot> current, List<Long> keys, List<String> candidates) {
        return lookupLatency.measure(() -> matches(current, keys, candidates));
    }

    public PrefixLatencyMetrics.Snapshot lookupLatency() { return lookupLatency.snapshot(); }

    public PrefixLatencyMetrics.Snapshot directoryLookupLatency() { return directoryLookupLatency.snapshot(); }

    public static Map<String, Integer> matches(Map<String, Snapshot> current, List<Long> keys, List<String> candidates) {
        Map<String, Integer> result = new HashMap<>();
        if (keys == null || candidates == null) { return Map.of(); }
        for (String route : candidates) {
            Snapshot snapshot = current.get(route);
            int matched = 0;
            if (snapshot != null) {
                while (matched < keys.size() && snapshot.keys().contains(keys.get(matched))
                        && !snapshot.unavailableKeys().contains(keys.get(matched))) {
                    matched++;
                }
            }
            result.put(route, matched);
        }
        return result;
    }

    public synchronized void invalidate(String route) { routes.remove(route); }
    public synchronized void exclude(String id, String worker, int lane, String epoch, long version, List<Long> keys) {
        if (id == null || id.isBlank() || worker == null || worker.isBlank() || lane < 0
                || epoch == null || epoch.isBlank() || version < 0 || keys.isEmpty() || keys.size() > 8192) {
            throw new IllegalArgumentException("invalid pending-delete exclusion");
        }
        var exclusion = new Exclusion(worker, lane, epoch, version, Set.copyOf(keys));
        if (exclusions.containsKey(id) && !exclusions.get(id).equals(exclusion)) {
            throw new IllegalStateException("pending-delete identity changed");
        }
        if (!exclusions.containsKey(id) && exclusions.size() >= 4096) {
            throw new IllegalStateException("pending-delete capacity exhausted");
        }
        exclusions.put(id, exclusion);
    }
    public synchronized void include(String id) { exclusions.remove(id); }
    public synchronized void clear() {
        lanes.clear();
        routes.clear();
        retiredEpochs.clear();
        exclusions.clear();
        blockSize = null;
    }
}
