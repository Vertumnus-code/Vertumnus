package org.flexlb.sync.prefix;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.framework.state.ConnectionStateListener;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.common.PathUtils;
import org.apache.zookeeper.data.Stat;
import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.cache.prefix.PrefixObserver;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Clock;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;

public final class ZookeeperPrefixTopologyBarrier implements AutoCloseable {
    public record Acknowledgement(String instance, long expiresMillis, Set<String> plans,
                                  Set<String> workers, Set<String> routes, PrefixObserver.PreservationSnapshot demand) {
        public Acknowledgement { plans = Set.copyOf(plans); workers = Set.copyOf(workers); routes = Set.copyOf(routes); }
    }
    public record Reader(String instance, long session, long creation) { }
    public record Cohort(String switchId, Map<String, Reader> readers,
                         List<PrefixObserver.PreservationSnapshot> demands) {
        public Cohort { readers = Map.copyOf(readers); demands = List.copyOf(demands); }
    }

    private final CuratorFramework client;
    private final PrefixObserver observer;
    private final Predicate<Set<String>> drained;
    private final Clock clock;
    private final long ttlMillis;
    private final Set<String> required;
    private final String readerId;
    private final String instance = UUID.randomUUID().toString();
    private final String root;
    private final byte[] configuration;
    private final ObjectMapper mapper = new ObjectMapper();
    private final ConnectionStateListener listener;
    private volatile boolean connected;
    private final AtomicLong sessionLosses = new AtomicLong();
    private volatile long registeredSessionLosses;
    private volatile boolean closed;
    private volatile long validUntil;

    public ZookeeperPrefixTopologyBarrier(CuratorFramework client, String path, String namespace, String readerId,
                                          Set<String> required, PrefixObserver observer, Predicate<Set<String>> drained,
                                          Clock clock, long ttlMillis) {
        this.client = Objects.requireNonNull(client);
        this.observer = Objects.requireNonNull(observer);
        this.drained = Objects.requireNonNull(drained);
        this.clock = Objects.requireNonNull(clock);
        this.ttlMillis = ttlMillis;
        this.required = Set.copyOf(required);
        this.readerId = Objects.requireNonNull(readerId);
        PathUtils.validatePath(path);
        if (path.equals("/") || namespace.isBlank() || !namespace.equals(observer.settings().namespace())
                || required.isEmpty() || required.size() > 128 || !required.contains(readerId)
                || required.stream().anyMatch(id -> !id.matches("[A-Za-z0-9_-]{1,128}"))
                || ttlMillis < 100 || ttlMillis > 300000) {
            throw new IllegalArgumentException("invalid topology reader cohort");
        }
        listener = (unused, state) -> {
            connected = state == ConnectionState.CONNECTED || state == ConnectionState.RECONNECTED;
            if (state == ConnectionState.LOST) { sessionLosses.incrementAndGet(); }
            if (!connected) { validUntil = 0; observer.directory().invalidateTopologyObservation(); }
        };
        client.getConnectionStateListenable().addListener(listener);
        try {
            root = path + "/" + HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(namespace.getBytes(StandardCharsets.UTF_8)));
            configuration = mapper.writeValueAsBytes(List.of("prefix-topology-readers-v1", ttlMillis,
                    required.stream().sorted().toList()));
            try { client.create().creatingParentsIfNeeded().forPath(root, configuration); }
            catch (KeeperException.NodeExistsException exists) {
                if (!java.util.Arrays.equals(configuration, client.getData().forPath(root))) {
                    throw new IllegalStateException("incoherent topology reader configuration");
                }
            }
            registerAfterPreviousSession();
            connected = client.getZookeeperClient().isConnected();
        } catch (Exception failure) {
            if (failure instanceof InterruptedException) { Thread.currentThread().interrupt(); }
            client.getConnectionStateListenable().removeListener(listener);
            throw new IllegalStateException("cannot register topology reader", failure);
        }
    }

    private void registerAfterPreviousSession() throws Exception {
        var session = client.getZookeeperClient().getZooKeeper();
        long waitMillis = Math.min(60000L, Math.max(1000L, 2L * session.getSessionTimeout()));
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(waitMillis);
        byte[] initial = encode(new Acknowledgement(instance, 0, Set.of(), Set.of(), Set.of(), emptyDemand()));
        while (true) {
            requireConfiguration();
            try {
                client.create().withMode(CreateMode.EPHEMERAL).forPath(root + "/" + readerId, initial);
                return;
            } catch (KeeperException.NodeExistsException exists) {
                var owner = client.checkExists().forPath(root + "/" + readerId);
                // A crashed predecessor retains its ephemeral node until session
                // expiration. Wait for release without taking over a live reader.
                if (owner != null && (owner.getEphemeralOwner() == 0
                        || owner.getEphemeralOwner() == session.getSessionId())) { throw exists; }
                if (System.nanoTime() >= deadline) { throw exists; }
                Thread.sleep(50);
            }
        }
    }

    public boolean healthy() {
        return !closed && registeredSessionLosses == sessionLosses.get() && connected
                && client.getZookeeperClient().isConnected() && clock.millis() < validUntil;
    }

    private void rejoinCurrentSession(long observedLosses) throws Exception {
        validUntil = 0;
        observer.directory().invalidateTopologyObservation();
        requireConfiguration();
        try {
            client.create().withMode(CreateMode.EPHEMERAL).forPath(root + "/" + readerId,
                    encode(new Acknowledgement(instance, 0, Set.of(), Set.of(), Set.of(), emptyDemand())));
        } catch (KeeperException.NodeExistsException exists) {
            // A retry may observe our successful creation after an ambiguous reply.
        }
        Stat stat = new Stat();
        var current = read(readerId, stat);
        if (!instance.equals(current.instance())
                || stat.getEphemeralOwner() != client.getZookeeperClient().getZooKeeper().getSessionId()) {
            throw new IllegalStateException("topology reader belongs to another session");
        }
        // A concurrent LOST increments the counter again and keeps healthy() false.
        registeredSessionLosses = observedLosses;
    }

    public synchronized void publish(Map<String, ConfirmedCacheDirectory.TopologyFence> fences) {
        publish(fences, null);
    }

    public synchronized void publish(Map<String, ConfirmedCacheDirectory.TopologyFence> fences, Set<String> preservationRoutes) {
        if (closed || !connected) { throw new IllegalStateException("topology reader session is unavailable"); }
        long observedLosses = sessionLosses.get();
        if (registeredSessionLosses != observedLosses) {
            try { rejoinCurrentSession(observedLosses); }
            catch (Exception failure) { throw new IllegalStateException("cannot rejoin topology reader", failure); }
        }
        var routes = new java.util.HashSet<String>();
        var workers = new java.util.HashSet<String>();
        fences.values().forEach(fence -> routes.addAll(fence.routes()));
        fences.values().forEach(fence -> workers.addAll(fence.workers()));
        // Drain is established by request ownership, not by lossless telemetry delivery.
        // The snapshot also protects pending observations that still have held prefixes.
        var plans = !fences.isEmpty() && drained.test(Set.copyOf(routes))
                ? fences.keySet() : Set.<String>of();
        // A heartbeat without a drained plan cannot be captured for preservation.
        // Historical demand must not exhaust its znode budget and fence live routing.
        var demand = plans.isEmpty() ? emptyDemand() : preservationRoutes == null
                ? observer.preservationSnapshot() : observer.preservationSnapshot(preservationRoutes);
        var acknowledgement = new Acknowledgement(instance, Math.addExact(clock.millis(), ttlMillis), plans,
                workers, routes, demand);
        try {
            requireConfiguration();
            Stat stat = new Stat();
            var current = read(readerId, stat);
            if (!instance.equals(current.instance()) || stat.getEphemeralOwner() != client.getZookeeperClient().getZooKeeper().getSessionId()) {
                throw new IllegalStateException("topology reader identity changed");
            }
            client.setData().withVersion(stat.getVersion()).forPath(root + "/" + readerId, encode(acknowledgement));
            validUntil = acknowledgement.expiresMillis();
        } catch (Exception failure) {
            validUntil = 0;
            observer.directory().invalidateTopologyObservation();
            throw new IllegalStateException("cannot publish topology reader barrier", failure);
        }
    }

    public synchronized Cohort capture(String switchId) {
        if (!healthy()) { throw new IllegalStateException("local topology reader is not current"); }
        var readers = new HashMap<String, Reader>();
        var demands = new ArrayList<PrefixObserver.PreservationSnapshot>();
        try {
            requireConfiguration();
            var local = read(readerId, new Stat());
            if (!instance.equals(local.instance())) { throw new IllegalStateException("local topology reader was replaced"); }
            for (String id : required.stream().sorted().toList()) {
                Stat stat = new Stat();
                var acknowledgement = read(id, stat);
                if (stat.getEphemeralOwner() == 0 || acknowledgement.expiresMillis() <= clock.millis()
                        || !acknowledgement.plans().contains(switchId)
                        || !acknowledgement.workers().equals(local.workers()) || !acknowledgement.routes().equals(local.routes())
                        || !observer.settings().namespace().equals(acknowledgement.demand().namespace())) {
                    throw new IllegalStateException("topology reader has not drained the pending plan");
                }
                readers.put(id, new Reader(acknowledgement.instance(), stat.getEphemeralOwner(), stat.getCzxid()));
                demands.add(acknowledgement.demand());
            }
            if (!healthy()) { throw new IllegalStateException("local topology reader expired during cohort capture"); }
            return new Cohort(switchId, readers, demands);
        } catch (Exception failure) { throw new IllegalStateException("topology reader cohort is not confirmed", failure); }
    }

    public void verify(Cohort expected) {
        if (!capture(expected.switchId()).readers().equals(expected.readers())) {
            throw new IllegalStateException("topology reader cohort changed");
        }
    }

    private Acknowledgement read(String id, Stat stat) throws Exception {
        byte[] data = client.getData().storingStatIn(stat).forPath(root + "/" + id);
        if (data.length > 786432) { throw new IllegalStateException("topology reader snapshot exceeds bound"); }
        return mapper.readValue(data, Acknowledgement.class);
    }

    private void requireConfiguration() throws Exception {
        if (!java.util.Arrays.equals(configuration, client.getData().forPath(root))) {
            validUntil = 0;
            observer.directory().invalidateTopologyObservation();
            throw new IllegalStateException("topology reader configuration changed");
        }
    }

    private byte[] encode(Acknowledgement acknowledgement) throws Exception {
        byte[] data = mapper.writeValueAsBytes(acknowledgement);
        if (data.length > 786432) { throw new IllegalStateException("topology demand exceeds reader snapshot budget"); }
        return data;
    }

    private PrefixObserver.PreservationSnapshot emptyDemand() {
        return new PrefixObserver.PreservationSnapshot(observer.settings().namespace(), clock.millis(), List.of(), Map.of());
    }

    @Override public synchronized void close() {
        closed = true;
        validUntil = 0;
        observer.directory().invalidateTopologyObservation();
        client.getConnectionStateListenable().removeListener(listener);
        try {
            Stat stat = new Stat();
            if (instance.equals(read(readerId, stat).instance())
                    && stat.getEphemeralOwner() == client.getZookeeperClient().getZooKeeper().getSessionId()) {
                client.delete().withVersion(stat.getVersion()).forPath(root + "/" + readerId);
            }
        } catch (Exception unavailable) { }
    }
}
