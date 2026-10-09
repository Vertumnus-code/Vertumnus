package org.flexlb.sync.prefix;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.curator.framework.CuratorFramework;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.common.PathUtils;
import org.apache.zookeeper.data.Stat;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Result;
import org.flexlb.cache.prefix.ConfirmedCacheDirectory;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.time.Clock;

public final class ZookeeperPrefixTransferJournal implements PrefixTransferJournal {
    private record Payload(int schema, Intent intent, Result result) { }

    private static final int MAX_PAYLOAD_BYTES = 524288;
    private final CuratorFramework client;
    private final ObjectMapper mapper = new ObjectMapper();
    private final String namespace;
    private final String root;
    private final int capacity;
    private final Clock clock;

    public ZookeeperPrefixTransferJournal(CuratorFramework client, String journalRoot, String namespace, int capacity) {
        this(client, journalRoot, namespace, capacity, Clock.systemUTC());
    }

    public ZookeeperPrefixTransferJournal(CuratorFramework client, String journalRoot, String namespace, int capacity,
                                          Clock clock) {
        this.client = Objects.requireNonNull(client);
        this.clock = Objects.requireNonNull(clock);
        this.namespace = Objects.requireNonNull(namespace);
        this.capacity = capacity;
        PathUtils.validatePath(journalRoot);
        if (journalRoot.equals("/") || namespace.isBlank()
                || namespace.getBytes(StandardCharsets.UTF_8).length > 512 || capacity <= 0 || capacity > 4096) {
            throw new IllegalArgumentException("invalid journal configuration");
        }
        try {
            root = journalRoot + "/" + HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(namespace.getBytes(StandardCharsets.UTF_8)));
            byte[] configuration = ("prefix-transfer-journal-v1:" + capacity).getBytes(StandardCharsets.UTF_8);
            try {
                client.create().creatingParentsIfNeeded().forPath(root, configuration);
            } catch (KeeperException.NodeExistsException exists) {
                deadlineFloor(client.getData().forPath(root));
            }
        } catch (Exception failure) {
            throw new IllegalStateException("cannot initialize prefix transfer journal", failure);
        }
    }

    @Override
    public Entry create(Intent intent) {
        validate(intent);
        byte[] data = encode(new Payload(1, intent, null));
        for (int attempt = 0; attempt < 8; attempt++) {
            try {
                Stat guard = new Stat();
                byte[] configuration = client.getData().storingStatIn(guard).forPath(root);
                long floor = deadlineFloor(configuration);
                List<String> children = client.getChildren().forPath(root);
                if (children.contains(intent.requestId())) {
                    Entry existing = read(intent.requestId());
                    if (!existing.intent().equals(intent)) {
                        throw new IllegalStateException("durable transfer ID collision");
                    }
                    return existing;
                }
                if (intent.deadlineUnixMs() <= Math.max(floor, clock.millis())) {
                    throw new IllegalStateException("expired or retired transfer intent cannot be recreated");
                }
                if (children.size() >= capacity) {
                    throw new IllegalStateException("prefix transfer journal capacity exhausted");
                }
                for (String child : children) {
                    Entry pending = read(child);
                    if (pending.intent().destination().lane().equals(intent.destination().lane())
                            && pending.intent().destination().epoch().equals(intent.destination().epoch())
                            && pending.intent().keys().stream().anyMatch(intent.keys()::contains)
                            && (pending.result() == null || pending.result().cleanupPending()
                            || pending.result().outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN
                            || ((pending.result().committed()
                                || pending.result().outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED)
                                && (intent.destination().version() < pending.result().committedVersion()
                                    || intent.destination().rankCount() != pending.intent().destination().rankCount()
                                    || !intent.destination().layoutFingerprint().equals(pending.intent().destination().layoutFingerprint())
                                    || (intent.keys().getLast().equals(pending.intent().keys().getLast())
                                        && intent.action() == pending.intent().action()
                                        && intent.destination().version() <= pending.result().committedVersion()))))) {
                        throw new IllegalStateException("destination path already has a durable unresolved transfer");
                    }
                }
                client.transaction().forOperations(
                        client.transactionOp().setData().withVersion(guard.getVersion()).forPath(root, configuration),
                        client.transactionOp().create().forPath(root + "/" + intent.requestId(), data));
                return new Entry(intent, null, 0);
            } catch (KeeperException.BadVersionException | KeeperException.NodeExistsException retry) {
                continue;
            } catch (Exception failure) {
                throw new IllegalStateException("durable intent creation not confirmed", failure);
            }
        }
        throw new IllegalStateException("journal admission contention limit exceeded");
    }

    @Override
    public Entry record(Entry expected, Result result) {
        validate(expected.intent());
        Objects.requireNonNull(result);
        Result previous = expected.result();
        if (previous != null && (previous.committed() || previous.outcome() == PrefixTransferCoordinator.Outcome.ABORTED
                || previous.outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED
                || previous.outcome() == PrefixTransferCoordinator.Outcome.SUPERSEDED
                || (previous.outcome() == PrefixTransferCoordinator.Outcome.REJECTED && !previous.cleanupPending()))
                && (result.outcome() != previous.outcome() || result.operationId() != previous.operationId()
                || result.sourceLeaseId() != previous.sourceLeaseId()
                || result.committedVersion() != previous.committedVersion()
                || (!previous.cleanupPending() && result.cleanupPending()))) {
            throw new IllegalArgumentException("resolved transfer evidence cannot regress");
        }
        var replacement = new Entry(expected.intent(), result, Math.addExact(expected.revision(), 1));
        try {
            Entry actual = read(expected.intent().requestId());
            if (!actual.equals(expected)) {
                throw new IllegalStateException("journal entry changed before result recording");
            }
            byte[] data = encode(new Payload(1, expected.intent(), result));
            client.setData().withVersion(expected.revision()).forPath(root + "/" + expected.intent().requestId(), data);
            return replacement;
        } catch (Exception failure) {
            throw new IllegalStateException("durable result recording not confirmed", failure);
        }
    }

    @Override
    public List<Entry> entries() {
        try {
            List<String> children = client.getChildren().forPath(root);
            if (children.size() > capacity) {
                throw new IllegalStateException("journal exceeds its configured capacity");
            }
            List<Entry> result = new ArrayList<>();
            for (String child : children.stream().sorted().toList()) {
                result.add(read(child));
            }
            return List.copyOf(result);
        } catch (Exception failure) {
            throw new IllegalStateException("cannot load durable prefix transfers", failure);
        }
    }

    @Override
    public Retirement retire(Entry expected, ConfirmedCacheDirectory directory, long retentionMillis) {
        validate(expected.intent());
        Objects.requireNonNull(directory);
        if (retentionMillis < 0) {
            throw new IllegalArgumentException("negative transfer retention");
        }
        Result result = expected.result();
        if (result == null || result.cleanupPending() || result.outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN
                || clock.millis() < Math.addExact(expected.intent().deadlineUnixMs(), retentionMillis)) {
            return Retirement.WAITING;
        }
        Retirement decision = Retirement.TERMINAL;
        var view = directory.view();
        boolean replacedDestination = (result.committed()
                || result.outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED)
                && view.values().stream().anyMatch(snapshot -> {
                    var report = snapshot.report();
                    return namespace.equals(report.namespace()) && expected.intent().destination().workerId().equals(report.workerId())
                            && report.dpLane() == expected.intent().destination().dpLane() && report.deviceReady()
                            && report.rankReadinessGeneration() > 0
                            && !expected.intent().destination().epoch().equals(report.epoch());
                });
        if (!replacedDestination && (result.committed() || result.outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED)) {
            var destination = expected.intent().destination();
            var snapshots = view.values().stream().filter(snapshot -> {
                var report = snapshot.report();
                return namespace.equals(report.namespace()) && destination.workerId().equals(report.workerId())
                        && destination.epoch().equals(report.epoch()) && report.dpLane() == destination.dpLane()
                        && destination.layoutFingerprint().equals(report.layoutFingerprint())
                        && report.cpDegree() == destination.rankCount() && report.deviceReady()
                        && report.rankReadinessGeneration() > 0 && report.version() >= result.committedVersion();
            }).toList();
            if (snapshots.size() != 1) {
                return Retirement.WAITING;
            }
            if (expected.intent().action() == Action.RECLAIM) {
                if (snapshots.getFirst().keys().stream().anyMatch(expected.intent().keys()::contains)) {
                    if (snapshots.getFirst().report().version() == result.committedVersion()) {
                        return Retirement.WAITING;
                    }
                    decision = Retirement.TERMINAL;
                } else {
                    decision = Retirement.NO_LONGER_RETAINED;
                }
            } else {
                decision = snapshots.getFirst().keys().containsAll(expected.intent().keys())
                        ? Retirement.REPORTED_READY : Retirement.NO_LONGER_RETAINED;
            }
        }
        try {
            Entry actual = read(expected.intent().requestId());
            if (!actual.equals(expected)) {
                throw new IllegalStateException("journal entry changed before retirement");
            }
            Stat guard = new Stat();
            long floor = deadlineFloor(client.getData().storingStatIn(guard).forPath(root));
            byte[] configuration = ("prefix-transfer-journal-v1:" + capacity + "\n"
                    + Math.max(floor, expected.intent().deadlineUnixMs())).getBytes(StandardCharsets.UTF_8);
            client.transaction().forOperations(
                    client.transactionOp().setData().withVersion(guard.getVersion()).forPath(root, configuration),
                    client.transactionOp().delete().withVersion(expected.revision())
                            .forPath(root + "/" + expected.intent().requestId()));
            return decision;
        } catch (Exception failure) {
            throw new IllegalStateException("durable transfer retirement not confirmed", failure);
        }
    }

    private long deadlineFloor(byte[] configuration) {
        String[] fields = new String(configuration, StandardCharsets.UTF_8).split("\n", -1);
        if (fields.length > 2 || !fields[0].equals("prefix-transfer-journal-v1:" + capacity)) {
            throw new IllegalStateException("journal capacity/schema mismatch");
        }
        long floor = fields.length == 1 ? 0 : Long.parseLong(fields[1]);
        if (floor < 0) {
            throw new IllegalStateException("invalid journal retirement watermark");
        }
        return floor;
    }

    private Entry read(String requestId) throws Exception {
        if (!java.util.UUID.fromString(requestId).toString().equals(requestId)) {
            throw new IllegalStateException("unexpected journal child");
        }
        Stat stat = new Stat();
        byte[] data = client.getData().storingStatIn(stat).forPath(root + "/" + requestId);
        if (data.length > MAX_PAYLOAD_BYTES) {
            throw new IllegalStateException("journal payload exceeds limit");
        }
        Payload payload = mapper.readValue(data, Payload.class);
        validate(payload.intent());
        if (payload.schema() != 1 || !payload.intent().requestId().equals(requestId)) {
            throw new IllegalStateException("journal payload identity/schema mismatch");
        }
        return new Entry(payload.intent(), payload.result(), stat.getVersion());
    }

    private byte[] encode(Payload payload) {
        try {
            byte[] data = mapper.writeValueAsBytes(payload);
            if (data.length > MAX_PAYLOAD_BYTES) {
                throw new IllegalArgumentException("prefix transfer intent exceeds journal payload limit");
            }
            return data;
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("cannot encode durable transfer", failure);
        }
    }

    private void validate(Intent intent) {
        if (!namespace.equals(intent.namespace())) {
            throw new IllegalArgumentException("journal namespace mismatch");
        }
    }
}
