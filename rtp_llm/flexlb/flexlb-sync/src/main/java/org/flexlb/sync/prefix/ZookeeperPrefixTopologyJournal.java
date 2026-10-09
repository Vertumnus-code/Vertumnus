package org.flexlb.sync.prefix;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.curator.framework.CuratorFramework;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.common.PathUtils;
import org.apache.zookeeper.data.Stat;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.dao.cache.CacheReportV2;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Clock;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;

public final class ZookeeperPrefixTopologyJournal {
    public enum State { PLANNED, SUBMITTED, CONFIRMED, CANCELLED, ADMITTED }
    public record Entry(PrefixTopologyPlan plan, State state, List<String> reports, int revision,
                        List<PrefixWorkerRegistry.Registration> registrations, long creationOrder,
                        PrefixTopologyPreservation preservation) {
        public Entry(PrefixTopologyPlan plan, State state, List<String> reports, int revision) {
            this(plan, state, reports, revision, List.of(), 0, null);
        }
        public Entry(PrefixTopologyPlan plan, State state, List<String> reports, int revision,
                     List<PrefixWorkerRegistry.Registration> registrations, long creationOrder) {
            this(plan, state, reports, revision, registrations, creationOrder, null);
        }
        public Entry {
            Objects.requireNonNull(plan);
            Objects.requireNonNull(state);
            reports = List.copyOf(reports);
            registrations = registrations == null ? List.of() : List.copyOf(registrations);
            if (revision < 0 || (state != State.CONFIRMED && state != State.ADMITTED && !reports.isEmpty())
                    || (state != State.ADMITTED && !registrations.isEmpty())) {
                throw new IllegalArgumentException("invalid topology journal state");
            }
            if ((state == State.CONFIRMED || state == State.ADMITTED) && !plan.confirmed(decodeReports(reports))) {
                throw new IllegalArgumentException("topology outcome lacks all-rank confirmation");
            }
            if (state == State.ADMITTED) { PrefixTopologyAdmission.validate(plan, registrations, decodeReports(reports)); }
            if (preservation != null) {
                preservation.validate(plan);
                if (state == State.CONFIRMED || state == State.ADMITTED) {
                    preservation.validateResidentAfter(plan, decodeReports(reports));
                }
            }
        }
    }
    private record Payload(int schema, PrefixTopologyPlan plan, State state, List<String> reports,
                           List<PrefixWorkerRegistry.Registration> registrations, PrefixTopologyPreservation preservation) { }
    private final CuratorFramework client;
    private final ObjectMapper mapper = new ObjectMapper();
    private final String root;
    private final String namespace;
    private final int capacity;
    private final Clock clock;
    private final byte[] configuration;
    private final boolean requirePreservation;
    private record Decoded(byte[] wire, int jsonBytes, Entry entry) { }
    private static final int MAX_DECODED_ENTRIES = 128;
    private static final int MAX_DECODED_JSON_BYTES = 32 * 1024 * 1024;
    private final LinkedHashMap<String, Decoded> decoded = new LinkedHashMap<>(16, 0.75f, true);
    private int decodedJsonBytes;

    public ZookeeperPrefixTopologyJournal(CuratorFramework client, String path, String namespace, int capacity, Clock clock) {
        this(client, path, namespace, capacity, clock, false);
    }

    public ZookeeperPrefixTopologyJournal(CuratorFramework client, String path, String namespace, int capacity, Clock clock,
                                         boolean requirePreservation) {
        this.client = Objects.requireNonNull(client);
        this.clock = Objects.requireNonNull(clock);
        this.namespace = Objects.requireNonNull(namespace);
        this.capacity = capacity;
        this.requirePreservation = requirePreservation;
        PathUtils.validatePath(path);
        if (path.equals("/") || namespace.isBlank() || namespace.getBytes(StandardCharsets.UTF_8).length > 512
                || capacity < 1 || capacity > 4096) { throw new IllegalArgumentException("invalid topology journal configuration"); }
        configuration = ((requirePreservation ? "prefix-topology-writer-v1:" : "prefix-topology-journal-v1:") + capacity)
                .getBytes(StandardCharsets.UTF_8);
        try {
            root = path + "/" + HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(namespace.getBytes(StandardCharsets.UTF_8)));
            try { client.create().creatingParentsIfNeeded().forPath(root, configuration); }
            catch (KeeperException.NodeExistsException exists) { checkConfiguration(client.getData().forPath(root)); }
        } catch (Exception failure) { throw new IllegalStateException("cannot initialize topology journal", failure); }
    }

    private void checkConfiguration(byte[] data) {
        if (!java.util.Arrays.equals(configuration, data)) { throw new IllegalStateException("topology journal configuration changed"); }
    }

    public boolean requiresPreservation() { return requirePreservation; }

    public Entry create(PrefixTopologyPlan plan) {
        if (!plan.namespace().equals(namespace)) { throw new IllegalArgumentException("topology namespace mismatch"); }
        byte[] data = encode(new Payload(1, plan, State.PLANNED, List.of(), List.of(), null));
        for (int attempt = 0; attempt < 16; attempt++) {
            try {
                Stat guard = new Stat();
                checkConfiguration(client.getData().storingStatIn(guard).forPath(root));
                var children = client.getChildren().forPath(root);
                if (children.contains(plan.switchId())) {
                    Entry existing = read(plan.switchId());
                    if (!existing.plan().equals(plan)) { throw new IllegalStateException("topology plan ID collision"); }
                    return existing;
                }
                plan.payload(clock.millis());
                if (children.size() >= capacity) { throw new IllegalStateException("topology journal capacity exhausted"); }
                var endpoints = plan.participants().stream().map(PrefixTopologyPlan.Participant::controlEndpoint).toList();
                for (String child : children) {
                    var existing = read(child);
                    var previousEndpoints = existing.plan().participants().stream()
                            .map(PrefixTopologyPlan.Participant::controlEndpoint).collect(java.util.stream.Collectors.toSet());
                    var workers = PrefixTopologyAdmission.workers(plan);
                    var previousWorkers = PrefixTopologyAdmission.workers(existing.plan());
                    boolean overlaps = previousEndpoints.stream().anyMatch(endpoints::contains)
                            || previousWorkers.stream().anyMatch(workers::contains);
                    if (overlaps && (!previousEndpoints.equals(java.util.Set.copyOf(endpoints)) || !previousWorkers.equals(workers))) {
                        throw new IllegalStateException("topology plan changes a registered world boundary");
                    }
                    if ((existing.state() == State.PLANNED || existing.state() == State.SUBMITTED)
                            && overlaps) {
                        throw new IllegalStateException("world has an unresolved topology plan");
                    }
                }
                client.transaction().forOperations(
                        client.transactionOp().setData().withVersion(guard.getVersion()).forPath(root, configuration),
                        client.transactionOp().create().forPath(root + "/" + plan.switchId(), data));
                return read(plan.switchId());
            } catch (KeeperException.BadVersionException | KeeperException.NodeExistsException retry) {
                continue;
            } catch (Exception failure) { throw new IllegalStateException("topology intent persistence not confirmed", failure); }
        }
        throw new IllegalStateException("topology admission contention limit");
    }

    public Entry submitted(Entry expected) {
        if (expected.state() != State.PLANNED) { throw new IllegalStateException("topology plan already submitted"); }
        if (requirePreservation && (expected.preservation() == null || !expected.preservation().complete())) {
            throw new IllegalStateException("writer topology requires durable preservation proof before submission");
        }
        expected.plan().payload(clock.millis());
        return replace(expected, State.SUBMITTED, List.of());
    }

    public Entry cancelPlanned(Entry expected) {
        if (expected.state() != State.PLANNED) { throw new IllegalStateException("submitted topology cannot be cancelled as unissued"); }
        return replace(expected, State.CANCELLED, List.of());
    }

    public Entry confirmed(Entry expected, List<EngineRpcService.CacheStatusPB> reports) {
        if (expected.state() != State.SUBMITTED || !expected.plan().confirmed(reports)) {
            throw new IllegalStateException("topology completion is unproven");
        }
        return replace(expected, State.CONFIRMED, reports.stream()
                .map(ZookeeperPrefixTopologyJournal::encodeReport).toList());
    }

    private Entry replace(Entry expected, State state, List<String> reports) {
        return replace(expected, state, reports, List.of());
    }

    public Entry admitted(Entry expected, List<PrefixWorkerRegistry.Registration> registrations,
                          List<EngineRpcService.CacheStatusPB> reports) {
        if (expected.state() != State.CONFIRMED) { throw new IllegalStateException("only confirmed topology can be admitted"); }
        PrefixTopologyAdmission.validate(expected.plan(), registrations, reports);
        PrefixTopologyAdmission.validateContinuation(decodeReports(expected.reports()), reports);
        return replace(expected, State.ADMITTED, reports.stream()
                .map(ZookeeperPrefixTopologyJournal::encodeReport).toList(), registrations);
    }

    private Entry replace(Entry expected, State state, List<String> reports, List<PrefixWorkerRegistry.Registration> registrations) {
        return replace(expected, state, reports, registrations, expected.preservation());
    }

    public Entry planPreservation(Entry expected, List<PrefixTopologyPreservation.Required> required) {
        return planPreservation(expected, required, List.of());
    }

    public Entry planPreservation(Entry expected, List<PrefixTopologyPreservation.Required> required,
                                  List<org.flexlb.cache.prefix.PrefixReplicaPlanner.Proposal> copies) {
        if (expected.state() != State.PLANNED || expected.preservation() != null) {
            throw new IllegalStateException("preservation intent is immutable or topology already submitted");
        }
        return replace(expected, State.PLANNED, List.of(), List.of(), new PrefixTopologyPreservation(required, false, java.util.Map.of(), copies));
    }

    public Entry planResidentMerge(Entry expected, List<PrefixTopologyPreservation.ResidentMerge> merges) {
        if (expected.state() != State.PLANNED || expected.preservation() != null || !expected.plan().residentLeaderMerge()) {
            throw new IllegalStateException("resident merge intent is immutable or mode is absent");
        }
        return replace(expected, State.PLANNED, List.of(), List.of(),
                new PrefixTopologyPreservation(List.of(), false, java.util.Map.of(), List.of(), merges));
    }

    public Entry preserved(Entry expected, java.util.Map<String, org.flexlb.dao.cache.CacheReportV2> reports) {
        if (expected.state() != State.PLANNED || expected.preservation() == null || expected.preservation().complete()) {
            throw new IllegalStateException("preservation intent is absent or already completed");
        }
        var evidence = reports.entrySet().stream().collect(java.util.stream.Collectors.toMap(
                java.util.Map.Entry::getKey, entry -> withoutTelemetry(entry.getValue())));
        return replace(expected, State.PLANNED, List.of(), List.of(), new PrefixTopologyPreservation(expected.preservation().required(), true, evidence,
                expected.preservation().copies(), expected.preservation().residentMerges()));
    }

    // Durable evidence keeps every inventory key and identity/version field.
    // Historical execution/copy telemetry is unrelated to topology correctness
    // and can exceed the znode bound even for a single required prefix.
    private static CacheReportV2 withoutTelemetry(CacheReportV2 report) {
        return new CacheReportV2(report.namespace(), report.workerId(), report.dpLane(), report.epoch(),
                report.layoutFingerprint(), report.snapshotKind(), report.version(), report.baseVersion(),
                report.rankReadinessGeneration(), report.deviceReady(), report.cpDegree(), report.blockSize(),
                report.physicalFreeBytes(), report.deviceKeys(), List.of(), report.droppedReuseEvents(),
                report.physicalBlockBytes(), report.evictableBytes(), report.activeBytes(), report.residentBytes(),
                report.operationReservedBytes(), report.physicalMemoryStatsValid(), report.physicalCapacityBytes());
    }

    private static String encodeReport(EngineRpcService.CacheStatusPB report) {
        var evidence = report.toBuilder().clearCacheKeys();
        if (report.hasInventory()) {
            evidence.setInventory(report.getInventory().toBuilder().clearReuseEvents().clearCopyLifetime());
        }
        return Base64.getEncoder().encodeToString(evidence.build().toByteArray());
    }

    private Entry replace(Entry expected, State state, List<String> reports, List<PrefixWorkerRegistry.Registration> registrations,
                          PrefixTopologyPreservation preservation) {
        if (!expected.plan().namespace().equals(namespace)) { throw new IllegalArgumentException("topology namespace mismatch"); }
        if (preservation != null) { preservation.validate(expected.plan()); }
        if (preservation != null && (state == State.CONFIRMED || state == State.ADMITTED)) {
            preservation.validateResidentAfter(expected.plan(), decodeReports(reports));
        }
        try {
            if (!read(expected.plan().switchId()).equals(expected)) { throw new IllegalStateException("topology entry changed"); }
            byte[] data = encode(new Payload(1, expected.plan(), state, reports, registrations, preservation));
            if (state == State.ADMITTED) {
                Stat guard = new Stat();
                checkConfiguration(client.getData().storingStatIn(guard).forPath(root));
                for (var other : entries()) {
                    if (other.state() != State.CANCELLED && Long.compareUnsigned(other.creationOrder(), expected.creationOrder()) > 0
                            && PrefixTopologyAdmission.workers(other.plan()).stream()
                            .anyMatch(PrefixTopologyAdmission.workers(expected.plan())::contains)) {
                        throw new IllegalStateException("newer topology plan supersedes admission");
                    }
                }
                client.transaction().forOperations(
                        client.transactionOp().check().withVersion(guard.getVersion()).forPath(root),
                        client.transactionOp().setData().withVersion(expected.revision())
                                .forPath(root + "/" + expected.plan().switchId(), data));
            } else {
                client.setData().withVersion(expected.revision()).forPath(root + "/" + expected.plan().switchId(), data);
            }
            return new Entry(expected.plan(), state, reports, Math.addExact(expected.revision(), 1), registrations, expected.creationOrder(), preservation);
        } catch (Exception failure) { throw new IllegalStateException("topology journal update not confirmed", failure); }
    }

    public Entry read(String id) {
        if (!java.util.UUID.fromString(id).toString().equals(id)) { throw new IllegalArgumentException("invalid topology ID"); }
        try {
            Stat version = new Stat();
            byte[] data = client.getData().storingStatIn(version).forPath(root + "/" + id);
            // Always read ZooKeeper. Only immutable decoding/validation work is
            // reused, after both the node identity and its exact bytes match.
            synchronized (decoded) {
                var cached = decoded.get(id);
                if (cached != null && cached.entry().revision() == version.getVersion()
                        && cached.entry().creationOrder() == version.getCzxid()
                        && java.util.Arrays.equals(cached.wire(), data)) { return cached.entry(); }
            }
            byte[] json = PrefixTopologyJournalCodec.decode(data);
            Payload payload = mapper.readValue(json, Payload.class);
            if (payload.schema() != 1 || !payload.plan().namespace().equals(namespace) || !payload.plan().switchId().equals(id)) {
                throw new IllegalStateException("invalid topology journal payload");
            }
            var entry = new Entry(payload.plan(), payload.state(), payload.reports(), version.getVersion(), payload.registrations(),
                    version.getCzxid(), payload.preservation());
            if (requirePreservation && entry.state() != State.PLANNED && entry.state() != State.CANCELLED
                    && (entry.preservation() == null || !entry.preservation().complete())) {
                throw new IllegalStateException("submitted writer plan has no preservation proof");
            }
            synchronized (decoded) {
                var previous = decoded.remove(id);
                if (previous != null) { decodedJsonBytes -= previous.jsonBytes(); }
                while (!decoded.isEmpty() && (decoded.size() >= MAX_DECODED_ENTRIES
                        || decodedJsonBytes + json.length > MAX_DECODED_JSON_BYTES)) {
                    decodedJsonBytes -= decoded.pollFirstEntry().getValue().jsonBytes();
                }
                decoded.put(id, new Decoded(data, json.length, entry));
                decodedJsonBytes += json.length;
            }
            return entry;
        } catch (Exception failure) { throw new IllegalStateException("topology journal read failed", failure); }
    }

    public List<Entry> entries() {
        try {
            checkConfiguration(client.getData().forPath(root));
            var children = client.getChildren().forPath(root);
            if (children.size() > capacity) { throw new IllegalStateException("topology journal capacity exceeded"); }
            var entries = new ArrayList<Entry>();
            for (String child : children.stream().sorted().toList()) { entries.add(read(child)); }
            return List.copyOf(entries);
        } catch (Exception failure) { throw new IllegalStateException("cannot read topology plans", failure); }
    }

    private byte[] encode(Payload payload) {
        try {
            return PrefixTopologyJournalCodec.encode(mapper.writeValueAsBytes(payload));
        } catch (java.io.IOException failure) { throw new IllegalArgumentException("invalid topology plan", failure); }
    }

    public static List<EngineRpcService.CacheStatusPB> decodeReports(List<String> reports) {
        try {
            var decoded = new ArrayList<EngineRpcService.CacheStatusPB>();
            for (String report : reports) { decoded.add(EngineRpcService.CacheStatusPB.parseFrom(Base64.getDecoder().decode(report))); }
            return List.copyOf(decoded);
        } catch (java.io.IOException failure) { throw new IllegalArgumentException("invalid topology evidence", failure); }
    }
}
