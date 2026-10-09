package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.cache.prefix.PrefixReplicaPlanner.Proposal;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Result;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerSnapshot;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class PrefixCopyExecutor implements AutoCloseable {
    public record DegreePair(int source, int destination) {
        public DegreePair {
            if (source <= 0 || destination <= 0 || source == destination) {
                throw new IllegalArgumentException("cross-degree copy requires distinct positive degrees");
            }
        }
    }

    public record Limits(int maxCandidates, int maxOperations, long destinationBytes, long wireBytes,
                         long safetyBytes, long requestTimeoutMillis, long transferTimeoutMillis) {
        public Limits {
            if (maxCandidates <= 0 || maxCandidates > 4096 || maxOperations <= 0 || maxOperations > maxCandidates
                    || destinationBytes <= 0 || wireBytes <= 0 || safetyBytes < 0
                    || requestTimeoutMillis < 100 || requestTimeoutMillis > 300000
                    || transferTimeoutMillis < requestTimeoutMillis || transferTimeoutMillis > 300000) {
                throw new IllegalArgumentException("invalid prefix copy execution limits");
            }
        }
    }

    public record Attempt(Proposal proposal, Result result, String rejection, boolean executionStarted) { }
    public record Round(List<Attempt> attempts, int submitted, long destinationBytes, long leaderWireBytes,
                        long chargedWireBytes) {
        public Round { attempts = List.copyOf(attempts); }
    }
    private record Admission(WorkerSnapshot source, WorkerSnapshot destination, long physicalBytes, long leaderWireBytes,
                             long sourceBlockBytes, long blockBytes, int blockSize) { }

    private static final Set<String> REQUIRED_OPERATIONS = Set.of("controller_fencing_v1", "transfer_identity_v2",
            "missing_only_v1", "leader_rank_broadcast_v1", "prefix_transfer_endpoint_v1");
    private final EngineGrpcClient grpc;
    private final PrefixControllerAuthority authority;
    private final PrefixTransferJournal journal;
    private final PrefixTransferCoordinator coordinator;
    private final ConfirmedCacheDirectory directory;
    private final String namespace;
    private final String topologySwitchId;
    private final Limits limits;
    private final Set<DegreePair> crossDegreePairs;
    private final ThreadPoolExecutor executor;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicBoolean roundPending = new AtomicBoolean();

    public PrefixCopyExecutor(EngineGrpcClient grpc, PrefixControllerAuthority authority, PrefixTransferJournal journal,
                              ConfirmedCacheDirectory directory, String namespace, Limits limits) {
        this(grpc, authority, journal, directory, namespace, limits, Set.of());
    }

    public PrefixCopyExecutor(EngineGrpcClient grpc, PrefixControllerAuthority authority, PrefixTransferJournal journal,
                              ConfirmedCacheDirectory directory, String namespace, Limits limits,
                              Set<DegreePair> crossDegreePairs) {
        this(grpc, authority, journal, directory, namespace, limits, crossDegreePairs, null);
    }

    public PrefixCopyExecutor(EngineGrpcClient grpc, PrefixControllerAuthority authority, PrefixTransferJournal journal,
                              ConfirmedCacheDirectory directory, String namespace, Limits limits,
                              Set<DegreePair> crossDegreePairs, String topologySwitchId) {
        this.grpc = Objects.requireNonNull(grpc);
        this.authority = Objects.requireNonNull(authority);
        this.journal = Objects.requireNonNull(journal);
        this.directory = Objects.requireNonNull(directory);
        this.namespace = Objects.requireNonNull(namespace);
        this.topologySwitchId = topologySwitchId;
        new PrefixTransferJournal.Footprint(0, 0, topologySwitchId);
        this.limits = Objects.requireNonNull(limits);
        this.crossDegreePairs = Set.copyOf(crossDegreePairs);
        if (this.crossDegreePairs.size() > 64) { throw new IllegalArgumentException("too many cross-degree pairs"); }
        if (namespace.isBlank()) { throw new IllegalArgumentException("missing prefix copy namespace"); }
        coordinator = new PrefixTransferCoordinator(new GrpcPrefixTransferWorkerClient(grpc, limits.requestTimeoutMillis()),
                authority, journal);
        executor = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(1), task -> {
            Thread thread = new Thread(task, "prefix-copy-executor");
            thread.setDaemon(true);
            return thread;
        }, new ThreadPoolExecutor.AbortPolicy());
    }

    public CompletionStage<Round> execute(List<Proposal> proposals, Map<String, WorkerSnapshot> endpoints) {
        return execute(proposals, endpoints, Long.MAX_VALUE);
    }

    public CompletionStage<Round> execute(List<Proposal> proposals, Map<String, WorkerSnapshot> endpoints, long deadlineMillis) {
        Objects.requireNonNull(proposals);
        Objects.requireNonNull(endpoints);
        if (proposals.size() > limits.maxCandidates() || endpoints.size() > limits.maxCandidates() * 2) {
            throw new IllegalArgumentException("prefix copy candidate limit exceeded");
        }
        var frozen = List.copyOf(proposals);
        var registry = Map.copyOf(endpoints);
        if (closed.get() || !roundPending.compareAndSet(false, true)) {
            return CompletableFuture.failedFuture(new IllegalStateException("prefix copy executor unavailable or busy"));
        }
        var completion = new CompletableFuture<Round>();
        try {
            executor.execute(() -> {
                try {
                    var result = run(frozen, registry, deadlineMillis);
                    roundPending.set(false);
                    completion.complete(result);
                } catch (RuntimeException failure) {
                    roundPending.set(false);
                    completion.completeExceptionally(failure);
                }
            });
        } catch (RuntimeException failure) {
            roundPending.set(false);
            completion.completeExceptionally(failure);
        }
        return completion.minimalCompletionStage();
    }

    private Round run(List<Proposal> proposals, Map<String, WorkerSnapshot> endpoints, long deadlineMillis) {
        var controller = authority.snapshot(namespace);
        var attempts = new ArrayList<Attempt>();
        long physicalBytes = 0;
        long leaderWireBytes = 0;
        long chargedWireBytes = 0;
        int submitted = 0;
        for (Proposal proposal : proposals) {
            if (closed.get() || !authority.isCurrent(controller) || System.currentTimeMillis() >= deadlineMillis) { break; }
            boolean started = false;
            var source = endpoints.get(proposal.source());
            var destination = endpoints.get(proposal.destination());
            var workers = new HashSet<String>();
            if (source != null) { workers.add(source.workerId()); }
            if (destination != null) { workers.add(destination.workerId()); }
            try (var permit = directory.acquireOperations(workers)) {
                if (submitted >= limits.maxOperations()) {
                    throw new IllegalStateException("prefix copy operation budget exhausted");
                }
                if (journal.entries().stream().anyMatch(entry -> entry.result() == null || entry.result().cleanupPending()
                        || entry.result().outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN)) {
                    throw new IllegalStateException("unresolved durable transfer blocks new copy admission");
                }
                Admission admission = admit(proposal, endpoints);
                if (admission.physicalBytes() > limits.destinationBytes() - physicalBytes
                        || admission.physicalBytes() > limits.wireBytes() - chargedWireBytes) {
                    throw new IllegalStateException("prefix copy byte budget exhausted");
                }
                var sourceCapabilities = verifyWorker(admission.source(), proposal.path(), null, admission.sourceBlockBytes(), admission.blockSize());
                var destinationCapabilities = verifyWorker(admission.destination(), proposal.path(), proposal.missingBlocks(), admission.blockBytes(), admission.blockSize());
                if ("inter".equals(proposal.kind()) || admission.source().dpLane() != 0 || admission.destination().dpLane() != 0
                        || !admission.source().layoutFingerprint().equals(admission.destination().layoutFingerprint())
                        || sourceCapabilities.hasReplicatedLayout() || destinationCapabilities.hasReplicatedLayout()) {
                    long perRankBytes = PrefixLayoutCompatibility.requireCompatible(sourceCapabilities, destinationCapabilities, namespace);
                    if (Math.multiplyExact(perRankBytes, admission.destination().rankCount()) != admission.blockBytes()
                            || Math.multiplyExact(perRankBytes, admission.source().rankCount()) != admission.sourceBlockBytes()) {
                        throw new IllegalStateException("replicated layout disagrees with physical inventory footprint");
                    }
                }
                admit(proposal, endpoints);
                if (closed.get() || !authority.isCurrent(controller)) { break; }
                long timeout = Math.min(limits.transferTimeoutMillis(), deadlineMillis - System.currentTimeMillis());
                if (timeout < limits.requestTimeoutMillis()) { break; }
                submitted++;
                started = true;
                physicalBytes += admission.physicalBytes();
                leaderWireBytes += admission.leaderWireBytes();
                chargedWireBytes += admission.physicalBytes();
                Result result = coordinator.transfer(admission.source(), admission.destination(), namespace,
                        proposal.path(), timeout, controller,
                        new PrefixTransferJournal.Footprint(admission.physicalBytes(), admission.physicalBytes(), topologySwitchId));
                attempts.add(new Attempt(proposal, result, "", true));
            } catch (RuntimeException failure) {
                attempts.add(new Attempt(proposal, null, failure.getMessage() == null
                        ? failure.getClass().getSimpleName() : failure.getMessage(), started));
            }
        }
        return new Round(attempts, submitted, physicalBytes, leaderWireBytes, chargedWireBytes);
    }

    private Admission admit(Proposal proposal, Map<String, WorkerSnapshot> endpoints) {
        boolean inter = "inter".equals(proposal.kind());
        if ((!inter && !"intra".equals(proposal.kind())) || !proposal.reclaimKeys().isEmpty()
                || proposal.path().isEmpty() || proposal.path().size() > 8192
                || new HashSet<>(proposal.path()).size() != proposal.path().size()
                || proposal.source().equals(proposal.destination()) || proposal.degree() <= 0
                || !Double.isFinite(proposal.priority()) || proposal.priority() < 0
                || (inter ? proposal.replicas() != 0 : proposal.replicas() <= 0)) {
            throw new IllegalArgumentException("unsupported or invalid prefix copy proposal");
        }
        var source = Objects.requireNonNull(endpoints.get(proposal.source()), "source endpoint not registered");
        var destination = Objects.requireNonNull(endpoints.get(proposal.destination()), "destination endpoint not registered");
        if (source.lane().equals(destination.lane())
                || (source.workerId().equals(destination.workerId()) && source.epoch().equals(destination.epoch()))
                || source.controlEndpoint().equals(destination.controlEndpoint())
                || source.controlEndpoint().equals(destination.transferEndpoint())
                || source.transferEndpoint().equals(destination.controlEndpoint())
                || source.transferEndpoint().equals(destination.transferEndpoint())
                || source.rankCount() <= 0 || destination.rankCount() != proposal.degree()) {
            throw new IllegalArgumentException("copy requires independent workers and the requested destination degree");
        }
        if (inter) {
            if (source.rankCount() == destination.rankCount()
                    || !crossDegreePairs.contains(new DegreePair(source.rankCount(), destination.rankCount()))) {
                throw new IllegalArgumentException("cross-degree pair has not been explicitly enabled");
            }
        } else if (source.rankCount() != proposal.degree()) {
            throw new IllegalArgumentException("intra copy requires the same degree");
        }
        var view = directory.view();
        var sourceReport = requireSnapshot(view.get(proposal.source()), source);
        var destinationReport = requireSnapshot(view.get(proposal.destination()), destination);
        if (!sourceReport.available(proposal.path())
                || proposal.path().stream().anyMatch(destinationReport.unavailableKeys()::contains)
                || sourceReport.report().blockSize() != destinationReport.report().blockSize()) {
            throw new IllegalStateException("source path or physical layout is not confirmed");
        }
        if (inter && view.values().stream().anyMatch(snapshot -> snapshot.report().cpDegree() == proposal.degree()
                && snapshot.keys().containsAll(proposal.path()))) {
            throw new IllegalStateException("inter copy requires no confirmed replica at the destination degree");
        }
        List<Long> missing = proposal.path().stream().filter(key -> !destinationReport.keys().contains(key)).toList();
        if (missing.isEmpty() || !missing.equals(proposal.missingBlocks())) {
            throw new IllegalStateException("copy plan is stale or already fully retained");
        }
        long blockBytes = destinationReport.report().physicalBlockBytes();
        long sourceBlockBytes = sourceReport.report().physicalBlockBytes();
        if (blockBytes <= 0 || blockBytes % destination.rankCount() != 0
                || sourceBlockBytes <= 0 || sourceBlockBytes % source.rankCount() != 0
                || sourceBlockBytes / source.rankCount() != blockBytes / destination.rankCount()) {
            throw new IllegalStateException("replicated rank physical footprint is invalid");
        }
        long bytes = Math.multiplyExact(blockBytes, missing.size());
        if (bytes != proposal.destinationPhysicalBytes() || proposal.estimatedWireBytes() != bytes
                || bytes > Math.max(0, destinationReport.report().physicalFreeBytes() - limits.safetyBytes())) {
            throw new IllegalStateException("copy footprint or safe physical capacity changed"
                    + " [diagnostic worker=" + destination.workerId() + ", expectedVersion=" + destination.version()
                    + ", actualVersion=" + destinationReport.report().version() + ", requiredBytes=" + bytes
                    + ", plannedBytes=" + proposal.destinationPhysicalBytes() + ", plannedWireBytes=" + proposal.estimatedWireBytes()
                    + ", actualFreeBytes=" + destinationReport.report().physicalFreeBytes()
                    + ", safetyBytes=" + limits.safetyBytes() + ", missingBlocks=" + missing.size() + "]");
        }
        return new Admission(source, destination, bytes, bytes / destination.rankCount(), sourceBlockBytes, blockBytes,
                destinationReport.report().blockSize());
    }

    private ConfirmedCacheDirectory.Snapshot requireSnapshot(ConfirmedCacheDirectory.Snapshot snapshot, WorkerSnapshot worker) {
        if (snapshot == null) { throw new IllegalStateException("confirmed Worker report missing or expired"); }
        var report = snapshot.report();
        if (!namespace.equals(report.namespace()) || !worker.workerId().equals(report.workerId())
                || !worker.epoch().equals(report.epoch()) || worker.version() != report.version()
                || !worker.layoutFingerprint().equals(report.layoutFingerprint()) || report.dpLane() != worker.dpLane()
                || !report.deviceReady() || report.rankReadinessGeneration() < 0
                || (report.rankReadinessGeneration() == 0 && !snapshot.keys().isEmpty())
                || report.cpDegree() != worker.rankCount()) {
            throw new IllegalStateException("frozen Worker identity no longer matches confirmed report"
                    + " [diagnostic worker=" + worker.workerId() + ", actualWorker=" + report.workerId()
                    + ", expectedVersion=" + worker.version() + ", actualVersion=" + report.version()
                    + ", namespaceMatch=" + namespace.equals(report.namespace())
                    + ", epochMatch=" + worker.epoch().equals(report.epoch())
                    + ", layoutMatch=" + worker.layoutFingerprint().equals(report.layoutFingerprint())
                    + ", expectedLane=" + worker.dpLane() + ", actualLane=" + report.dpLane()
                    + ", expectedDegree=" + worker.rankCount() + ", actualDegree=" + report.cpDegree()
                    + ", ready=" + report.deviceReady() + ", generation=" + report.rankReadinessGeneration()
                    + ", keyCount=" + snapshot.keys().size() + "]");
        }
        return snapshot;
    }

    private EngineRpcService.PrefixCacheCapabilitiesResponsePB verifyWorker(
            WorkerSnapshot worker, List<Long> path, List<Long> missing, long blockBytes, int blockSize) {
        var endpoint = worker.controlEndpoint();
        var capabilities = grpc.getPrefixCacheCapabilities(endpoint.ip(), endpoint.port(),
                EngineRpcService.PrefixCacheCapabilitiesRequestPB.getDefaultInstance(), limits.requestTimeoutMillis());
        if (!namespace.equals(capabilities.getNamespace()) || !worker.layoutFingerprint().equals(capabilities.getLayoutFingerprint())
                || !capabilities.getDeviceReady() || capabilities.getDpLane() != worker.dpLane()
                || capabilities.getCpDegree() != worker.rankCount() || capabilities.getTpSize() != worker.rankCount()
                || capabilities.getBlockSize() != blockSize
                || !capabilities.getSupportedOperationsList().containsAll(REQUIRED_OPERATIONS)) {
            throw new IllegalStateException("Worker lacks the verified prefix transfer capability");
        }
        var response = grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion(-1).setNeedCacheKeys(false).build(), limits.requestTimeoutMillis());
        var report = response.getInventory();
        if (!response.hasInventory() || !namespace.equals(report.getNamespace()) || !worker.workerId().equals(report.getWorkerId())
                || !worker.epoch().equals(report.getEpoch()) || worker.version() != report.getVersion()
                || !worker.layoutFingerprint().equals(report.getLayoutFingerprint()) || !report.getSnapshotKind().equals("FULL")
                || report.getDpLane() != worker.dpLane() || report.getCpDegree() != worker.rankCount() || !report.getDeviceReady()
                || report.getRankReadinessGeneration() < 0
                || (report.getRankReadinessGeneration() == 0 && report.getDeviceKeysCount() != 0) || report.getBlockSize() != blockSize
                || report.getPhysicalBlockBytes() != blockBytes || report.getPhysicalFreeBytes() < 0) {
            throw new IllegalStateException("control endpoint does not confirm frozen Worker identity"
                    + " [diagnostic worker=" + worker.workerId() + ", actualWorker=" + report.getWorkerId()
                    + ", hasInventory=" + response.hasInventory()
                    + ", expectedVersion=" + worker.version() + ", actualVersion=" + report.getVersion()
                    + ", namespaceMatch=" + namespace.equals(report.getNamespace())
                    + ", epochMatch=" + worker.epoch().equals(report.getEpoch())
                    + ", layoutMatch=" + worker.layoutFingerprint().equals(report.getLayoutFingerprint())
                    + ", snapshotKind=" + report.getSnapshotKind()
                    + ", expectedLane=" + worker.dpLane() + ", actualLane=" + report.getDpLane()
                    + ", expectedDegree=" + worker.rankCount() + ", actualDegree=" + report.getCpDegree()
                    + ", ready=" + report.getDeviceReady() + ", generation=" + report.getRankReadinessGeneration()
                    + ", keyCount=" + report.getDeviceKeysCount()
                    + ", expectedBlockSize=" + blockSize + ", actualBlockSize=" + report.getBlockSize()
                    + ", expectedBlockBytes=" + blockBytes + ", actualBlockBytes=" + report.getPhysicalBlockBytes()
                    + ", actualFreeBytes=" + report.getPhysicalFreeBytes() + "]");
        }
        var keys = new HashSet<>(report.getDeviceKeysList());
        if (missing == null) {
            if (!keys.containsAll(path)) { throw new IllegalStateException("source no longer retains full prefix"); }
        } else {
            var actualMissing = path.stream().filter(key -> !keys.contains(key)).toList();
            long physicalBytes = Math.multiplyExact(report.getPhysicalBlockBytes(), actualMissing.size());
            if (!actualMissing.equals(missing) || report.getPhysicalBlockBytes() <= 0
                    || physicalBytes > Math.max(0, report.getPhysicalFreeBytes() - limits.safetyBytes())) {
                throw new IllegalStateException("destination missing plan or safe capacity changed before prepare"
                        + " [diagnostic worker=" + worker.workerId() + ", expectedVersion=" + worker.version()
                        + ", actualVersion=" + report.getVersion() + ", missingMatch=" + actualMissing.equals(missing)
                        + ", plannedMissingBlocks=" + missing.size() + ", actualMissingBlocks=" + actualMissing.size()
                        + ", requiredBytes=" + physicalBytes + ", actualFreeBytes=" + report.getPhysicalFreeBytes()
                        + ", safetyBytes=" + limits.safetyBytes() + "]");
            }
        }
        return capabilities;
    }

    @Override
    public void close() {
        closed.set(true);
        executor.shutdown();
    }
}
