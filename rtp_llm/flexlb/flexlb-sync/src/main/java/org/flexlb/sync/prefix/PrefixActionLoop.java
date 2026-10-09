package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.cache.prefix.PrefixObserver;
import org.flexlb.cache.prefix.PrefixReplicaPlanner.Proposal;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Outcome;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Result;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerSnapshot;

import java.time.Clock;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
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
import java.util.concurrent.atomic.AtomicLong;

public final class PrefixActionLoop implements AutoCloseable {
    public record Limits(int maxCandidates, int perList, int maxTrackedPaths, long scanTtlMillis,
                         long cooldownMillis, long initialBackoffMillis, long maxBackoffMillis,
                         long windowMillis, long windowWireBytes, long reclaimBytes, double maxForegroundTasks,
                         long rpcTimeoutMillis, long transferTimeoutMillis, long roundTimeoutMillis) {
        public Limits {
            if (maxCandidates < 1 || maxCandidates > 4096 || perList < 1 || perList > maxCandidates / 3
                    || maxTrackedPaths < maxCandidates || maxTrackedPaths > 100000 || scanTtlMillis < 1
                    || cooldownMillis < 1 || cooldownMillis > 3600000 || initialBackoffMillis < 1
                    || maxBackoffMillis < initialBackoffMillis || maxBackoffMillis > 3600000
                    || windowMillis < 1 || windowMillis > 3600000 || windowWireBytes < 0 || reclaimBytes < 0
                    || !Double.isFinite(maxForegroundTasks) || maxForegroundTasks < 0
                    || rpcTimeoutMillis < 100 || transferTimeoutMillis < rpcTimeoutMillis || transferTimeoutMillis > 300000
                    || roundTimeoutMillis < transferTimeoutMillis || roundTimeoutMillis > 3600000) {
                throw new IllegalArgumentException("invalid adaptive prefix action limits");
            }
        }
    }
    public record Decision(Proposal proposal, Result result, String reason, boolean dispatched) { }
    public record Round(long sequence, List<Decision> decisions, long destinationBytes, long wireBytes, long reclaimedBytes) {
        public Round { decisions = List.copyOf(decisions); }
    }
    private record PathDegree(List<Long> path, int degree) { }
    private record Retry(long nextMillis, long delayMillis) { }
    private final PrefixObserver observer;
    private final PrefixWorkerRegistry registry;
    private final EngineGrpcClient grpc;
    private final PrefixControllerAuthority authority;
    private final PrefixTransferJournal journal;
    private final PrefixCopyExecutor copies;
    private final PrefixReclaimCoordinator reclaims;
    private final Limits limits;
    private final Clock clock;
    private final ThreadPoolExecutor executor;
    private final Map<PathDegree, Retry> retries = new HashMap<>();
    private final Set<String> excluded = new HashSet<>();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicBoolean busy = new AtomicBoolean();
    private final AtomicLong skipped = new AtomicLong();
    private final AtomicLong failures = new AtomicLong();
    private long lastSequence;
    private long rounds;
    private long submitted;
    private long committed;
    private long reclaimed;
    private long totalWireBytes;
    private long totalDestinationBytes;
    private long totalReclaimedBytes;
    private volatile Round lastRound = new Round(0, List.of(), 0, 0, 0);

    public PrefixActionLoop(PrefixObserver observer, PrefixWorkerRegistry registry, EngineGrpcClient grpc,
                            PrefixControllerAuthority authority, PrefixTransferJournal journal, PrefixCopyExecutor copies,
                            PrefixReclaimCoordinator reclaims, Limits limits, Clock clock) {
        this.observer = Objects.requireNonNull(observer);
        this.registry = Objects.requireNonNull(registry);
        this.grpc = Objects.requireNonNull(grpc);
        this.authority = Objects.requireNonNull(authority);
        this.journal = Objects.requireNonNull(journal);
        this.copies = Objects.requireNonNull(copies);
        this.reclaims = Objects.requireNonNull(reclaims);
        this.limits = Objects.requireNonNull(limits);
        this.clock = Objects.requireNonNull(clock);
        if (!observer.settings().adaptive()) { throw new IllegalArgumentException("physical prefix actions require adaptive policy"); }
        executor = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(1), task -> {
            var thread = new Thread(task, "prefix-adaptive-actions");
            thread.setDaemon(true);
            return thread;
        }, new ThreadPoolExecutor.AbortPolicy());
    }

    public CompletionStage<Round> submit(PrefixObserver.Scan scan) {
        Objects.requireNonNull(scan);
        if (closed.get() || !busy.compareAndSet(false, true)) {
            skipped.incrementAndGet();
            return CompletableFuture.failedFuture(new IllegalStateException("adaptive round already pending or closed"));
        }
        var completion = new CompletableFuture<Round>();
        try {
            executor.execute(() -> {
                try {
                    lastRound = run(scan);
                    publish();
                    busy.set(false);
                    completion.complete(lastRound);
                } catch (RuntimeException failure) {
                    failures.incrementAndGet();
                    publish();
                    busy.set(false);
                    completion.completeExceptionally(failure);
                }
            });
        } catch (RuntimeException rejected) {
            busy.set(false);
            skipped.incrementAndGet();
            completion.completeExceptionally(rejected);
        }
        return completion.minimalCompletionStage();
    }

    public Round lastRound() { return lastRound; }

    private Round run(PrefixObserver.Scan scan) {
        long now = clock.millis();
        if (!scan.namespace().equals(observer.settings().namespace()) || scan.sequence() <= lastSequence
                || scan.createdMillis() > now || now - scan.createdMillis() > limits.scanTtlMillis()
                || scan.proposals().size() > limits.maxCandidates()) {
            throw new IllegalStateException("adaptive scan is stale, oversized or has wrong namespace");
        }
        rounds++;
        var entries = journal.entries();
        reconcileExclusions(entries);
        var controller = authority.snapshot(scan.namespace());
        if (!authority.isCurrent(controller) || closed.get()) { throw new IllegalStateException("adaptive controller is not current"); }
        var endpoints = registry.freeze(observer.directory().view());
        var decisions = new ArrayList<Decision>();
        var copyPlans = new ArrayList<Proposal>();
        var reclaimPlans = new ArrayList<Proposal>();
        var counts = new HashMap<String, Integer>();
        var selectedNodes = new HashSet<PathDegree>();
        long reservedWire = recentWire(entries, scan.inventory(), now);
        long plannedWire = 0;
        long plannedDestination = 0;
        long plannedReclaim = 0;
        retries.entrySet().removeIf(entry -> now > Math.addExact(entry.getValue().nextMillis(), limits.maxBackoffMillis()));
        for (var proposal : scan.proposals()) {
            try {
                if (uncertain(entries)) { throw new IllegalStateException("durable outcome unresolved"); }
                if (!Set.of("inter", "intra", "reclaim").contains(proposal.kind()) || proposal.path().isEmpty()
                        || proposal.path().size() > 8192 || new HashSet<>(proposal.path()).size() != proposal.path().size()) {
                    throw new IllegalArgumentException("invalid adaptive proposal");
                }
                if (!observer.settings().enabledActions().contains(proposal.kind())) {
                    throw new IllegalStateException("prefix action kind disabled");
                }
                if (counts.getOrDefault(proposal.kind(), 0) >= limits.perList()) { throw new IllegalStateException("per-list action budget"); }
                var key = new PathDegree(proposal.path(), proposal.degree());
                if (selectedNodes.contains(key)) { throw new IllegalStateException("node/degree already selected this scan"); }
                if ((!retries.containsKey(key) && retries.size() + copyPlans.size() + reclaimPlans.size() >= limits.maxTrackedPaths())
                        || (retries.containsKey(key) && now < retries.get(key).nextMillis()) || cooling(proposal, entries, now)) {
                    throw new IllegalStateException("path cooldown, retry backoff or tracking budget");
                }
                fresh(scan, proposal.destination(), endpoints.get(proposal.destination()), proposal.path(), false);
                if ("reclaim".equals(proposal.kind())) {
                    var destination = endpoints.get(proposal.destination());
                    validateReclaim(proposal, destination);
                    long blockBytes = scan.inventory().get(proposal.destination()).report().physicalBlockBytes();
                    if (blockBytes <= 0) { throw new IllegalStateException("unknown reclaim block footprint"); }
                    int blockCount = (int) Math.min(proposal.reclaimKeys().size(), (limits.reclaimBytes() - plannedReclaim) / blockBytes);
                    if (blockCount == 0) { throw new IllegalStateException("reclaim byte budget below one complete block"); }
                    var keys = proposal.reclaimKeys().subList(proposal.reclaimKeys().size() - blockCount, proposal.reclaimKeys().size());
                    var bounded = new Proposal(proposal.kind(), proposal.path(), proposal.degree(), proposal.source(), proposal.destination(),
                            proposal.missingBlocks(), proposal.destinationPhysicalBytes(), proposal.estimatedWireBytes(),
                            proposal.priority(), proposal.replicas(), keys);
                    long bytes = Math.multiplyExact(blockCount, blockBytes);
                    if (copyPlans.stream().anyMatch(copy -> overlaps(copy.path(), keys)
                            && (copy.source().equals(proposal.destination()) || copy.destination().equals(proposal.destination())))) {
                        throw new IllegalStateException("reclaim byte budget or copy conflict");
                    }
                    plannedReclaim += bytes;
                    reclaimPlans.add(bounded);
                } else {
                    fresh(scan, proposal.source(), endpoints.get(proposal.source()), proposal.path(), true);
                    if (proposal.destinationPhysicalBytes() < 0 || proposal.estimatedWireBytes() < 0
                            || proposal.destinationPhysicalBytes() > observer.settings().destinationBudgetBytes() - plannedDestination
                            || proposal.estimatedWireBytes() > observer.settings().wireBudgetBytes() - plannedWire
                            || proposal.estimatedWireBytes() > limits.windowWireBytes() - reservedWire - plannedWire) {
                        throw new IllegalStateException("copy round or rate-window byte budget");
                    }
                    plannedDestination += proposal.destinationPhysicalBytes();
                    plannedWire += proposal.estimatedWireBytes();
                    copyPlans.add(proposal);
                }
                counts.merge(proposal.kind(), 1, Integer::sum);
                selectedNodes.add(key);
            } catch (RuntimeException rejection) {
                decisions.add(new Decision(proposal, null, reason(rejection), false));
            }
        }
        long deadline = Math.addExact(System.currentTimeMillis(), limits.roundTimeoutMillis());
        long destinationBytes = 0;
        long wireBytes = 0;
        long reclaimedBytes = 0;
        if (!copyPlans.isEmpty() && !closed.get() && authority.isCurrent(controller)) {
            var usedEndpoints = new HashMap<String, WorkerSnapshot>();
            copyPlans.forEach(proposal -> {
                usedEndpoints.put(proposal.source(), endpoints.get(proposal.source()));
                usedEndpoints.put(proposal.destination(), endpoints.get(proposal.destination()));
            });
            var result = copies.execute(copyPlans, usedEndpoints, deadline).toCompletableFuture().join();
            destinationBytes = result.destinationBytes();
            wireBytes = result.chargedWireBytes();
            submitted += result.submitted();
            for (var attempt : result.attempts()) {
                decisions.add(new Decision(attempt.proposal(), attempt.result(), attempt.rejection(), attempt.executionStarted()));
                boolean success = attempt.result() != null && attempt.result().committed();
                if (success) { committed++; }
                remember(attempt.proposal(), success);
            }
        }
        for (var proposal : reclaimPlans) {
            if (closed.get() || !authority.isCurrent(controller) || System.currentTimeMillis() >= deadline) { break; }
            try {
                if (uncertain(journal.entries())) { throw new IllegalStateException("copy outcome unresolved before reclaim"); }
                var destination = endpoints.get(proposal.destination());
                fresh(scan, proposal.destination(), destination, proposal.path(), true);
                verifyReclaimCapability(destination, scan.inventory().get(proposal.destination()));
                fresh(scan, proposal.destination(), destination, proposal.path(), true);
                long timeout = Math.min(limits.transferTimeoutMillis(), deadline - System.currentTimeMillis());
                if (timeout < limits.rpcTimeoutMillis() || closed.get() || !authority.isCurrent(controller)) { break; }
                var attempt = reclaims.execute(destination, scan.namespace(), proposal.reclaimKeys(), timeout, controller);
                if (attempt.dispatched()) { submitted++; }
                boolean success = attempt.result().outcome() == Outcome.RECLAIMED;
                if (success) {
                    reclaimed++;
                    reclaimedBytes += Math.multiplyExact(proposal.reclaimKeys().size(), scan.inventory().get(proposal.destination()).report().physicalBlockBytes());
                }
                remember(proposal, success);
                decisions.add(new Decision(proposal, attempt.result(), "", attempt.dispatched()));
            } catch (RuntimeException failure) {
                remember(proposal, false);
                decisions.add(new Decision(proposal, null, reason(failure), false));
            }
        }
        totalDestinationBytes += destinationBytes;
        totalWireBytes += wireBytes;
        totalReclaimedBytes += reclaimedBytes;
        lastSequence = scan.sequence();
        return new Round(scan.sequence(), decisions, destinationBytes, wireBytes, reclaimedBytes);
    }

    private void fresh(PrefixObserver.Scan scan, String route, WorkerSnapshot worker, List<Long> path, boolean retained) {
        Objects.requireNonNull(worker, "Worker endpoint not registered");
        var snapshot = observer.directory().view().get(route);
        var planned = scan.inventory().get(route);
        if (planned == null || snapshot == null
                || !planned.report().namespace().equals(snapshot.report().namespace())
                || !planned.report().workerId().equals(snapshot.report().workerId())
                || !planned.report().epoch().equals(snapshot.report().epoch())
                || planned.report().dpLane() != snapshot.report().dpLane()
                || !planned.report().layoutFingerprint().equals(snapshot.report().layoutFingerprint())
                || planned.report().cpDegree() != snapshot.report().cpDegree()
                || planned.report().rankReadinessGeneration() > snapshot.report().rankReadinessGeneration()
                || planned.report().blockSize() != snapshot.report().blockSize()
                || planned.report().physicalBlockBytes() != snapshot.report().physicalBlockBytes()
                || !planned.report().deviceReady() || planned.report().version() > snapshot.report().version()) {
            throw new IllegalStateException("adaptive scan Worker identity, generation or physical layout changed");
        }
        if (!snapshot.report().namespace().equals(scan.namespace()) || snapshot.report().dpLane() != worker.dpLane()
                || !snapshot.report().workerId().equals(worker.workerId()) || !snapshot.report().epoch().equals(worker.epoch())
                || !snapshot.report().layoutFingerprint().equals(worker.layoutFingerprint())
                || !snapshot.report().deviceReady() || snapshot.report().cpDegree() != worker.rankCount()) {
            throw new IllegalStateException("adaptive Worker identity or eligibility changed after admission");
        }
        if (snapshot.report().version() != worker.version()) {
            throw new IllegalStateException("adaptive inventory version changed after admission: expected "
                    + worker.version() + ", observed " + snapshot.report().version());
        }
        if (path.stream().anyMatch(snapshot.unavailableKeys()::contains) || (retained && !snapshot.available(path))) {
            throw new IllegalStateException("adaptive retained path is missing or unavailable");
        }
        Double workload = observer.workloadSnapshot().get(route);
        if (workload == null || !Double.isFinite(workload) || workload < 0 || workload > limits.maxForegroundTasks()) {
            throw new IllegalStateException("foreground workload unknown or exceeds interference budget: route="
                    + route + ", observed=" + workload + ", limit=" + limits.maxForegroundTasks());
        }
    }

    private static void validateReclaim(Proposal proposal, WorkerSnapshot destination) {
        var keys = proposal.reclaimKeys();
        if (!proposal.source().isEmpty() || !proposal.missingBlocks().isEmpty() || proposal.destinationPhysicalBytes() != 0
                || proposal.estimatedWireBytes() != 0 || proposal.replicas() <= 0 || !Double.isFinite(proposal.priority())
                || proposal.priority() < 0 || destination.rankCount() != proposal.degree() || keys.isEmpty()
                || keys.size() > proposal.path().size()
                || !proposal.path().subList(proposal.path().size() - keys.size(), proposal.path().size()).equals(keys)) {
            throw new IllegalArgumentException("reclaim must remove a bounded cold suffix of a complete prefix");
        }
    }

    private void verifyReclaimCapability(WorkerSnapshot worker, ConfirmedCacheDirectory.Snapshot snapshot) {
        var endpoint = worker.controlEndpoint();
        var capabilities = grpc.getPrefixCacheCapabilities(endpoint.ip(), endpoint.port(),
                EngineRpcService.PrefixCacheCapabilitiesRequestPB.newBuilder().build(), limits.rpcTimeoutMillis());
        long perRank = PrefixLayoutCompatibility.requireCompatible(capabilities, capabilities, observer.settings().namespace());
        if (!capabilities.getDeviceReady() || capabilities.getDpLane() != worker.dpLane() || capabilities.getCpDegree() != worker.rankCount()
                || capabilities.getTpSize() != worker.rankCount() || !capabilities.getLayoutFingerprint().equals(worker.layoutFingerprint())
                || !capabilities.getSupportedOperationsList().containsAll(Set.of("controller_fencing_v1", "reclaim"))
                || Math.multiplyExact(perRank, worker.rankCount()) != snapshot.report().physicalBlockBytes()) {
            throw new IllegalStateException("reclaim full-KV capability or physical footprint mismatch");
        }
    }

    private long recentWire(List<PrefixTransferJournal.Entry> entries, Map<String, ConfirmedCacheDirectory.Snapshot> inventory, long now) {
        long bytes = 0;
        for (var entry : entries) {
            if (entry.intent().action() != PrefixTransferJournal.Action.COPY
                    || now > Math.addExact(entry.intent().deadlineUnixMs(), limits.windowMillis())) { continue; }
            if (entry.intent().footprint() != null) {
                bytes = Math.addExact(bytes, entry.intent().footprint().wireBytes());
                if (bytes >= limits.windowWireBytes()) { return limits.windowWireBytes(); }
                continue;
            }
            var destination = entry.intent().destination();
            var snapshot = inventory.values().stream().filter(item -> item.report().workerId().equals(destination.workerId())
                    && item.report().dpLane() == destination.dpLane()
                    && item.report().epoch().equals(destination.epoch()) && item.report().layoutFingerprint().equals(destination.layoutFingerprint()))
                    .findFirst().orElse(null);
            if (snapshot == null || snapshot.report().physicalBlockBytes() <= 0) { return limits.windowWireBytes(); }
            bytes = Math.addExact(bytes, Math.multiplyExact(entry.intent().keys().size(), snapshot.report().physicalBlockBytes()));
            if (bytes >= limits.windowWireBytes()) { return limits.windowWireBytes(); }
        }
        return bytes;
    }

    private boolean cooling(Proposal proposal, List<PrefixTransferJournal.Entry> entries, long now) {
        return entries.stream().filter(entry -> entry.result() == null || entry.result().cleanupPending()
                || (entry.result().outcome() != Outcome.ABORTED && entry.result().outcome() != Outcome.REJECTED))
                .anyMatch(entry -> entry.intent().destination().rankCount() == proposal.degree()
                && now < Math.addExact(entry.intent().deadlineUnixMs(), limits.cooldownMillis())
                && proposal.path().getLast().equals(entry.intent().keys().getLast()));
    }

    private static boolean uncertain(List<PrefixTransferJournal.Entry> entries) {
        return entries.stream().anyMatch(entry -> entry.result() == null || entry.result().cleanupPending() || entry.result().outcome() == Outcome.UNKNOWN);
    }

    private void reconcileExclusions(List<PrefixTransferJournal.Entry> entries) {
        var current = new HashSet<String>();
        for (var entry : entries) {
            if (entry.intent().action() != PrefixTransferJournal.Action.RECLAIM) { continue; }
            String id = entry.intent().requestId();
            if (entry.result() == null || entry.result().outcome() == Outcome.UNKNOWN || entry.result().outcome() == Outcome.RECLAIMED) {
                var destination = entry.intent().destination();
                observer.directory().exclude(id, destination.workerId(), destination.dpLane(), destination.epoch(), destination.version(), entry.intent().keys());
                current.add(id);
            } else { observer.directory().include(id); }
        }
        excluded.stream().filter(id -> !current.contains(id)).forEach(observer.directory()::include);
        excluded.clear();
        excluded.addAll(current);
    }

    private void remember(Proposal proposal, boolean success) {
        var key = new PathDegree(proposal.path(), proposal.degree());
        var previous = retries.get(key);
        long delay = success ? limits.cooldownMillis() : previous == null || previous.delayMillis() == 0 ? limits.initialBackoffMillis()
                : Math.min(limits.maxBackoffMillis(), previous.delayMillis() * 2);
        retries.put(key, new Retry(Math.addExact(clock.millis(), delay), success ? 0 : delay));
    }

    private static boolean overlaps(List<Long> left, List<Long> right) { return left.stream().anyMatch(right::contains); }
    private static String reason(RuntimeException failure) { return failure.getMessage() == null ? failure.getClass().getSimpleName() : failure.getMessage(); }

    private void publish() {
        if (closed.get()) { return; }
        var status = new LinkedHashMap<String, Object>();
        status.put("limits", limits);
        status.put("rounds", rounds);
        status.put("submitted", submitted);
        status.put("committed", committed);
        status.put("reclaimed", reclaimed);
        status.put("wire_bytes", totalWireBytes);
        status.put("destination_bytes", totalDestinationBytes);
        status.put("reclaimed_bytes", totalReclaimedBytes);
        status.put("busy_or_closed", skipped.get());
        status.put("failures", failures.get());
        status.put("tracked_paths", retries.size());
        status.put("last_round", lastRound);
        observer.actionStatus(status);
    }

    @Override
    public void close() {
        closed.set(true);
        copies.close();
        executor.shutdown();
    }
}
