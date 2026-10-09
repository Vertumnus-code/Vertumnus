package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.cache.prefix.PrefixObserver;
import org.flexlb.cache.prefix.PrefixPreservationPlanner;
import org.flexlb.cache.prefix.PrefixReplicaPlanner;
import org.flexlb.dao.cache.CacheReportV2;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;

import java.time.Clock;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class PrefixTopologyWriter {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(PrefixTopologyWriter.class);
    private final PrefixObserver observer;
    private final PrefixWorkerRegistry registry;
    private final EngineGrpcClient grpc;
    private final PrefixControllerAuthority authority;
    private final PrefixTransferJournal transfers;
    private final ZookeeperPrefixTopologyJournal journal;
    private final PrefixTopologyCoordinator coordinator;
    private final PrefixTopologyPlacement placement;
    private final ZookeeperPrefixTopologyBarrier barrier;
    private final PrefixCopyExecutor.Limits limits;
    private final Set<PrefixCopyExecutor.DegreePair> pairs;
    private final Clock clock;

    public PrefixTopologyWriter(PrefixObserver observer, PrefixWorkerRegistry registry, EngineGrpcClient grpc,
                                PrefixControllerAuthority authority, PrefixTransferJournal transfers,
                                ZookeeperPrefixTopologyJournal journal, PrefixTopologyCoordinator coordinator,
                                PrefixTopologyPlacement placement, ZookeeperPrefixTopologyBarrier barrier,
                                PrefixCopyExecutor.Limits limits, Set<PrefixCopyExecutor.DegreePair> pairs, Clock clock) {
        this.observer = Objects.requireNonNull(observer);
        this.registry = Objects.requireNonNull(registry);
        this.grpc = Objects.requireNonNull(grpc);
        this.authority = Objects.requireNonNull(authority);
        this.transfers = Objects.requireNonNull(transfers);
        this.journal = Objects.requireNonNull(journal);
        this.coordinator = Objects.requireNonNull(coordinator);
        this.placement = Objects.requireNonNull(placement);
        this.barrier = Objects.requireNonNull(barrier);
        this.limits = Objects.requireNonNull(limits);
        this.pairs = Set.copyOf(pairs);
        this.clock = Objects.requireNonNull(clock);
        if (!journal.requiresPreservation()) { throw new IllegalArgumentException("topology writer requires preservation journal"); }
        if (!observer.settings().confirmed() || !observer.settings().observe()) {
            throw new IllegalArgumentException("topology writer requires actual confirmed demand observation");
        }
    }

    public synchronized ZookeeperPrefixTopologyJournal.Entry advance(PrefixTopologyPlan plan) {
        requireAuthority(plan);
        var entry = journal.create(plan);
        if (entry.state() != ZookeeperPrefixTopologyJournal.State.PLANNED) { return coordinator.recover(plan.switchId()); }
        if (plan.deadlineUnixMs() <= clock.millis()) { return coordinator.cancelPlanned(plan.switchId()); }
        inventory(plan);
        placement.refresh();
        var cohort = barrier.capture(plan.switchId());
        return placement.exclusive(() -> preserveAndSubmit(plan, cohort));
    }

    private ZookeeperPrefixTopologyJournal.Entry preserveAndSubmit(PrefixTopologyPlan plan,
                                                                    ZookeeperPrefixTopologyBarrier.Cohort cohort) {
        requireAuthority(plan);
        requireSettledTransfers();
        var entry = journal.read(plan.switchId());
        if (entry.state() != ZookeeperPrefixTopologyJournal.State.PLANNED) { return entry; }
        var participants = new ArrayList<EngineRpcService.CacheStatusPB>();
        for (var participant : plan.participants()) {
            var endpoint = participant.controlEndpoint();
            participants.add(grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(-1).setNeedCacheKeys(true).build(), limits.requestTimeoutMillis()));
        }
        plan.validateBefore(participants);
        var current = inventory(plan);
        if (entry.preservation() == null && plan.residentLeaderMerge()) {
            entry = journal.planResidentMerge(entry, residentMergeLanes(plan, current, freeze(current)));
            for (var merge : entry.preservation().residentMerges()) {
                LOG.info("Resident merge inventory: switchId={} primary={} secondary={} retainedBlocks={} retiredSecondaryBlocks={}",
                        plan.switchId(), merge.primaryRoute(), merge.secondaryRoute(),
                        merge.retainedKeys().size(), merge.retiredKeys().size());
            }
        }
        if (entry.preservation() == null) {
            var demands = cohort.demands().stream().flatMap(snapshot -> snapshot.demands().stream()).toList();
            var workloads = new HashMap<String, Double>();
            cohort.demands().forEach(snapshot -> snapshot.workloads().forEach((route, count) -> workloads.merge(route, count, Double::sum)));
            var affectedWorkers = PrefixTopologyAdmission.workers(plan);
            var affected = registry.registrations().stream().filter(registration -> affectedWorkers.contains(registration.workerId()))
                    .map(PrefixWorkerRegistry.Registration::route).collect(java.util.stream.Collectors.toSet());
            // A flat inventory cannot prove suffix closure. Definitive rejection
            // selects a complete non-protected remainder instead of another subset.
            var rejectedReclaims = transfers.entries().stream()
                    .filter(transfer -> transfer.intent().action() == PrefixTransferJournal.Action.RECLAIM
                            && transfer.result() != null && transfer.result().outcome() == PrefixTransferCoordinator.Outcome.REJECTED
                            && !transfer.result().cleanupPending()
                            && transfer.result().committedVersion() == transfer.intent().destination().version())
                    .toList();
            var reclaimable = new HashMap<String, List<Long>>();
            var rejectedKeys = new HashMap<String, List<List<Long>>>();
            current.forEach((route, report) -> {
                // Without per-key pin metadata, only an entirely evictable, drained
                // inventory can nominate victims. Native version/ref checks still apply.
                if (!affected.contains(route) && report.memoryStatsConsistent() && report.activeBytes() == 0
                        && report.residentBytes() == 0 && report.operationReservedBytes() == 0
                        && Math.multiplyExact(report.deviceKeys().size(), report.physicalBlockBytes()) <= report.evictableBytes()) {
                    reclaimable.put(route, observer.reclamationOrder(Set.copyOf(report.deviceKeys())));
                    var rejected = rejectedReclaims.stream().filter(transfer -> {
                        var holder = transfer.intent().destination();
                        return holder.workerId().equals(report.workerId()) && holder.dpLane() == report.dpLane()
                                && holder.epoch().equals(report.epoch()) && holder.layoutFingerprint().equals(report.layoutFingerprint())
                                && holder.version() == report.version();
                    }).map(transfer -> transfer.intent().keys()).toList();
                    if (!rejected.isEmpty()) { rejectedKeys.put(route, rejected); }
                }
            });
            var preservationInventory = plan.residentCache() ? current.entrySet().stream()
                    .filter(item -> affected.contains(item.getKey()))
                    .collect(java.util.stream.Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)) : current;
            var preservation = preservationPlan(demands, preservationInventory, workloads, affected, reclaimable, rejectedKeys,
                    plan.residentCache() ? affected : Set.of());
            for (var reclaim : preservation.reclaims()) {
                refreshBarrier(cohort);
                requireAuthority(plan);
                requireSettledTransfers();
                current = inventory(plan);
                reclaimHeadroom(plan, reclaim, current);
            }
            if (!preservation.reclaims().isEmpty()) {
                // Validate the allocation that motivated reclamation. A fresh greedy
                // assignment can steal headroom reserved for a later, longer path.
                current = inventory(plan);
                verifyPhysicalHeadroom(preservation, current, limits.safetyBytes());
            }
            var endpoints = freeze(current);
            var required = preservation.required().stream().map(item -> new PrefixTopologyPreservation.Required(item.holder(),
                    Objects.requireNonNull(endpoints.get(item.holder())), item.path())).toList();
            entry = journal.planPreservation(entry, required, preservation.copies());
        }
        if (!entry.preservation().complete()) {
            for (var proposal : entry.preservation().copies()) {
                refreshBarrier(cohort);
                requireAuthority(plan);
                requireSettledTransfers();
                current = inventory(plan);
                var destination = Objects.requireNonNull(current.get(proposal.destination()));
                if (destination.deviceKeys().containsAll(proposal.path())) { continue; }
                // A committed copy without its physical inventory is a failed proof,
                // not permission to allocate again. Unissued copies can resume.
                if (transfers.entries().stream().anyMatch(transfer -> transfer.intent().footprint() != null
                        && plan.switchId().equals(transfer.intent().footprint().topologySwitchId())
                        && transfer.intent().keys().equals(proposal.path())
                        && transfer.intent().destination().workerId().equals(destination.workerId())
                        && transfer.intent().destination().dpLane() == destination.dpLane()
                        && transfer.result() != null && transfer.result().committed())) {
                    throw new IllegalStateException("committed preservation copy lacks inventory proof");
                }
                copy(plan, proposal, current);
            }
        }
        requireSettledTransfers();
        current = inventory(plan);
        var proof = proof(entry.preservation(), current);
        new PrefixTopologyPreservation(entry.preservation().required(), true, proof,
                entry.preservation().copies(), entry.preservation().residentMerges()).validate(plan);
        if (!entry.preservation().complete()) { entry = journal.preserved(entry, proof); }
        refreshBarrier(cohort);
        requireAuthority(plan);
        return coordinator.submit(plan, () -> {
            refreshBarrier(cohort);
            requireAuthority(plan);
            requireSettledTransfers();
            var expected = journal.read(plan.switchId()).preservation();
            new PrefixTopologyPreservation(expected.required(), true, proof(expected, inventory(plan)),
                    expected.copies(), expected.residentMerges()).validate(plan);
            barrier.verify(cohort);
        });
    }

    private PrefixPreservationPlanner.Plan preservationPlan(List<PrefixReplicaPlanner.Demand> demands,
            Map<String, CacheReportV2> reports, Map<String, Double> workloads, Set<String> affected,
            Map<String, List<Long>> reclaimable, Map<String, List<List<Long>>> rejectedReclaims, Set<String> residentHolders) {
        return new PrefixPreservationPlanner().plan(demands, workers(reports, workloads), affected,
                new PrefixPreservationPlanner.Limits(limits.maxOperations(), limits.destinationBytes(), limits.wireBytes(), limits.safetyBytes()),
                (source, destination) -> !reports.get(source.lane()).workerId().equals(reports.get(destination.lane()).workerId())
                        && (source.degree() == destination.degree() || pairs.contains(new PrefixCopyExecutor.DegreePair(source.degree(), destination.degree()))),
                reclaimable, rejectedReclaims, residentHolders);
    }

    static void verifyPhysicalHeadroom(PrefixPreservationPlanner.Plan plan, Map<String, CacheReportV2> reports,
                                       long safetyBytes) {
        var installed = new HashMap<String, Set<Long>>();
        var remaining = new HashMap<String, Long>();
        reports.forEach((route, report) -> {
            installed.put(route, new java.util.HashSet<>(report.deviceKeys()));
            remaining.put(route, report.physicalFreeBytes());
        });
        for (var copy : plan.copies()) {
            var source = Objects.requireNonNull(reports.get(copy.source()));
            var destination = Objects.requireNonNull(reports.get(copy.destination()));
            if (!source.deviceKeys().containsAll(copy.path()) || destination.cpDegree() != copy.degree()) {
                throw new IllegalStateException("preservation allocation lost its source or destination geometry");
            }
            var keys = installed.get(copy.destination());
            long missing = copy.path().stream().filter(key -> !keys.contains(key)).count();
            long bytes = Math.multiplyExact(missing, destination.physicalBlockBytes());
            if (bytes > copy.destinationPhysicalBytes()
                    || (bytes > 0 && Math.addExact(bytes, safetyBytes) > remaining.get(copy.destination()))) {
                throw new IllegalStateException("preservation allocation lacks confirmed physical headroom");
            }
            remaining.compute(copy.destination(), (route, free) -> Math.subtractExact(free, bytes));
            keys.addAll(copy.path());
        }
        for (var required : plan.required()) {
            if (!Objects.requireNonNull(installed.get(required.holder())).containsAll(required.path())) {
                throw new IllegalStateException("preservation allocation lost a required holder");
            }
        }
    }

    private void reclaimHeadroom(PrefixTopologyPlan plan, PrefixPreservationPlanner.Reclaim reclaim,
                                 Map<String, CacheReportV2> reports) {
        long remaining = plan.deadlineUnixMs() - clock.millis();
        if (remaining < limits.requestTimeoutMillis()) { throw new IllegalStateException("preservation reclaim deadline expired"); }
        var isolated = directory(reports);
        var destination = Objects.requireNonNull(registry.freeze(isolated.view()).get(reclaim.route()));
        var coordinator = new PrefixReclaimCoordinator(new GrpcPrefixTransferWorkerClient(grpc, limits.requestTimeoutMillis()),
                authority, transfers, isolated);
        var result = coordinator.reclaim(destination, plan.namespace(), reclaim.keys(), Math.min(remaining, 300000), plan.controller());
        if (result.outcome() != PrefixTransferCoordinator.Outcome.RECLAIMED || result.cleanupPending()) {
            throw new IllegalStateException("preservation reclaim is not confirmed: " + result);
        }
        var after = Objects.requireNonNull(inventory(plan).get(reclaim.route()));
        var before = Objects.requireNonNull(reports.get(reclaim.route()));
        long expectedFree = Math.addExact(before.physicalFreeBytes(),
                Math.multiplyExact(reclaim.keys().size(), before.physicalBlockBytes()));
        if (!after.epoch().equals(before.epoch()) || after.version() < result.committedVersion()
                || after.deviceKeys().stream().anyMatch(reclaim.keys()::contains) || after.physicalFreeBytes() < expectedFree) {
            throw new IllegalStateException("preservation reclaim lacks fresh physical headroom proof");
        }
        LOG.info("Topology preservation reclaimed headroom: switchId={} requestId={} route={} blocks={} blockBytes={} beforeFreeBytes={} afterFreeBytes={} version={}",
                plan.switchId(), result.requestId(), reclaim.route(), reclaim.keys().size(), before.physicalBlockBytes(),
                before.physicalFreeBytes(), after.physicalFreeBytes(), after.version());
    }

    private void copy(PrefixTopologyPlan plan, PrefixReplicaPlanner.Proposal proposal, Map<String, CacheReportV2> current) {
        var destination = Objects.requireNonNull(current.get(proposal.destination()));
        var missing = proposal.path().stream().filter(key -> !destination.deviceKeys().contains(key)).toList();
        if (missing.isEmpty()) { return; }
        long bytes = Math.multiplyExact(missing.size(), destination.physicalBlockBytes());
        if (bytes > proposal.destinationPhysicalBytes()) { throw new IllegalStateException("preservation inventory lost planned blocks"); }
        var refreshed = new PrefixReplicaPlanner.Proposal(proposal.kind(), proposal.path(), proposal.degree(), proposal.source(),
                proposal.destination(), missing, bytes, bytes, proposal.priority(), proposal.replicas());
        var isolated = directory(current);
        try (var copies = new PrefixCopyExecutor(grpc, authority, transfers, isolated, plan.namespace(), limits, pairs, plan.switchId())) {
            long remaining = plan.deadlineUnixMs() - clock.millis();
            if (remaining < limits.requestTimeoutMillis()) { throw new IllegalStateException("preservation deadline expired"); }
            var round = copies.execute(List.of(refreshed), registry.freeze(isolated.view()), plan.deadlineUnixMs())
                    .toCompletableFuture().get(remaining, TimeUnit.MILLISECONDS);
            if (round.attempts().size() != 1 || round.attempts().getFirst().result() == null
                    || round.attempts().getFirst().result().outcome() != PrefixTransferCoordinator.Outcome.COMMITTED
                    || round.attempts().getFirst().result().cleanupPending()) {
                throw new IllegalStateException("preservation copy is not confirmed: " + round);
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("preservation copy interrupted; durable intent retained", interrupted);
        } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException failure) {
            throw new IllegalStateException("preservation copy unresolved; durable intent retained", failure);
        }
    }

    private void refreshBarrier(ZookeeperPrefixTopologyBarrier.Cohort cohort) {
        placement.refresh();
        barrier.verify(cohort);
    }

    private void requireAuthority(PrefixTopologyPlan plan) {
        var controller = authority.snapshot(plan.namespace());
        if (!observer.settings().namespace().equals(plan.namespace()) || !controller.equals(plan.controller())
                || !authority.isCurrent(controller)) { throw new IllegalStateException("topology writer no longer owns immutable plan"); }
    }

    private void requireSettledTransfers() {
        if (transfers.entries().stream().anyMatch(entry -> entry.result() == null || entry.result().cleanupPending()
                || entry.result().outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN)) {
            throw new IllegalStateException("unresolved durable transfer blocks topology preservation");
        }
    }

    private Map<String, CacheReportV2> inventory(PrefixTopologyPlan plan) {
        var result = new HashMap<String, CacheReportV2>();
        var affected = PrefixTopologyAdmission.workers(plan);
        for (var registration : registry.registrations()) {
            if ((plan.residentCache() || plan.residentLeaderMerge()) && !affected.contains(registration.workerId())) {
                continue;
            }
            if (Thread.currentThread().isInterrupted()) { throw new IllegalStateException("topology inventory interrupted"); }
            var endpoint = registration.controlEndpoint();
            var status = grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(-1).setNeedCacheKeys(true).build(), limits.requestTimeoutMillis());
            if (!status.hasInventory()) { throw new IllegalStateException("topology preservation requires native inventory"); }
            var report = report(status.getInventory());
            if (!observer.settings().namespace().equals(report.namespace()) || !registration.workerId().equals(report.workerId())
                    || registration.dpLane() != report.dpLane() || !report.deviceReady() || !"FULL".equals(report.snapshotKind())
                    || (!report.deviceKeys().isEmpty() && report.rankReadinessGeneration() < 1)
                    || report.cpDegree() < 1 || report.physicalBlockBytes() < 1) {
                throw new IllegalStateException("topology preservation inventory is not ready or registered");
            }
            observer.report(registration.route(), report);
            result.put(registration.route(), report);
        }
        return Map.copyOf(result);
    }

    static CacheReportV2 report(EngineRpcService.CacheReportV2PB inventory) {
        return new CacheReportV2(inventory.getNamespace(), inventory.getWorkerId(), inventory.getDpLane(), inventory.getEpoch(),
                inventory.getLayoutFingerprint(), inventory.getSnapshotKind(), inventory.getVersion(), inventory.getBaseVersion(),
                inventory.getRankReadinessGeneration(), inventory.getDeviceReady(), inventory.getCpDegree(), inventory.getBlockSize(),
                inventory.getPhysicalFreeBytes(), inventory.getDeviceKeysList(), inventory.getReuseEventsList().stream()
                .map(event -> new CacheReportV2.PrefixReuseEvent(event.getSequence(), event.getRequestId(), event.getDeviceReuseTokens(),
                        event.getInputTokens(), event.getPrefill(), event.getAttemptId())).toList(),
                inventory.getDroppedReuseEvents(), inventory.getPhysicalBlockBytes(), inventory.getEvictableBytes(),
                inventory.getActiveBytes(), inventory.getResidentBytes(), inventory.getOperationReservedBytes(),
                inventory.getPhysicalMemoryStatsValid(), inventory.getPhysicalCapacityBytes(),
                inventory.hasCopyLifetime() ? org.flexlb.service.grpc.EngineStatusConverter.convertCopyLifetime(inventory.getCopyLifetime()) : null);
    }

    private ConfirmedCacheDirectory directory(Map<String, CacheReportV2> reports) {
        var isolated = new ConfirmedCacheDirectory(observer.settings().namespace(), clock, limits.transferTimeoutMillis());
        reports.forEach((route, report) -> {
            if (isolated.apply(route, report) != ConfirmedCacheDirectory.Result.APPLIED) {
                throw new IllegalStateException("invalid preservation inventory projection");
            }
        });
        return isolated;
    }

    private Map<String, PrefixTransferCoordinator.WorkerSnapshot> freeze(Map<String, CacheReportV2> reports) {
        return registry.freeze(directory(reports).view());
    }

    private List<PrefixReplicaPlanner.Worker> workers(Map<String, CacheReportV2> reports, Map<String, Double> workloads) {
        var result = new ArrayList<PrefixReplicaPlanner.Worker>();
        reports.forEach((route, report) -> result.add(new PrefixReplicaPlanner.Worker(route, report.workerId() + "/" + report.dpLane(),
                report.cpDegree(), Set.copyOf(report.deviceKeys()), report.physicalBlockBytes(), report.physicalFreeBytes(),
                workloads.getOrDefault(route, 0.0))));
        return List.copyOf(result);
    }

    private Map<String, CacheReportV2> proof(PrefixTopologyPreservation preservation, Map<String, CacheReportV2> reports) {
        var result = new HashMap<String, CacheReportV2>();
        preservation.required().forEach(item -> result.put(item.route(), Objects.requireNonNull(reports.get(item.route()))));
        preservation.residentMerges().forEach(merge -> {
            result.put(merge.primaryRoute(), Objects.requireNonNull(reports.get(merge.primaryRoute())));
            result.put(merge.secondaryRoute(), Objects.requireNonNull(reports.get(merge.secondaryRoute())));
        });
        return Map.copyOf(result);
    }

    static List<PrefixTopologyPreservation.ResidentMerge> residentMergeLanes(PrefixTopologyPlan plan,
            Map<String, CacheReportV2> reports, Map<String, PrefixTransferCoordinator.WorkerSnapshot> endpoints) {
        if (!plan.residentLeaderMerge()) { throw new IllegalArgumentException("resident merge mode is required"); }
        var result = new ArrayList<PrefixTopologyPreservation.ResidentMerge>();
        for (int rank = 0; rank < plan.participants().size(); rank += plan.tpSize()) {
            var primary = plan.participants().get(rank);
            var secondary = plan.participants().get(rank + plan.tpSize() / 2);
            String primaryRoute = residentRoute(primary, endpoints);
            String secondaryRoute = residentRoute(secondary, endpoints);
            result.add(new PrefixTopologyPreservation.ResidentMerge(primaryRoute, endpoints.get(primaryRoute),
                    Objects.requireNonNull(reports.get(primaryRoute)).deviceKeys(), secondaryRoute, endpoints.get(secondaryRoute),
                    Objects.requireNonNull(reports.get(secondaryRoute)).deviceKeys()));
        }
        return List.copyOf(result);
    }

    private static String residentRoute(PrefixTopologyPlan.Participant participant,
                                        Map<String, PrefixTransferCoordinator.WorkerSnapshot> endpoints) {
        var routes = endpoints.entrySet().stream().filter(entry -> {
            var holder = entry.getValue();
            return holder.workerId().equals(participant.workerId()) && holder.dpLane() == participant.dpLane()
                    && holder.epoch().equals(participant.epoch()) && holder.rankCount() == participant.cpDegree()
                    && holder.controlEndpoint().equals(participant.controlEndpoint());
        }).map(Map.Entry::getKey).toList();
        if (routes.size() != 1) { throw new IllegalStateException("resident lane does not have one current registered route"); }
        return routes.getFirst();
    }
}
