package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class PrefixTopologyPlacement {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(PrefixTopologyPlacement.class);
    private static final int MAX_PARALLEL_WORLDS = 16;
    private final Map<String, String> projectionFailures = new HashMap<>();
    private final ZookeeperPrefixTopologyJournal journal;
    private final PrefixControllerAuthority authority;
    private final EngineGrpcClient grpc;
    private final ConfirmedCacheDirectory directory;
    private final PrefixWorkerRegistry registry;
    private final long timeoutMillis;
    private final long observationMillis;
    private ZookeeperPrefixTopologyBarrier barrier;
    private PrefixTopologyFrontend frontend;

    public PrefixTopologyPlacement(ZookeeperPrefixTopologyJournal journal, PrefixControllerAuthority authority,
                                   EngineGrpcClient grpc, ConfirmedCacheDirectory directory, PrefixWorkerRegistry registry,
                                   long timeoutMillis, long observationMillis) {
        this.journal = Objects.requireNonNull(journal);
        this.authority = Objects.requireNonNull(authority);
        this.grpc = Objects.requireNonNull(grpc);
        this.directory = Objects.requireNonNull(directory);
        this.registry = Objects.requireNonNull(registry);
        if (timeoutMillis < 1 || timeoutMillis > 300000 || observationMillis < 1 || observationMillis > 300000) {
            throw new IllegalArgumentException("invalid topology placement time bound");
        }
        this.timeoutMillis = timeoutMillis;
        this.observationMillis = observationMillis;
        directory.suspendTopology();
    }

    public synchronized ZookeeperPrefixTopologyJournal.Entry admit(String switchId,
                                                                  List<PrefixWorkerRegistry.Registration> registrations) {
        directory.suspendTopology();
        var expected = journal.read(switchId);
        var controller = authority.snapshot(expected.plan().namespace());
        if (!authority.isCurrent(controller) || !controller.getNamespace().equals(expected.plan().namespace())
                || Long.compareUnsigned(controller.getTerm(), expected.plan().controllerTerm()) < 0
                || (controller.getTerm() == expected.plan().controllerTerm() && !controller.equals(expected.plan().controller()))) {
            throw new IllegalStateException("placement admission requires current non-stale authority");
        }
        var reports = inspect(expected.plan());
        PrefixTopologyAdmission.validate(expected.plan(), registrations, reports);
        validateFrontend(expected.plan(), registrations);
        if (frontend != null) { frontend.validate(expected.plan(), registrations, reports); }
        new PrefixWorkerRegistry(registry.registrations()).replaceWorld(PrefixTopologyAdmission.workers(expected.plan()), registrations);
        if (!authority.isCurrent(controller)) { throw new IllegalStateException("placement authority changed"); }
        var admitted = journal.admitted(expected, registrations, reports);
        refresh();
        return admitted;
    }

    public synchronized void refresh() {
        directory.suspendTopology();
        var entries = journal.entries();
        var latest = latest(entries);
        var observations = observe(latest);
        Map<String, ConfirmedCacheDirectory.TopologyFence> fences = new HashMap<>();
        for (var entry : latest) {
            requireNotInterrupted();
            var workers = PrefixTopologyAdmission.workers(entry.plan());
            var routes = new HashSet<String>();
            registry.registrations().stream().filter(registration -> workers.contains(registration.workerId()))
                    .map(registration -> ConfirmedCacheDirectory.routeBase(registration.route())).forEach(routes::add);
            entry.registrations().stream().map(registration -> ConfirmedCacheDirectory.routeBase(registration.route())).forEach(routes::add);
            if (routes.isEmpty()) { throw new IllegalStateException("topology world has no registered frontend fence"); }
            fences.put(entry.plan().switchId(), new ConfirmedCacheDirectory.TopologyFence(workers, routes));
            if (entry.state() != ZookeeperPrefixTopologyJournal.State.ADMITTED) { continue; }
            try {
                var observation = observations.get(entry.plan().switchId());
                if (observation.failure() != null) { throw observation.failure(); }
                var reports = observation.reports();
                var recorded = ZookeeperPrefixTopologyJournal.decodeReports(entry.reports());
                PrefixTopologyAdmission.validateContinuation(recorded, reports);
                var inventory = PrefixTopologyAdmission.validate(entry.plan(), entry.registrations(), reports);
                validateFrontend(entry.plan(), entry.registrations());
                if (frontend != null) { frontend.reconcile(entry.plan(), entry.registrations(), reports, observation.frontend()); }
                var updated = new PrefixWorkerRegistry(registry.registrations());
                updated.replaceWorld(workers, entry.registrations());
                directory.installTopologyReports(workers, inventory);
                registry.replaceWorld(workers, entry.registrations());
                fences.remove(entry.plan().switchId());
                projectionFailures.remove(entry.plan().switchId());
            } catch (RuntimeException unavailable) {
                requireNotInterrupted();
                String reason = unavailable.getClass().getSimpleName() + ": " + unavailable.getMessage();
                if (!reason.equals(projectionFailures.put(entry.plan().switchId(), reason))) {
                    LOG.warn("Topology placement remains fenced: switchId={}, reason={}", entry.plan().switchId(), reason);
                }
                continue;
            }
        }
        if (!journal.entries().equals(entries)) { throw new IllegalStateException("topology journal changed during projection"); }
        var preservationRoutes = new HashSet<String>();
        fences.values().forEach(fence -> registry.registrations().stream()
                .filter(registration -> fence.workers().contains(registration.workerId()))
                .forEach(registration -> preservationRoutes.add(registration.route())));
        // Resident repair only reads/writes its own world. External preservation
        // still drains every potential donor and destination before freezing KV.
        boolean externalPreservation = latest.stream().anyMatch(entry -> fences.containsKey(entry.plan().switchId())
                && !entry.plan().residentCache() && !entry.plan().residentLeaderMerge());
        if (barrier != null && externalPreservation) {
            var allWorkers = registry.registrations().stream().map(PrefixWorkerRegistry.Registration::workerId)
                    .collect(java.util.stream.Collectors.toSet());
            var allRoutes = registry.registrations().stream().map(registration -> ConfirmedCacheDirectory.routeBase(registration.route()))
                    .collect(java.util.stream.Collectors.toSet());
            fences.replaceAll((id, fence) -> new ConfirmedCacheDirectory.TopologyFence(allWorkers, allRoutes));
        }
        directory.publishTopologyFences(fences, observationMillis, barrier != null && !fences.isEmpty());
        if (barrier != null) { barrier.publish(fences, preservationRoutes); }
    }

    public synchronized void attachBarrier(ZookeeperPrefixTopologyBarrier barrier) {
        if (this.barrier != null) { throw new IllegalStateException("topology barrier already attached"); }
        this.barrier = Objects.requireNonNull(barrier);
        directory.topologyGuard(barrier::healthy);
    }

    public synchronized void attachFrontend(PrefixTopologyFrontend frontend) {
        if (this.frontend != null) { throw new IllegalStateException("topology frontend already attached"); }
        this.frontend = Objects.requireNonNull(frontend);
    }

    public synchronized <Result> Result exclusive(java.util.function.Supplier<Result> operation) {
        try (var permit = directory.acquireMaintenance()) { return operation.get(); }
    }

    static List<ZookeeperPrefixTopologyJournal.Entry> latest(List<ZookeeperPrefixTopologyJournal.Entry> entries) {
        var selected = new ArrayList<ZookeeperPrefixTopologyJournal.Entry>();
        var workers = new HashSet<String>();
        var endpoints = new HashSet<PrefixTransferCoordinator.Endpoint>();
        for (var entry : entries.stream().sorted((left, right) -> Long.compareUnsigned(right.creationOrder(), left.creationOrder())).toList()) {
            if (entry.state() == ZookeeperPrefixTopologyJournal.State.CANCELLED) { continue; }
            var world = PrefixTopologyAdmission.workers(entry.plan());
            var controls = entry.plan().participants().stream().map(PrefixTopologyPlan.Participant::controlEndpoint)
                    .collect(java.util.stream.Collectors.toSet());
            if (world.stream().anyMatch(workers::contains) || controls.stream().anyMatch(endpoints::contains)) {
                if (!workers.containsAll(world) || !endpoints.containsAll(controls)) {
                    throw new IllegalStateException("overlapping topology worlds cannot be projected");
                }
                continue;
            }
            selected.add(entry);
            workers.addAll(world);
            endpoints.addAll(controls);
        }
        return List.copyOf(selected);
    }

    private List<EngineRpcService.CacheStatusPB> inspect(PrefixTopologyPlan plan) {
        var reports = new ArrayList<EngineRpcService.CacheStatusPB>();
        for (var participant : plan.participants()) {
            requireNotInterrupted();
            var endpoint = participant.controlEndpoint();
            reports.add(grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(-1).setNeedCacheKeys(true).build(), timeoutMillis));
        }
        return List.copyOf(reports);
    }

    private record Observation(List<EngineRpcService.CacheStatusPB> reports, CacheStatus frontend,
                               RuntimeException failure) {}

    private Map<String, Observation> observe(List<ZookeeperPrefixTopologyJournal.Entry> entries) {
        var observations = new HashMap<String, Observation>();
        // Independent worlds can be read concurrently. Validation and publication
        // remain serialized under the placement lock, including the journal check.
        try (var executor = Executors.newFixedThreadPool(MAX_PARALLEL_WORLDS,
                Thread.ofVirtual().name("prefix-topology-observation-", 0).factory())) {
            var pending = new HashMap<String, Future<Observation>>();
            for (var entry : entries) {
                if (entry.state() != ZookeeperPrefixTopologyJournal.State.ADMITTED) { continue; }
                pending.put(entry.plan().switchId(), executor.submit(() -> {
                    try {
                        var reports = inspect(entry.plan());
                        var response = frontend == null ? null : frontend.fetch(entry.registrations());
                        return new Observation(reports, response, null);
                    } catch (RuntimeException failure) { return new Observation(null, null, failure); }
                }));
            }
            try {
                for (var request : pending.entrySet()) { observations.put(request.getKey(), request.getValue().get()); }
            } catch (InterruptedException interrupted) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
                throw new IllegalStateException("topology observation interrupted", interrupted);
            } catch (ExecutionException failed) {
                executor.shutdownNow();
                throw new IllegalStateException("topology observation failed", failed.getCause());
            }
        }
        return observations;
    }

    private void validateFrontend(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> replacements) {
        var workers = PrefixTopologyAdmission.workers(plan);
        var previous = registry.registrations().stream().filter(registration -> workers.contains(registration.workerId()))
                .map(registration -> ConfirmedCacheDirectory.routeBase(registration.route())).collect(java.util.stream.Collectors.toSet());
        var updated = replacements.stream().map(registration -> ConfirmedCacheDirectory.routeBase(registration.route()))
                .collect(java.util.stream.Collectors.toSet());
        if (!previous.equals(updated)) { throw new IllegalStateException("frontend relocation needs explicit route retirement"); }
    }

    private static void requireNotInterrupted() {
        if (Thread.currentThread().isInterrupted()) { throw new IllegalStateException("topology placement interrupted"); }
    }
}
