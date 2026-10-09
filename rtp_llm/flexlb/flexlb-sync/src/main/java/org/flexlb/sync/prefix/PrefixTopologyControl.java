package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;

import java.net.URI;
import java.time.Clock;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public final class PrefixTopologyControl {
    public record Rank(PrefixTransferCoordinator.Endpoint control, PrefixTransferCoordinator.Endpoint transfer) {
        public Rank {
            Objects.requireNonNull(control);
            Objects.requireNonNull(transfer);
            if (control.equals(transfer)) { throw new IllegalArgumentException("aliased world endpoints"); }
        }
    }
    public record Request(String switchId, String route, int tpSize, int dpSize, long deadlineUnixMs) {
        public Request {
            if (!UUID.fromString(switchId).toString().equals(switchId) || route == null || route.isBlank()
                    || tpSize < 1 || dpSize < 1 || Math.multiplyExact(tpSize, dpSize) > 4096 || deadlineUnixMs <= 0) {
                throw new IllegalArgumentException("invalid topology control request");
            }
        }
    }
    public record Status(ZookeeperPrefixTopologyJournal.Entry entry, boolean frontendAdmitted) { }
    private final Map<String, List<Rank>> worlds;
    private final PrefixWorkerRegistry registry;
    private final PrefixControllerAuthority authority;
    private final ZookeeperPrefixTopologyJournal journal;
    private final PrefixTopologyCoordinator coordinator;
    private final PrefixTopologyWriter writer;
    private final PrefixTopologyPlacement placement;
    private final ConfirmedCacheDirectory directory;
    private final EngineGrpcClient grpc;
    private final String namespace;
    private final Clock clock;
    private final long timeoutMillis;

    public PrefixTopologyControl(Map<String, List<Rank>> worlds, PrefixWorkerRegistry registry, PrefixControllerAuthority authority,
                                  ZookeeperPrefixTopologyJournal journal, PrefixTopologyCoordinator coordinator, PrefixTopologyWriter writer,
                                  PrefixTopologyPlacement placement, ConfirmedCacheDirectory directory, EngineGrpcClient grpc,
                                  String namespace, Clock clock, long timeoutMillis) {
        var copy = new java.util.HashMap<String, List<Rank>>();
        worlds.forEach((route, ranks) -> copy.put(route, List.copyOf(ranks)));
        this.worlds = Map.copyOf(copy);
        this.registry = Objects.requireNonNull(registry);
        this.authority = Objects.requireNonNull(authority);
        this.journal = Objects.requireNonNull(journal);
        this.coordinator = Objects.requireNonNull(coordinator);
        this.writer = Objects.requireNonNull(writer);
        this.placement = Objects.requireNonNull(placement);
        this.directory = Objects.requireNonNull(directory);
        this.grpc = Objects.requireNonNull(grpc);
        this.namespace = Objects.requireNonNull(namespace);
        this.clock = Objects.requireNonNull(clock);
        this.timeoutMillis = timeoutMillis;
        if (worlds.isEmpty() || worlds.size() > 4096 || timeoutMillis < 100 || timeoutMillis > 300000) {
            throw new IllegalArgumentException("invalid topology world configuration");
        }
        var endpoints = new java.util.HashSet<PrefixTransferCoordinator.Endpoint>();
        this.worlds.forEach((route, ranks) -> {
            var uri = URI.create("http://" + route);
            if (uri.getHost() == null || uri.getPort() < 1 || uri.getPort() > 65535 || uri.getUserInfo() != null
                    || !uri.getRawPath().isEmpty() || uri.getQuery() != null || uri.getFragment() != null
                    || ranks.isEmpty() || ranks.size() > 4096) { throw new IllegalArgumentException("invalid configured frontend world"); }
            for (var rank : ranks) {
                if (!endpoints.add(rank.control()) || !endpoints.add(rank.transfer())) { throw new IllegalArgumentException("world endpoints overlap"); }
            }
        });
    }

    public synchronized Status prepare(Request request) {
        var ranks = worlds.get(request.route());
        if (ranks == null || ranks.size() != request.tpSize() * request.dpSize()) { throw new IllegalArgumentException("unknown configured topology world"); }
        var existing = journal.entries().stream().filter(entry -> entry.plan().switchId().equals(request.switchId())).findFirst();
        if (existing.isPresent()) {
            var plan = existing.get().plan();
            if (!route(plan).equals(request.route()) || plan.tpSize() != request.tpSize() || plan.dpSize() != request.dpSize()
                    || plan.deadlineUnixMs() != request.deadlineUnixMs()) { throw new IllegalArgumentException("caller changed immutable topology request"); }
            return status(existing.get());
        }
        var controller = authority.snapshot(namespace);
        long remaining = request.deadlineUnixMs() - clock.millis();
        if (remaining <= 0 || remaining > 300000) { throw new IllegalArgumentException("new topology plan exceeds deadline bound"); }
        var participants = new ArrayList<PrefixTopologyPlan.Participant>();
        var reports = new ArrayList<EngineRpcService.CacheStatusPB>();
        for (int rank = 0; rank < ranks.size(); rank++) {
            remaining = request.deadlineUnixMs() - clock.millis();
            if (remaining <= 0) { throw new IllegalStateException("topology discovery exceeded immutable deadline"); }
            var endpoint = ranks.get(rank).control();
            var report = grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(-1).setNeedCacheKeys(true).build(), Math.min(timeoutMillis, remaining));
            if (!report.hasTopologyIdentity() || !report.hasInventory()) { throw new IllegalStateException("missing original topology identity"); }
            var inventory = report.getInventory();
            participants.add(new PrefixTopologyPlan.Participant(rank, endpoint, inventory.getWorkerId(), report.getTopologyIdentity().getBootId(),
                    inventory.getEpoch(), inventory.getCpDegree(), inventory.getDpLane()));
            reports.add(report);
        }
        boolean residentCache = participants.getFirst().cpDegree() % 2 == 0
                && request.tpSize() == participants.getFirst().cpDegree() / 2;
        boolean residentLeaderMerge = request.tpSize() % 2 == 0
                && request.tpSize() / 2 == participants.getFirst().cpDegree();
        String capabilityName = residentCache ? "resident_prefix_split_v1" : "resident_leader_merge_v1";
        if (residentCache || residentLeaderMerge) {
            for (var participant : participants) {
                remaining = request.deadlineUnixMs() - clock.millis();
                if (remaining <= 0) { throw new IllegalStateException("topology capability discovery exceeded deadline"); }
                var endpoint = participant.controlEndpoint();
                var capability = grpc.getPrefixCacheCapabilities(endpoint.ip(), endpoint.port(),
                        EngineRpcService.PrefixCacheCapabilitiesRequestPB.getDefaultInstance(), Math.min(timeoutMillis, remaining));
                if (capability == null || !capability.getNamespace().equals(namespace)
                        || capability.getCpDegree() != participant.cpDegree() || capability.getDpLane() != participant.dpLane()
                        || (participant.worldRank() % participant.cpDegree() == 0 && !capability.getDeviceReady())
                        || !capability.getSupportedOperationsList().contains(capabilityName)) {
                    residentCache = false;
                    residentLeaderMerge = false;
                    break;
                }
            }
        }
        var plan = new PrefixTopologyPlan(request.switchId(), namespace, controller.getControllerId(), controller.getTerm(),
                request.tpSize(), request.dpSize(), request.deadlineUnixMs(), participants, residentCache, residentLeaderMerge);
        plan.validateBefore(reports);
        plan.payload(clock.millis());
        var registered = registry.registrations().stream().filter(item -> ConfirmedCacheDirectory.routeBase(item.route()).equals(request.route())).toList();
        int degree = participants.getFirst().cpDegree();
        if (registered.size() != ranks.size() / degree) { throw new IllegalStateException("current frontend registration is incomplete"); }
        for (var registration : registered) {
            int rank = registration.dpLane() * degree;
            if (rank < 0 || rank >= ranks.size() || !registration.controlEndpoint().equals(ranks.get(rank).control())
                    || !registration.transferEndpoint().equals(ranks.get(rank).transfer())
                    || !registration.workerId().equals(participants.get(rank).workerId())) {
                throw new IllegalStateException("configured world differs from current registered lanes");
            }
        }
        if (!authority.isCurrent(controller)) { throw new IllegalStateException("controller changed during plan creation"); }
        return status(journal.create(plan));
    }

    public synchronized Status advance(String switchId) {
        var entry = journal.read(switchId);
        route(entry.plan());
        return status(entry.state() == ZookeeperPrefixTopologyJournal.State.PLANNED
                ? writer.advance(entry.plan()) : coordinator.recover(switchId));
    }

    public synchronized Status admit(String switchId) {
        // Keep the admission receipt consistent with its completed placement projection.
        synchronized (placement) {
            return admitProjected(switchId);
        }
    }

    private Status admitProjected(String switchId) {
        var entry = journal.read(switchId);
        var plan = entry.plan();
        String route = route(plan);
        var ranks = worlds.get(route);
        if (entry.state() == ZookeeperPrefixTopologyJournal.State.ADMITTED) { placement.refresh(); return status(journal.read(switchId)); }
        var registrations = new ArrayList<PrefixWorkerRegistry.Registration>();
        for (int lane = 0; lane < plan.dpSize(); lane++) {
            int rank = lane * plan.tpSize();
            registrations.add(new PrefixWorkerRegistry.Registration(plan.dpSize() == 1 ? route : route + "#dp" + lane,
                    plan.participants().get(rank).workerId(), ranks.get(rank).control(), ranks.get(rank).transfer(), lane));
        }
        return status(placement.admit(switchId, registrations));
    }

    public synchronized Status read(String switchId) { return status(journal.read(switchId)); }

    private String route(PrefixTopologyPlan plan) {
        var endpoints = plan.participants().stream().map(PrefixTopologyPlan.Participant::controlEndpoint).toList();
        return worlds.entrySet().stream().filter(world -> world.getValue().stream().map(Rank::control).toList().equals(endpoints))
                .map(Map.Entry::getKey).findFirst().orElseThrow(() -> new IllegalStateException("plan world is not configured"));
    }

    private Status status(ZookeeperPrefixTopologyJournal.Entry entry) {
        var plan = entry.plan();
        String route = route(plan);
        boolean admitted = entry.state() == ZookeeperPrefixTopologyJournal.State.ADMITTED;
        for (int lane = 0; lane < plan.dpSize() && admitted; lane++) {
            admitted = directory.routingAvailable(route, plan.dpSize() == 1 ? route : route + "#dp" + lane, plan.tpSize(), plan.dpSize());
        }
        return new Status(entry, admitted);
    }
}
