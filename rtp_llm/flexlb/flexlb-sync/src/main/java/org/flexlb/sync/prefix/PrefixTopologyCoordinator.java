package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;

import java.time.Clock;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PrefixTopologyCoordinator {
    private final EngineGrpcClient grpc;
    private final PrefixControllerAuthority authority;
    private final ZookeeperPrefixTopologyJournal journal;
    private final Clock clock;
    private final long timeoutMillis;

    public PrefixTopologyCoordinator(EngineGrpcClient grpc, PrefixControllerAuthority authority,
                                     ZookeeperPrefixTopologyJournal journal, Clock clock, long timeoutMillis) {
        this.grpc = Objects.requireNonNull(grpc);
        this.authority = Objects.requireNonNull(authority);
        this.journal = Objects.requireNonNull(journal);
        this.clock = Objects.requireNonNull(clock);
        if (timeoutMillis < 100 || timeoutMillis > 300000) { throw new IllegalArgumentException("invalid topology RPC timeout"); }
        this.timeoutMillis = timeoutMillis;
    }

    public ZookeeperPrefixTopologyJournal.Entry submit(PrefixTopologyPlan plan) {
        return submit(plan, () -> { });
    }

    public ZookeeperPrefixTopologyJournal.Entry submit(PrefixTopologyPlan plan, Runnable beforeSubmission) {
        var controller = authority.snapshot(plan.namespace());
        if (!controller.equals(plan.controller()) || !authority.isCurrent(controller)) {
            throw new IllegalStateException("topology authority no longer owns caller plan");
        }
        var entry = journal.create(plan);
        if (entry.state() != ZookeeperPrefixTopologyJournal.State.PLANNED) { return recover(plan.switchId()); }
        String payload;
        try {
            plan.validateBefore(inspect(plan));
            for (var participant : plan.participants()) {
                if (!authority.isCurrent(controller)) { throw new IllegalStateException("topology authority changed"); }
                var endpoint = participant.controlEndpoint();
                var response = grpc.activatePrefixController(endpoint.ip(), endpoint.port(),
                        EngineRpcService.ActivatePrefixControllerRequestPB.newBuilder().setController(controller)
                                .setEpoch(participant.epoch()).build(), timeoutMillis);
                if (!response.getSuccess()) { throw new IllegalStateException("topology participant rejected controller"); }
            }
            plan.validateBefore(inspect(plan));
            payload = plan.payload(clock.millis());
            if (!authority.isCurrent(controller)) { throw new IllegalStateException("topology authority changed before dispatch"); }
            beforeSubmission.run();
        } catch (RuntimeException preflightFailure) {
            try { return journal.cancelPlanned(entry); }
            catch (RuntimeException raced) { return journal.read(plan.switchId()); }
        }
        entry = journal.submitted(entry);
        var coordinator = plan.participants().getFirst().controlEndpoint();
        try {
            long remaining = plan.deadlineUnixMs() - clock.millis();
            if (!authority.isCurrent(controller) || remaining <= 0) { return entry; }
            grpc.submitPrefixTopology(coordinator.ip(), coordinator.port(), EngineRpcService.UpdateSchedulerInfoRequestPB
                    .newBuilder().setSchedulerInfo(payload).build(), Math.min(timeoutMillis, remaining));
        } catch (RuntimeException unknown) {
            return recover(plan.switchId());
        }
        return recover(plan.switchId());
    }

    public ZookeeperPrefixTopologyJournal.Entry recover(String switchId) {
        var entry = journal.read(switchId);
        if (entry.state() == ZookeeperPrefixTopologyJournal.State.PLANNED && entry.plan().deadlineUnixMs() <= clock.millis()) {
            return cancelPlanned(switchId);
        }
        if (entry.state() != ZookeeperPrefixTopologyJournal.State.SUBMITTED) { return entry; }
        try {
            var reports = inspect(entry.plan());
            if (entry.plan().confirmed(reports)) { return journal.confirmed(entry, reports); }
            var targets = entry.plan().confirmationTargets(reports);
            for (int rank : targets) {
                if (!journal.read(switchId).equals(entry)) { return journal.read(switchId); }
                var endpoint = entry.plan().participants().get(rank).controlEndpoint();
                grpc.submitPrefixTopology(endpoint.ip(), endpoint.port(), EngineRpcService.UpdateSchedulerInfoRequestPB
                        .newBuilder().setSchedulerInfo(entry.plan().confirmationPayload(rank, timeoutMillis)).build(), timeoutMillis);
            }
            if (!targets.isEmpty()) {
                reports = inspect(entry.plan());
                if (entry.plan().confirmed(reports)) { return journal.confirmed(entry, reports); }
            }
        } catch (RuntimeException unavailable) {
            return journal.read(switchId);
        }
        return entry;
    }

    public ZookeeperPrefixTopologyJournal.Entry cancelPlanned(String switchId) {
        var entry = journal.read(switchId);
        var controller = authority.snapshot(entry.plan().namespace());
        if (!authority.isCurrent(controller) || !controller.getNamespace().equals(entry.plan().namespace())
                || Long.compareUnsigned(controller.getTerm(), entry.plan().controllerTerm()) < 0
                || (controller.getTerm() == entry.plan().controllerTerm() && !controller.equals(entry.plan().controller()))) {
            throw new IllegalStateException("topology cancellation requires current non-stale authority");
        }
        if (entry.state() == ZookeeperPrefixTopologyJournal.State.CANCELLED) { return entry; }
        return journal.cancelPlanned(entry);
    }

    private List<EngineRpcService.CacheStatusPB> inspect(PrefixTopologyPlan plan) {
        var reports = new ArrayList<EngineRpcService.CacheStatusPB>();
        for (var participant : plan.participants()) {
            var endpoint = participant.controlEndpoint();
            reports.add(grpc.getCacheStatus(endpoint.ip(), endpoint.port(), EngineRpcService.CacheVersionPB.newBuilder()
                    .setLatestCacheVersion(-1).setNeedCacheKeys(true).build(), timeoutMillis));
        }
        return List.copyOf(reports);
    }
}
