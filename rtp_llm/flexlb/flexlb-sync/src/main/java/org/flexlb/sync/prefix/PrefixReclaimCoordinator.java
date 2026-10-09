package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Outcome;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Result;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerClient;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerSnapshot;

import java.util.List;
import java.util.Objects;
import java.util.UUID;

public final class PrefixReclaimCoordinator {
    public record Attempt(Result result, boolean dispatched) { }
    private final WorkerClient client;
    private final PrefixControllerAuthority authority;
    private final PrefixTransferJournal journal;
    private final org.flexlb.cache.prefix.ConfirmedCacheDirectory directory;

    public PrefixReclaimCoordinator(WorkerClient client, PrefixControllerAuthority authority, PrefixTransferJournal journal) {
        this(client, authority, journal, null);
    }

    public PrefixReclaimCoordinator(WorkerClient client, PrefixControllerAuthority authority, PrefixTransferJournal journal,
                                    org.flexlb.cache.prefix.ConfirmedCacheDirectory directory) {
        this.client = Objects.requireNonNull(client);
        this.authority = Objects.requireNonNull(authority);
        this.journal = Objects.requireNonNull(journal);
        this.directory = directory;
    }

    public Result reclaim(WorkerSnapshot destination, String namespace, List<Long> keys, long timeoutMillis,
                          EngineRpcService.PrefixControllerPB expectedController) {
        return execute(destination, namespace, keys, timeoutMillis, expectedController).result();
    }

    public Attempt execute(WorkerSnapshot destination, String namespace, List<Long> keys, long timeoutMillis,
                           EngineRpcService.PrefixControllerPB expectedController) {
        try (var permit = directory == null ? null : directory.acquireOperations(java.util.Set.of(destination.workerId()))) {
            return executeAdmitted(destination, namespace, keys, timeoutMillis, expectedController);
        }
    }

    private Attempt executeAdmitted(WorkerSnapshot destination, String namespace, List<Long> keys, long timeoutMillis,
                                    EngineRpcService.PrefixControllerPB expectedController) {
        if (timeoutMillis <= 0 || timeoutMillis > 300000) {
            throw new IllegalArgumentException("invalid reclaim deadline");
        }
        var controller = Objects.requireNonNull(authority.snapshot(namespace));
        if (!namespace.equals(controller.getNamespace()) || !controller.equals(expectedController)) {
            throw new IllegalStateException("reclaim controller changed before admission");
        }
        requireLeadership(controller);
        var intent = new PrefixTransferJournal.Intent(UUID.randomUUID().toString(), null, destination, namespace, keys,
                Math.addExact(System.currentTimeMillis(), timeoutMillis), controller.getControllerId(), controller.getTerm(),
                PrefixTransferJournal.Action.RECLAIM);
        var entry = journal.create(intent);
        Result result;
        boolean sent = false;
        try {
            if (directory != null) {
                directory.exclude(intent.requestId(), destination.workerId(), destination.dpLane(), destination.epoch(), destination.version(), intent.keys());
            }
            activate(destination, controller);
            requireLeadership(controller);
            var status = client.inventory(destination.controlEndpoint());
            var report = status.getInventory();
            if (!status.hasInventory() || !namespace.equals(report.getNamespace())
                    || !destination.workerId().equals(report.getWorkerId()) || report.getDpLane() != destination.dpLane()
                    || !destination.epoch().equals(report.getEpoch()) || destination.version() != report.getVersion()
                    || !destination.layoutFingerprint().equals(report.getLayoutFingerprint())
                    || destination.rankCount() != report.getCpDegree() || !report.getDeviceReady()
                    || !"FULL".equals(report.getSnapshotKind()) || report.getRankReadinessGeneration() <= 0
                    || !report.getDeviceKeysList().containsAll(intent.keys())) {
                throw new IllegalStateException("reclaim frozen inventory no longer confirmed");
            }
            requireLeadership(controller);
            if (System.currentTimeMillis() >= intent.deadlineUnixMs()) {
                throw new IllegalStateException("reclaim admission expired");
            }
            sent = true;
            var response = client.reclaim(destination.controlEndpoint(), EngineRpcService.ReclaimPrefixRequestPB.newBuilder()
                    .setNamespace(namespace).setEpoch(destination.epoch()).setExpectedVersion(destination.version())
                    .addAllCacheKeys(intent.keys()).setController(controller).build());
            if (response == null || response.getVersion() < destination.version()
                    || (response.getSuccess() && response.getVersion() == destination.version())) {
                throw new IllegalStateException("invalid reclaim result version");
            }
            result = result(intent, response.getSuccess() ? Outcome.RECLAIMED : Outcome.REJECTED,
                    response.getVersion(), false, response.getSuccess() ? "reclaim acknowledged; awaiting report" : "reclaim rejected");
        } catch (RuntimeException failure) {
            result = result(intent, sent ? Outcome.UNKNOWN : Outcome.REJECTED, destination.version(), sent,
                    sent ? "reclaim outcome unknown; never blindly replay" : "reclaim not dispatched");
        }
        journal.record(entry, result);
        if (directory != null && result.outcome() == Outcome.REJECTED) { directory.include(intent.requestId()); }
        return new Attempt(result, sent);
    }

    Result recoverAttempt(PrefixTransferJournal.Intent intent) {
        if (intent.action() != PrefixTransferJournal.Action.RECLAIM) {
            throw new IllegalArgumentException("reclaim recovery requires reclaim intent");
        }
        var controller = Objects.requireNonNull(authority.snapshot(intent.namespace()));
        if (!controller.getNamespace().equals(intent.namespace())
                || Long.compareUnsigned(controller.getTerm(), intent.controllerTerm()) < 0
                || (controller.getTerm() == intent.controllerTerm() && !controller.equals(intent.controller()))) {
            throw new IllegalStateException("recovery authority cannot fence reclaim controller");
        }
        requireLeadership(controller);
        var destination = intent.destination();
        var status = client.inventory(destination.controlEndpoint());
        var report = status.getInventory();
        if (!status.hasInventory() || !intent.namespace().equals(report.getNamespace())
                || !destination.workerId().equals(report.getWorkerId()) || report.getDpLane() != destination.dpLane()
                || report.getEpoch().isBlank() || !"FULL".equals(report.getSnapshotKind()) || report.getVersion() < 0) {
            throw new IllegalStateException("reclaim recovery inventory identity mismatch");
        }
        requireLeadership(controller);
        if (!destination.epoch().equals(report.getEpoch())) {
            return result(intent, Outcome.SUPERSEDED, destination.version(), false, "Worker epoch replaced; old reclaim fenced");
        }
        if (!destination.layoutFingerprint().equals(report.getLayoutFingerprint())
                || destination.rankCount() != report.getCpDegree() || report.getVersion() < destination.version()) {
            throw new IllegalStateException("reclaim recovery snapshot regressed");
        }
        if (report.getVersion() > destination.version()) {
            return result(intent, Outcome.SUPERSEDED, report.getVersion(), false, "snapshot advanced; do not reclaim a new generation");
        }
        if (Long.compareUnsigned(controller.getTerm(), intent.controllerTerm()) > 0) {
            activate(destination, controller);
            requireLeadership(controller);
            return result(intent, Outcome.SUPERSEDED, destination.version(), false, "new controller fenced old reclaim; no replay");
        }
        return result(intent, Outcome.UNKNOWN, destination.version(), true, "unchanged snapshot cannot exclude late reclaim");
    }

    private void activate(WorkerSnapshot destination, EngineRpcService.PrefixControllerPB controller) {
        requireLeadership(controller);
        var response = client.activateController(destination.controlEndpoint(),
                EngineRpcService.ActivatePrefixControllerRequestPB.newBuilder().setController(controller)
                        .setEpoch(destination.epoch()).build());
        if (response == null || !response.getSuccess() || !response.getEpoch().equals(destination.epoch())
                || !response.getController().equals(controller)) {
            throw new IllegalStateException("reclaim controller activation not confirmed");
        }
    }

    private void requireLeadership(EngineRpcService.PrefixControllerPB controller) {
        if (!authority.isCurrent(controller)) { throw new IllegalStateException("prefix reclaim leadership lost"); }
    }

    private static Result result(PrefixTransferJournal.Intent intent, Outcome outcome, long version, boolean pending, String message) {
        return new Result(outcome, 0, 0, version, pending, message, intent.requestId());
    }
}
