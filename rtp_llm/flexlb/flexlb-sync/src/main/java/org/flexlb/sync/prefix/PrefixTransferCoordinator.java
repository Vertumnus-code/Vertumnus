package org.flexlb.sync.prefix;

import io.grpc.Status;
import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.engine.grpc.EngineRpcService;

import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.Supplier;

public final class PrefixTransferCoordinator {
    public interface WorkerClient {
        EngineRpcService.ActivatePrefixControllerResponsePB activateController(
                Endpoint endpoint, EngineRpcService.ActivatePrefixControllerRequestPB request);

        EngineRpcService.PreparePrefixExportResponsePB prepareExport(
                Endpoint endpoint, EngineRpcService.PreparePrefixExportRequestPB request);

        EngineRpcService.PreparePrefixImportResponsePB prepareImport(
                Endpoint endpoint, EngineRpcService.PreparePrefixImportRequestPB request);

        EngineRpcService.GetPrefixPrepareResponsePB getPrepare(
                Endpoint endpoint, EngineRpcService.GetPrefixPrepareRequestPB request);

        EngineRpcService.PrefixOperationResponsePB commitImport(
                Endpoint endpoint, EngineRpcService.CommitPrefixImportRequestPB request);

        EngineRpcService.PrefixOperationResponsePB abort(
                Endpoint endpoint, EngineRpcService.AbortPrefixOperationRequestPB request);

        EngineRpcService.PrefixOperationResponsePB releaseExport(
                Endpoint endpoint, EngineRpcService.ReleasePrefixExportRequestPB request);

        EngineRpcService.GetPrefixOperationResponsePB getOperation(
                Endpoint endpoint, EngineRpcService.GetPrefixOperationRequestPB request);

        EngineRpcService.PrefixOperationResponsePB startSend(
                Endpoint endpoint, EngineRpcService.PrefixTransferEndpointRequestPB request, long timeoutMs);

        EngineRpcService.PrefixOperationResponsePB startReceive(
                Endpoint endpoint, EngineRpcService.PrefixTransferEndpointRequestPB request, long timeoutMs);

        default EngineRpcService.PrefixOperationResponsePB reclaim(
                Endpoint endpoint, EngineRpcService.ReclaimPrefixRequestPB request) {
            throw new UnsupportedOperationException("prefix reclaim is not available");
        }

        default EngineRpcService.CacheStatusPB inventory(Endpoint endpoint) {
            throw new UnsupportedOperationException("prefix inventory is not available");
        }
    }

    public record Endpoint(String ip, int port) {
        public Endpoint {
            Objects.requireNonNull(ip);
            if (ip.isBlank() || port <= 0 || port > 65535) {
                throw new IllegalArgumentException("invalid endpoint");
            }
        }
    }

    public record WorkerSnapshot(String workerId, String epoch, long version, String layoutFingerprint,
                                 Endpoint controlEndpoint, Endpoint transferEndpoint, int rankCount, int dpLane) {
        public WorkerSnapshot(String workerId, String epoch, long version, String layoutFingerprint,
                              Endpoint controlEndpoint, Endpoint transferEndpoint, int rankCount) {
            this(workerId, epoch, version, layoutFingerprint, controlEndpoint, transferEndpoint, rankCount, 0);
        }

        public ConfirmedCacheDirectory.Lane lane() {
            return new ConfirmedCacheDirectory.Lane(workerId, dpLane);
        }

        public WorkerSnapshot {
            Objects.requireNonNull(workerId);
            Objects.requireNonNull(epoch);
            Objects.requireNonNull(layoutFingerprint);
            Objects.requireNonNull(controlEndpoint);
            Objects.requireNonNull(transferEndpoint);
            if (workerId.isBlank() || workerId.getBytes(StandardCharsets.UTF_8).length > 512
                    || epoch.isBlank() || epoch.getBytes(StandardCharsets.UTF_8).length > 512
                    || layoutFingerprint.isBlank() || layoutFingerprint.getBytes(StandardCharsets.UTF_8).length > 65536
                    || version < 0 || rankCount <= 0 || dpLane < 0) {
                throw new IllegalArgumentException("invalid worker snapshot");
            }
        }
    }

    public enum Outcome { COMMITTED, ABORTED, REJECTED, UNKNOWN, RECLAIMED, SUPERSEDED }

    public record Result(Outcome outcome, long operationId, long sourceLeaseId, long committedVersion,
                         boolean cleanupPending, String message, String requestId) {
        public Result(Outcome outcome, long operationId, long sourceLeaseId, long committedVersion,
                      boolean cleanupPending, String message) {
            this(outcome, operationId, sourceLeaseId, committedVersion, cleanupPending, message, "");
        }

        public boolean committed() {
            return outcome == Outcome.COMMITTED;
        }
    }

    private final WorkerClient workerClient;
    private final PrefixControllerAuthority authority;
    private final PrefixTransferJournal journal;

    private static final class PrepareSnapshotRejected extends RuntimeException {
        private PrepareSnapshotRejected(RuntimeException cause) {
            super(cause);
        }
    }

    public PrefixTransferCoordinator(WorkerClient workerClient, PrefixControllerAuthority authority) {
        this(workerClient, authority, null);
    }

    public PrefixTransferCoordinator(WorkerClient workerClient, PrefixControllerAuthority authority,
                                     PrefixTransferJournal journal) {
        this.workerClient = Objects.requireNonNull(workerClient);
        this.authority = Objects.requireNonNull(authority);
        this.journal = journal;
    }

    public Result transfer(WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                           List<Long> keys, long timeoutMs) {
        return transfer(source, destination, namespace, keys, timeoutMs, null);
    }

    public Result transfer(WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                           List<Long> keys, long timeoutMs, EngineRpcService.PrefixControllerPB expectedController) {
        return transfer(source, destination, namespace, keys, timeoutMs, expectedController, null);
    }

    public Result transfer(WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                           List<Long> keys, long timeoutMs, EngineRpcService.PrefixControllerPB expectedController,
                           PrefixTransferJournal.Footprint footprint) {
        String requestId = UUID.randomUUID().toString();
        Objects.requireNonNull(source);
        Objects.requireNonNull(destination);
        Objects.requireNonNull(namespace);
        keys = List.copyOf(keys);
        if (namespace.isBlank() || keys.isEmpty() || new HashSet<>(keys).size() != keys.size() || timeoutMs <= 0) {
            throw new IllegalArgumentException("invalid prefix transfer plan");
        }
        long deadline = Math.addExact(System.currentTimeMillis(), Math.min(timeoutMs, 300000));
        EngineRpcService.PrefixControllerPB controller;
        try {
            controller = Objects.requireNonNull(authority.snapshot(namespace));
            if ((expectedController != null && !expectedController.equals(controller))
                    || !controller.getNamespace().equals(namespace) || controller.getNamespaceBytes().size() > 512
                    || controller.getControllerId().isEmpty() || controller.getControllerIdBytes().size() > 128
                    || controller.getTerm() == 0) {
                throw new IllegalStateException("invalid controller identity");
            }
        } catch (RuntimeException exception) {
            return new Result(Outcome.REJECTED, 0, 0, destination.version(), false,
                    "controller barrier not confirmed", requestId);
        }
        var intent = new PrefixTransferJournal.Intent(requestId, source, destination, namespace, keys, deadline,
                controller.getControllerId(), controller.getTerm(), PrefixTransferJournal.Action.COPY, footprint);
        var entry = journal == null ? null : journal.create(intent);
        Result result = transferAttempt(intent, timeoutMs);
        result = new Result(result.outcome(), result.operationId(), result.sourceLeaseId(), result.committedVersion(),
                result.cleanupPending(), result.message(), requestId);
        if (journal != null) {
            journal.record(entry, result);
        }
        return result;
    }

    private Result transferAttempt(PrefixTransferJournal.Intent intent, long timeoutMs) {
        var source = intent.source();
        var destination = intent.destination();
        var namespace = intent.namespace();
        var keys = intent.keys();
        var requestId = intent.requestId();
        var controller = intent.controller();
        long deadline = intent.deadlineUnixMs();
        try {
            activate(source, controller);
            activate(destination, controller);
            requireLeadership(controller);
            if (System.currentTimeMillis() >= deadline) {
                throw new IllegalStateException("activation deadline exhausted");
            }
        } catch (RuntimeException exception) {
            return new Result(Outcome.REJECTED, 0, 0, destination.version(), false, "controller barrier not confirmed");
        }
        var transfer = intent.transfer();
        var exportRequest = EngineRpcService.PreparePrefixExportRequestPB.newBuilder()
                .setNamespace(namespace).setEpoch(source.epoch()).setVersion(source.version()).addAllCacheKeys(keys)
                .setTransfer(transfer)
                .setController(controller)
                .setRequestId(requestId + "-export").setDeadlineUnixMs(deadline).build();
        EngineRpcService.PreparePrefixExportResponsePB export;
        try {
            export = prepareWithRecovery(source.controlEndpoint(),
                    prepareQuery(source, namespace, exportRequest.getRequestId(), EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_EXPORT, controller),
                    deadline, () -> workerClient.prepareExport(source.controlEndpoint(), exportRequest), response -> {
                        if (!response.hasExportResult()) {
                            throw new IllegalStateException("missing export prepare result");
                        }
                        return response.getExportResult();
                    });
        } catch (PrepareSnapshotRejected exception) {
            return new Result(Outcome.REJECTED, 0, 0, destination.version(), false, "source prepare snapshot rejected");
        } catch (RuntimeException exception) {
            return new Result(Outcome.UNKNOWN, 0, 0, destination.version(), true, "source prepare result unknown");
        }
        if (!export.getSupportsMissingOnly() || !export.getTransfer().equals(transfer)
                || !export.getController().equals(controller)
                || export.getLeaseId() == 0 || !export.getEpoch().equals(source.epoch())
                || export.getVersion() != source.version() || !export.getCacheKeysList().equals(keys)
                || !export.getRequestId().equals(exportRequest.getRequestId()) || export.getDeadlineUnixMs() != deadline) {
            return new Result(Outcome.REJECTED, 0, export.getLeaseId(), destination.version(), true,
                    "source prepare identity mismatch");
        }
        if (!export.getLeaseActive()) {
            boolean released = releaseExport(source, export.getLeaseId(), controller, transfer);
            return new Result(Outcome.ABORTED, 0, export.getLeaseId(), destination.version(), !released,
                    "source lease is inactive");
        }
        Result result;
        boolean released;
        try {
            result = transferPrepared(source, destination, namespace, keys, timeoutMs, export, requestId, deadline);
        } finally {
            released = releaseExport(source, export.getLeaseId(), controller, transfer);
        }
        return new Result(result.outcome(), result.operationId(), export.getLeaseId(), result.committedVersion(),
                result.cleanupPending() || !released, result.message());
    }

    private Result transferPrepared(WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                                    List<Long> keys, long timeoutMs,
                                    EngineRpcService.PreparePrefixExportResponsePB export, String requestId, long deadline) {
        var importRequest = EngineRpcService.PreparePrefixImportRequestPB.newBuilder()
                .setController(export.getController()).setTransfer(export.getTransfer())
                .setNamespace(namespace).setEpoch(destination.epoch()).setExpectedVersion(destination.version())
                .setRankCount(destination.rankCount()).setSourceLeaseId(export.getLeaseId()).addAllCacheKeys(keys)
                .setRequestId(requestId + "-import").setDeadlineUnixMs(deadline).build();
        EngineRpcService.PreparePrefixImportResponsePB operation;
        try {
            operation = prepareWithRecovery(destination.controlEndpoint(),
                    prepareQuery(destination, namespace, importRequest.getRequestId(), EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_IMPORT, export.getController()),
                    deadline, () -> workerClient.prepareImport(destination.controlEndpoint(), importRequest), response -> {
                        if (!response.hasImportResult()) {
                            throw new IllegalStateException("missing import prepare result");
                        }
                        return response.getImportResult();
                    });
        } catch (PrepareSnapshotRejected exception) {
            return new Result(Outcome.ABORTED, 0, export.getLeaseId(), destination.version(), false,
                    "destination prepare snapshot rejected");
        } catch (RuntimeException exception) {
            return new Result(Outcome.UNKNOWN, 0, export.getLeaseId(), destination.version(), true, "destination prepare result unknown");
        }
        long id = operation.getOperationId();
        if (!operation.getTransfer().equals(export.getTransfer())
                || !operation.getHasCopyPlan() || !validMissingIndices(operation.getMissingIndicesList(), keys.size())
                || !operation.getController().equals(export.getController())
                || id == 0 || !operation.getEpoch().equals(destination.epoch())
                || operation.getExpectedVersion() != destination.version()
                || operation.getSourceLeaseId() != export.getLeaseId() || !operation.getCacheKeysList().equals(keys)
                || !operation.getRequestId().equals(importRequest.getRequestId()) || operation.getDeadlineUnixMs() != deadline
                || operation.getRankCount() != destination.rankCount()) {
            return new Result(Outcome.REJECTED, id, export.getLeaseId(), destination.version(), true,
                    "destination prepare identity mismatch");
        }
        if (operation.getState() == 3 && operation.getCommittedVersion() >= destination.version()) {
            return new Result(Outcome.COMMITTED, id, export.getLeaseId(), operation.getCommittedVersion(), false,
                    "commit confirmed by prepare query");
        }
        if (operation.getState() == 4) {
            return new Result(Outcome.ABORTED, id, export.getLeaseId(), destination.version(), false, "import is aborted");
        }
        if (operation.getState() != 0) {
            return resolve(destination, operation, "prepare returned an already active operation");
        }
        String rejectionReason = "commit rejected";
        try {
            long receiveBudget = Math.min(timeoutMs, Math.min(export.getLeaseRemainingMs(),
                    Math.min(operation.getLeaseRemainingMs(), deadline - System.currentTimeMillis())));
            if (receiveBudget <= 0 || export.getLeaseRemainingMs() <= 0) {
                return resolve(destination, operation, "prepare deadline exhausted");
            }
            var receive = EngineRpcService.PrefixTransferEndpointRequestPB.newBuilder()
                    .setHasCopyPlan(true).addAllMissingIndices(operation.getMissingIndicesList())
                    .setTransfer(operation.getTransfer()).setController(operation.getController())
                    .setOperationId(id).setLeaseId(export.getLeaseId()).setEpoch(destination.epoch())
                    .setPeerIp(source.transferEndpoint().ip()).setPeerPort(source.transferEndpoint().port())
                    .setTimeoutMs(receiveBudget).build();
            requireLeadership(operation.getController());
            if (!workerClient.startReceive(destination.controlEndpoint(), receive, receiveBudget).getSuccess()) {
                return resolve(destination, operation, "receive registration failed");
            }
            long sendBudget = Math.min(receiveBudget, deadline - System.currentTimeMillis());
            if (sendBudget <= 0) {
                return resolve(destination, operation, "send deadline exhausted");
            }
            var send = receive.toBuilder().setEpoch(source.epoch()).setTimeoutMs(sendBudget)
                    .setPeerIp(destination.transferEndpoint().ip()).setPeerPort(destination.transferEndpoint().port())
                    .build();
            requireLeadership(operation.getController());
            if (!workerClient.startSend(source.controlEndpoint(), send, sendBudget).getSuccess()) {
                return resolve(destination, operation, "send failed");
            }
            requireLeadership(operation.getController());
            var commit = workerClient.commitImport(destination.controlEndpoint(),
                    EngineRpcService.CommitPrefixImportRequestPB.newBuilder()
                            .setTransfer(operation.getTransfer()).setController(operation.getController())
                            .setOperationId(id).setLeaseId(export.getLeaseId())
                            .setEpoch(destination.epoch()).setVersion(destination.version()).build());
            if (commit.getSuccess() && commit.getVersion() >= destination.version()) {
                return new Result(Outcome.COMMITTED, id, export.getLeaseId(), commit.getVersion(), false, "committed");
            }
            rejectionReason += ": expected_version=" + destination.version() + ", response_version="
                    + commit.getVersion() + ", message=" + commit.getMessage();
        } catch (RuntimeException exception) {
            return resolve(destination, operation, "transfer or commit result unknown: "
                    + exception.getClass().getSimpleName() + ": " + exception.getMessage());
        }
        return resolve(destination, operation, rejectionReason);
    }

    public Result recover(PrefixTransferJournal.Entry entry) {
        if (journal == null || !journal.entries().contains(entry)) {
            throw new IllegalStateException("recovery requires the current durable entry");
        }
        if (entry.result() != null && !entry.result().cleanupPending()
                && entry.result().outcome() != Outcome.UNKNOWN) {
            return entry.result();
        }
        Result result;
        try {
            result = entry.intent().action() == PrefixTransferJournal.Action.RECLAIM
                    ? new PrefixReclaimCoordinator(workerClient, authority, journal).recoverAttempt(entry.intent())
                    : recoverCopy(entry);
        } catch (RuntimeException failure) {
            var previous = entry.result();
            result = new Result(previous == null ? Outcome.UNKNOWN : previous.outcome(),
                    previous == null ? 0 : previous.operationId(), previous == null ? 0 : previous.sourceLeaseId(),
                    previous == null ? entry.intent().destination().version() : previous.committedVersion(),
                    true, "durable recovery remains unconfirmed", entry.intent().requestId());
        }
        if (entry.result() != null && entry.result().committed() && !result.committed()) {
            result = new Result(Outcome.COMMITTED, entry.result().operationId(), entry.result().sourceLeaseId(),
                    entry.result().committedVersion(), true, "committed transfer cleanup remains unconfirmed",
                    entry.intent().requestId());
        }
        journal.record(entry, result);
        return result;
    }

    private Result recoverCopy(PrefixTransferJournal.Entry entry) {
        try {
            return recoverAttempt(entry);
        } catch (RuntimeException failure) {
            return recoverExpiredBindings(entry);
        }
    }

    private EngineRpcService.PrefixControllerPB recoveryController(PrefixTransferJournal.Intent intent) {
        var controller = Objects.requireNonNull(authority.snapshot(intent.namespace()));
        if (!controller.getNamespace().equals(intent.namespace())
                || Long.compareUnsigned(controller.getTerm(), intent.controllerTerm()) < 0
                || (controller.getTerm() == intent.controllerTerm() && !controller.equals(intent.controller()))) {
            throw new IllegalStateException("recovery authority cannot fence the recorded controller");
        }
        requireLeadership(controller);
        return controller;
    }

    private WorkerSnapshot currentRecoveryWorker(WorkerSnapshot recorded, String namespace,
                                                 EngineRpcService.PrefixControllerPB controller) {
        requireLeadership(controller);
        var status = workerClient.inventory(recorded.controlEndpoint());
        requireLeadership(controller);
        if (status == null || !status.hasInventory()) {
            throw new IllegalStateException("recovery inventory unavailable");
        }
        var report = status.getInventory();
        if (!report.getNamespace().equals(namespace) || !report.getWorkerId().equals(recorded.workerId())
                || report.getDpLane() != recorded.dpLane() || !report.getSnapshotKind().equals("FULL")
                || !report.getDeviceReady() || !report.getPhysicalMemoryStatsValid()
                || (report.getEpoch().equals(recorded.epoch())
                    && (report.getVersion() < recorded.version() || report.getCpDegree() != recorded.rankCount()
                        || !report.getLayoutFingerprint().equals(recorded.layoutFingerprint())))) {
            throw new IllegalStateException("recovery inventory identity mismatch");
        }
        return new WorkerSnapshot(recorded.workerId(), report.getEpoch(), report.getVersion(),
                report.getLayoutFingerprint(), recorded.controlEndpoint(), recorded.transferEndpoint(),
                report.getCpDegree(), recorded.dpLane());
    }

    private void confirmRecoveryWorker(WorkerSnapshot observed, String namespace,
                                       EngineRpcService.PrefixControllerPB controller) {
        var current = currentRecoveryWorker(observed, namespace, controller);
        if (!current.epoch().equals(observed.epoch())) {
            throw new IllegalStateException("worker changed during recovery proof");
        }
    }

    private EngineRpcService.GetPrefixPrepareResponsePB expiredPrepare(WorkerSnapshot worker, String namespace,
            String requestId, EngineRpcService.PrefixPrepareKindPB kind, EngineRpcService.PrefixControllerPB controller) {
        requireLeadership(controller);
        try {
            var response = workerClient.getPrepare(worker.controlEndpoint(),
                    prepareQuery(worker, namespace, requestId, kind, controller));
            requireLeadership(controller);
            return Objects.requireNonNull(response);
        } catch (RuntimeException failure) {
            requireLeadership(controller);
            if (Status.fromThrowable(failure).getCode() == Status.Code.NOT_FOUND) {
                return null;
            }
            throw failure;
        }
    }

    private Result recoverExpiredBindings(PrefixTransferJournal.Entry entry) {
        var intent = entry.intent();
        var controller = recoveryController(intent);
        if (System.currentTimeMillis() <= intent.deadlineUnixMs()) {
            throw new IllegalStateException("old prepare deadline has not elapsed");
        }
        var previous = entry.result();
        boolean sourceOnly = previous != null && previous.outcome() == Outcome.ABORTED
                && previous.operationId() == 0 && previous.sourceLeaseId() != 0;
        var source = currentRecoveryWorker(intent.source(), intent.namespace(), controller);
        var destination = sourceOnly ? null : currentRecoveryWorker(intent.destination(), intent.namespace(), controller);
        activate(source, controller);
        if (destination != null) {
            activate(destination, controller);
        }
        requireLeadership(controller);
        long leaseId = previous == null ? 0 : previous.sourceLeaseId();
        long operationId = previous == null ? 0 : previous.operationId();
        long version = previous == null ? intent.destination().version() : previous.committedVersion();
        Outcome outcome = sourceOnly ? Outcome.ABORTED : Outcome.SUPERSEDED;
        boolean currentOwner = controller.equals(intent.controller());
        if (source.epoch().equals(intent.source().epoch())) {
            var response = expiredPrepare(source, intent.namespace(), intent.requestId() + "-export",
                    EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_EXPORT, controller);
            if (response == null && currentOwner) {
                throw new IllegalStateException("missing prepare requires a newer controller fence");
            }
            if (response != null) {
                if (!response.hasExportResult()) {
                    throw new IllegalStateException("durable export record not available");
                }
                var export = response.getExportResult();
                validateRecoveredExport(intent, export);
                if (leaseId != 0 && leaseId != export.getLeaseId()) {
                    throw new IllegalStateException("durable export lease mismatch");
                }
                leaseId = export.getLeaseId();
                if (currentOwner ? !releaseExport(source, leaseId, controller, intent.transfer()) : export.getLeaseActive()) {
                    throw new IllegalStateException("old source lease cleanup unconfirmed");
                }
            }
        }
        if (destination != null && destination.epoch().equals(intent.destination().epoch())) {
            var response = expiredPrepare(destination, intent.namespace(), intent.requestId() + "-import",
                    EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_IMPORT, controller);
            if (response == null && currentOwner) {
                throw new IllegalStateException("missing prepare requires a newer controller fence");
            }
            if (response != null) {
                if (!response.hasImportResult()) {
                    throw new IllegalStateException("durable import record not available");
                }
                var operation = response.getImportResult();
                validateRecoveredImport(intent, operation, leaseId == 0 ? operation.getSourceLeaseId() : leaseId);
                if ((operationId != 0 && operationId != operation.getOperationId())
                        || (previous != null && previous.outcome() == Outcome.ABORTED && operation.getState() == 3)
                        || (previous != null && previous.committed()
                            && (operation.getState() != 3 || operation.getCommittedVersion() != version))) {
                    throw new IllegalStateException("durable import history mismatch");
                }
                operationId = operation.getOperationId();
                leaseId = operation.getSourceLeaseId();
                if (operation.getState() == 3 && operation.getCommittedVersion() >= intent.destination().version()) {
                    outcome = Outcome.COMMITTED;
                    version = operation.getCommittedVersion();
                } else if (operation.getState() < 0 || operation.getState() > 4 || operation.getState() == 3
                        || (currentOwner ? !abort(destination, operationId, controller, intent.transfer())
                            : operation.getState() != 4)) {
                    throw new IllegalStateException("old destination cleanup unconfirmed");
                }
            }
        }
        confirmRecoveryWorker(source, intent.namespace(), controller);
        if (destination != null) {
            confirmRecoveryWorker(destination, intent.namespace(), controller);
        }
        requireLeadership(controller);
        if (previous != null && (previous.committed() || previous.outcome() == Outcome.ABORTED)) {
            return new Result(previous.outcome(), previous.operationId(), previous.sourceLeaseId(),
                    previous.committedVersion(), false, "resolved transfer obsolete bindings cleaned", intent.requestId());
        }
        return new Result(outcome, operationId, leaseId, version, false,
                "expired transfer bindings cleaned without replay; no publication inferred", intent.requestId());
    }

    private void validateRecoveredExport(PrefixTransferJournal.Intent intent,
                                         EngineRpcService.PreparePrefixExportResponsePB export) {
        var source = intent.source();
        if (!export.getController().equals(intent.controller()) || !export.getTransfer().equals(intent.transfer())
                || export.getLeaseId() == 0 || !export.getEpoch().equals(source.epoch())
                || export.getVersion() != source.version() || !export.getCacheKeysList().equals(intent.keys())
                || !export.getRequestId().equals(intent.requestId() + "-export")
                || export.getDeadlineUnixMs() != intent.deadlineUnixMs() || !export.getSupportsMissingOnly()) {
            throw new IllegalStateException("durable export recovery identity mismatch");
        }
    }

    private void validateRecoveredImport(PrefixTransferJournal.Intent intent,
                                         EngineRpcService.PreparePrefixImportResponsePB operation, long sourceLeaseId) {
        var destination = intent.destination();
        if (!operation.getController().equals(intent.controller()) || !operation.getTransfer().equals(intent.transfer())
                || operation.getOperationId() == 0 || !operation.getEpoch().equals(destination.epoch())
                || operation.getExpectedVersion() != destination.version() || operation.getSourceLeaseId() == 0
                || operation.getSourceLeaseId() != sourceLeaseId || !operation.getCacheKeysList().equals(intent.keys())
                || !operation.getRequestId().equals(intent.requestId() + "-import")
                || operation.getDeadlineUnixMs() != intent.deadlineUnixMs()
                || operation.getRankCount() != destination.rankCount() || !operation.getHasCopyPlan()
                || !validMissingIndices(operation.getMissingIndicesList(), intent.keys().size())) {
            throw new IllegalStateException("durable transfer recovery identity mismatch");
        }
    }

    private Result recoverAttempt(PrefixTransferJournal.Entry entry) {
        var intent = entry.intent();
        var source = intent.source();
        var destination = intent.destination();
        boolean sourceOnly = entry.result() != null && entry.result().outcome() == Outcome.ABORTED
                && entry.result().operationId() == 0 && entry.result().sourceLeaseId() != 0;
        var controller = recoveryController(intent);
        activate(source, controller);
        if (!sourceOnly) {
            activate(destination, controller);
        }
        requireLeadership(controller);
        var exported = workerClient.getPrepare(source.controlEndpoint(), prepareQuery(source, intent.namespace(),
                intent.requestId() + "-export", EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_EXPORT, controller));
        if (!exported.hasExportResult()) {
            throw new IllegalStateException("durable export record not available");
        }
        var export = exported.getExportResult();
        validateRecoveredExport(intent, export);
        boolean currentOwner = controller.equals(intent.controller());
        if (sourceOnly) {
            if (export.getLeaseId() != entry.result().sourceLeaseId()) {
                throw new IllegalStateException("durable source-only cleanup lease mismatch");
            }
            boolean sourceClean = currentOwner ? releaseExport(source, export.getLeaseId(), controller, intent.transfer())
                    : !export.getLeaseActive();
            requireLeadership(controller);
            return new Result(Outcome.ABORTED, 0, export.getLeaseId(), destination.version(), !sourceClean,
                    "source-only cleanup recovered without reserving destination", intent.requestId());
        }
        requireLeadership(controller);
        var imported = workerClient.getPrepare(destination.controlEndpoint(), prepareQuery(destination, intent.namespace(),
                intent.requestId() + "-import", EngineRpcService.PrefixPrepareKindPB.PREFIX_PREPARE_IMPORT, controller));
        if (!imported.hasImportResult()) {
            throw new IllegalStateException("durable import record not available");
        }
        var operation = imported.getImportResult();
        validateRecoveredImport(intent, operation, export.getLeaseId());
        Outcome outcome = Outcome.UNKNOWN;
        long version = destination.version();
        boolean destinationClean = false;
        if (operation.getState() == 3 && operation.getCommittedVersion() >= destination.version()) {
            outcome = Outcome.COMMITTED;
            version = operation.getCommittedVersion();
            destinationClean = true;
        } else if (operation.getState() == 4) {
            outcome = Outcome.ABORTED;
            destinationClean = !currentOwner || abort(destination, operation.getOperationId(), controller, intent.transfer());
        } else if (operation.getState() >= 0 && operation.getState() <= 2
                && currentOwner && abort(destination, operation.getOperationId(), controller, intent.transfer())) {
            outcome = Outcome.ABORTED;
            destinationClean = true;
        }
        boolean sourceClean = currentOwner ? releaseExport(source, export.getLeaseId(), controller, intent.transfer())
                : !export.getLeaseActive();
        requireLeadership(controller);
        return new Result(outcome, operation.getOperationId(), export.getLeaseId(), version,
                !sourceClean || !destinationClean, "durable transfer recovered without replay", intent.requestId());
    }

    private EngineRpcService.GetPrefixPrepareRequestPB prepareQuery(WorkerSnapshot worker, String namespace,
            String requestId, EngineRpcService.PrefixPrepareKindPB kind, EngineRpcService.PrefixControllerPB controller) {
        return EngineRpcService.GetPrefixPrepareRequestPB.newBuilder().setNamespace(namespace).setEpoch(worker.epoch())
                .setRequestId(requestId).setKind(kind).setController(controller).build();
    }

    private <Response> Response prepareWithRecovery(Endpoint endpoint, EngineRpcService.GetPrefixPrepareRequestPB query,
            long deadline, Supplier<Response> prepare, Function<EngineRpcService.GetPrefixPrepareResponsePB, Response> decode) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                requireLeadership(query.getController());
                return prepare.get();
            } catch (RuntimeException failure) {
                requireLeadership(query.getController());
                String snapshotRejection = switch (query.getKind()) {
                    case PREFIX_PREPARE_EXPORT -> "prefix export snapshot is not available";
                    case PREFIX_PREPARE_IMPORT -> "prefix import snapshot is not available";
                    default -> "";
                };
                var status = Status.fromThrowable(failure);
                if (attempt == 0 && status.getCode() == Status.Code.FAILED_PRECONDITION
                        && !snapshotRejection.isEmpty() && snapshotRejection.equals(status.getDescription())) {
                    throw new PrepareSnapshotRejected(failure);
                }
                try {
                    requireLeadership(query.getController());
                    return decode.apply(workerClient.getPrepare(endpoint, query));
                } catch (RuntimeException queryFailure) {
                    if (attempt == 0 && Status.fromThrowable(queryFailure).getCode() == Status.Code.NOT_FOUND
                            && System.currentTimeMillis() < deadline) {
                        continue;
                    }
                    throw queryFailure;
                }
            }
        }
        throw new IllegalStateException("prepare recovery exhausted");
    }

    private Result resolve(WorkerSnapshot destination, EngineRpcService.PreparePrefixImportResponsePB operation,
                           String message) {
        long id = operation.getOperationId();
        try {
            requireLeadership(operation.getController());
            var status = workerClient.getOperation(destination.controlEndpoint(),
                    EngineRpcService.GetPrefixOperationRequestPB.newBuilder()
                            .setController(operation.getController())
                            .setOperationId(id).setEpoch(destination.epoch()).build());
            if (!matches(status, operation)) {
                return new Result(Outcome.UNKNOWN, id, operation.getSourceLeaseId(), destination.version(), true,
                        "operation identity mismatch");
            }
            if (status.getState() == 3 && status.getCommittedVersion() >= destination.version()) {
                return new Result(Outcome.COMMITTED, id, operation.getSourceLeaseId(), status.getCommittedVersion(), false,
                        "commit confirmed by query");
            }
            if (status.getState() == 4 || (status.getState() >= 0 && status.getState() <= 2
                    && abort(destination, id, operation.getController(), operation.getTransfer()))) {
                return new Result(Outcome.ABORTED, id, operation.getSourceLeaseId(), destination.version(), false, message);
            }
        } catch (RuntimeException ignored) {
        }
        return new Result(Outcome.UNKNOWN, id, operation.getSourceLeaseId(), destination.version(), true, message);
    }

    private static boolean validMissingIndices(List<Integer> indices, int keyCount) {
        int previous = -1;
        for (int index : indices) {
            if (index <= previous || index >= keyCount) {
                return false;
            }
            previous = index;
        }
        return true;
    }

    private boolean matches(EngineRpcService.GetPrefixOperationResponsePB status,
                            EngineRpcService.PreparePrefixImportResponsePB operation) {
        return status.getOperationId() == operation.getOperationId()
                && status.getHasCopyPlan() && status.getMissingIndicesList().equals(operation.getMissingIndicesList())
                && status.getController().equals(operation.getController())
                && status.getTransfer().equals(operation.getTransfer())
                && status.getEpoch().equals(operation.getEpoch())
                && status.getExpectedVersion() == operation.getExpectedVersion()
                && status.getSourceLeaseId() == operation.getSourceLeaseId()
                && status.getCacheKeysList().equals(operation.getCacheKeysList());
    }

    private boolean abort(WorkerSnapshot worker, long operationId, EngineRpcService.PrefixControllerPB controller,
                          EngineRpcService.PrefixTransferPB transfer) {
        try {
            requireLeadership(controller);
            return workerClient.abort(worker.controlEndpoint(), EngineRpcService.AbortPrefixOperationRequestPB.newBuilder()
                    .setTransfer(transfer).setController(controller)
                    .setOperationId(operationId).setEpoch(worker.epoch()).build()).getSuccess();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    private boolean releaseExport(WorkerSnapshot worker, long leaseId, EngineRpcService.PrefixControllerPB controller,
                                  EngineRpcService.PrefixTransferPB transfer) {
        try {
            requireLeadership(controller);
            return workerClient.releaseExport(worker.controlEndpoint(), EngineRpcService.ReleasePrefixExportRequestPB.newBuilder()
                    .setTransfer(transfer).setController(controller)
                    .setLeaseId(leaseId).setEpoch(worker.epoch()).build()).getSuccess();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    private void requireLeadership(EngineRpcService.PrefixControllerPB controller) {
        if (!authority.isCurrent(controller)) {
            throw new IllegalStateException("prefix controller leadership lost");
        }
    }

    private void activate(WorkerSnapshot worker, EngineRpcService.PrefixControllerPB controller) {
        var request = EngineRpcService.ActivatePrefixControllerRequestPB.newBuilder()
                .setController(controller).setEpoch(worker.epoch()).build();
        for (int attempt = 0; attempt < 2; ++attempt) {
            requireLeadership(controller);
            EngineRpcService.ActivatePrefixControllerResponsePB response;
            try {
                response = workerClient.activateController(worker.controlEndpoint(), request);
            } catch (RuntimeException failure) {
                if (attempt == 0) {
                    continue;
                }
                throw failure;
            }
            if (!response.getSuccess() || !response.getController().equals(controller)
                    || !response.getEpoch().equals(worker.epoch())) {
                throw new IllegalStateException("prefix controller activation identity mismatch");
            }
            return;
        }
        throw new IllegalStateException("prefix controller activation not confirmed");
    }
}
