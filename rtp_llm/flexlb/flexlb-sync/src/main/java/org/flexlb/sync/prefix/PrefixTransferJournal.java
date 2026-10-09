package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Result;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerSnapshot;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

public interface PrefixTransferJournal {
    enum Action { COPY, RECLAIM }
    record Footprint(long destinationBytes, long wireBytes, String topologySwitchId) {
        public Footprint(long destinationBytes, long wireBytes) { this(destinationBytes, wireBytes, null); }
        public Footprint {
            if (topologySwitchId != null && !UUID.fromString(topologySwitchId).toString().equals(topologySwitchId)) {
                throw new IllegalArgumentException("invalid parent topology identity");
            }
            if (destinationBytes < 0 || wireBytes < 0) { throw new IllegalArgumentException("negative durable copy footprint"); }
        }
    }

    record Intent(String requestId, WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                  List<Long> keys, long deadlineUnixMs, String controllerId, long controllerTerm, Action action, Footprint footprint) {
        public Intent(String requestId, WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                      List<Long> keys, long deadlineUnixMs, String controllerId, long controllerTerm) {
            this(requestId, source, destination, namespace, keys, deadlineUnixMs, controllerId, controllerTerm, Action.COPY, null);
        }

        public Intent(String requestId, WorkerSnapshot source, WorkerSnapshot destination, String namespace,
                      List<Long> keys, long deadlineUnixMs, String controllerId, long controllerTerm, Action action) {
            this(requestId, source, destination, namespace, keys, deadlineUnixMs, controllerId, controllerTerm, action, null);
        }

        public Intent {
            action = action == null ? Action.COPY : action;
            if (action == Action.COPY) { Objects.requireNonNull(source); }
            if (action == Action.RECLAIM && (source != null || footprint != null)) {
                throw new IllegalArgumentException("reclaim must not invent a source Worker");
            }
            Objects.requireNonNull(destination);
            Objects.requireNonNull(namespace);
            Objects.requireNonNull(controllerId);
            if (!UUID.fromString(requestId).toString().equals(requestId) || namespace.isBlank()
                    || namespace.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 512
                    || controllerId.isBlank()
                    || controllerId.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 128
                    || controllerTerm == 0 || deadlineUnixMs <= 0) {
                throw new IllegalArgumentException("invalid durable transfer identity");
            }
            keys = List.copyOf(keys);
            if (keys.isEmpty() || keys.size() > 8192 || new HashSet<>(keys).size() != keys.size()) {
                throw new IllegalArgumentException("invalid durable transfer path");
            }
        }

        public EngineRpcService.PrefixControllerPB controller() {
            return EngineRpcService.PrefixControllerPB.newBuilder().setNamespace(namespace)
                    .setControllerId(controllerId).setTerm(controllerTerm).build();
        }

        public EngineRpcService.PrefixTransferPB transfer() {
            if (action != Action.COPY) { throw new IllegalStateException("reclaim has no transfer identity"); }
            return EngineRpcService.PrefixTransferPB.newBuilder().setTransferId(requestId)
                    .setSourceWorkerId(source.workerId()).setDestinationWorkerId(destination.workerId())
                    .setSourceEpoch(source.epoch()).setDestinationEpoch(destination.epoch())
                    .setSourceLayout(source.layoutFingerprint()).setDestinationLayout(destination.layoutFingerprint())
                    .build();
        }
    }

    record Entry(Intent intent, Result result, int revision) {
        public Entry {
            Objects.requireNonNull(intent);
            if (revision < 0 || (result != null && !result.requestId().equals(intent.requestId()))) {
                throw new IllegalArgumentException("invalid journal entry");
            }
            if (result != null && (result.outcome() == null || result.message() == null
                    || result.committedVersion() < intent.destination().version()
                    || (result.committed() && (result.operationId() == 0 || result.sourceLeaseId() == 0))
                    || (result.outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN && !result.cleanupPending()))) {
                throw new IllegalArgumentException("invalid durable transfer outcome");
            }
            if (result != null && (intent.action() == Action.RECLAIM
                    ? result.operationId() != 0 || result.sourceLeaseId() != 0 || result.committed()
                        || result.outcome() == PrefixTransferCoordinator.Outcome.ABORTED
                        || (result.outcome() != PrefixTransferCoordinator.Outcome.UNKNOWN && result.cleanupPending())
                        || (result.outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED
                            && (result.committedVersion() <= intent.destination().version() || result.cleanupPending()))
                    : result.outcome() == PrefixTransferCoordinator.Outcome.RECLAIMED
                        || (result.outcome() == PrefixTransferCoordinator.Outcome.SUPERSEDED && result.cleanupPending()))) {
                throw new IllegalArgumentException("outcome does not match durable action kind");
            }
        }
    }

    Entry create(Intent intent);

    Entry record(Entry expected, Result result);

    List<Entry> entries();

    enum Retirement { WAITING, REPORTED_READY, NO_LONGER_RETAINED, TERMINAL }

    default Retirement retire(Entry expected, ConfirmedCacheDirectory directory, long retentionMillis) {
        return Retirement.WAITING;
    }
}
