package org.flexlb.sync.prefix;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Endpoint;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public record PrefixTopologyPlan(String switchId, String namespace, String controllerId, long controllerTerm,
                                 int tpSize, int dpSize, long deadlineUnixMs, List<Participant> participants,
                                 boolean residentCache, boolean residentLeaderMerge) {
    public PrefixTopologyPlan(String switchId, String namespace, String controllerId, long controllerTerm,
                              int tpSize, int dpSize, long deadlineUnixMs, List<Participant> participants,
                              boolean residentCache) {
        this(switchId, namespace, controllerId, controllerTerm, tpSize, dpSize, deadlineUnixMs, participants, residentCache, false);
    }
    public PrefixTopologyPlan(String switchId, String namespace, String controllerId, long controllerTerm,
                              int tpSize, int dpSize, long deadlineUnixMs, List<Participant> participants) {
        this(switchId, namespace, controllerId, controllerTerm, tpSize, dpSize, deadlineUnixMs, participants, false);
    }
    public record Participant(int worldRank, Endpoint controlEndpoint, String workerId, String bootId,
                              String epoch, int cpDegree, int dpLane) {
        public Participant {
            Objects.requireNonNull(controlEndpoint);
            bounded(workerId, 512);
            bounded(bootId, 512);
            bounded(epoch, 512);
            if (worldRank < 0 || cpDegree < 1 || dpLane < 0) {
                throw new IllegalArgumentException("invalid topology participant");
            }
        }
    }

    public PrefixTopologyPlan {
        if (!UUID.fromString(switchId).toString().equals(switchId)) {
            throw new IllegalArgumentException("topology plan requires canonical UUID");
        }
        bounded(namespace, 512);
        bounded(controllerId, 128);
        participants = List.copyOf(participants);
        if (controllerTerm == 0 || deadlineUnixMs <= 0 || participants.isEmpty() || participants.size() > 4096
                || tpSize < 1 || dpSize < 1 || participants.size() % tpSize != 0
                || dpSize != participants.size() / tpSize) {
            throw new IllegalArgumentException("invalid bounded topology plan");
        }
        var boots = new HashSet<String>();
        var endpoints = new HashSet<Endpoint>();
        int oldDegree = participants.getFirst().cpDegree();
        if (residentCache && (oldDegree % 2 != 0 || tpSize != oldDegree / 2)) {
            throw new IllegalArgumentException("resident cache requires adjacent CP split");
        }
        if (residentLeaderMerge && (residentCache || tpSize % 2 != 0 || tpSize / 2 != oldDegree)) {
            throw new IllegalArgumentException("resident leader merge requires adjacent CP merge");
        }
        if (participants.size() % oldDegree != 0 || oldDegree == tpSize) {
            throw new IllegalArgumentException("topology plan must change a valid world composition");
        }
        for (int rank = 0; rank < participants.size(); rank++) {
            var participant = participants.get(rank);
            if (participant.worldRank() != rank || !boots.add(participant.bootId())
                    || !endpoints.add(participant.controlEndpoint()) || participant.cpDegree() != oldDegree
                    || participant.dpLane() != rank / oldDegree) {
                throw new IllegalArgumentException("topology participants are incomplete or aliased");
            }
        }
    }

    private static void bounded(String value, int limit) {
        if (value == null || value.isBlank() || value.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > limit) {
            throw new IllegalArgumentException("invalid topology identity");
        }
    }

    public EngineRpcService.PrefixControllerPB controller() {
        return EngineRpcService.PrefixControllerPB.newBuilder().setNamespace(namespace)
                .setControllerId(controllerId).setTerm(controllerTerm).build();
    }

    public String payload(long nowMillis) {
        long remaining = deadlineUnixMs - nowMillis;
        if (nowMillis < 0 || remaining <= 0 || remaining > 300000) {
            throw new IllegalStateException("topology plan is expired or exceeds RPC bound");
        }
        try {
            return new ObjectMapper().writeValueAsString(Map.of(
                    "action", "switch_parallel_topology", "switch_id", switchId,
                    "tp_size", tpSize, "dp_size", dpSize, "timeout_ms", remaining,
                    "resident_cache", residentCache,
                    "resident_leader_merge", residentLeaderMerge,
                    "prefix_controller", Map.of("namespace", namespace, "controller_id", controllerId,
                            "term", new BigInteger(Long.toUnsignedString(controllerTerm))),
                    "expected_participants", participants.stream().map(participant -> Map.of(
                            "world_rank", participant.worldRank(), "boot_id", participant.bootId(),
                            "epoch", participant.epoch())).toList()));
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("cannot encode topology plan", failure);
        }
    }

    public void validateBefore(List<EngineRpcService.CacheStatusPB> reports) {
        if (reports.size() != participants.size()) { throw new IllegalStateException("missing topology participants"); }
        for (int rank = 0; rank < participants.size(); rank++) {
            var participant = participants.get(rank);
            var report = reports.get(rank);
            validateInstance(participant, report);
            var inventory = report.getInventory();
            if (!inventory.getEpoch().equals(participant.epoch()) || inventory.getCpDegree() != participant.cpDegree()
                    || inventory.getDpLane() != participant.dpLane()) {
                throw new IllegalStateException("topology source mapping changed");
            }
        }
    }

    private void validateInstance(Participant participant, EngineRpcService.CacheStatusPB report) {
        var identity = report.getTopologyIdentity();
        if (!report.hasTopologyIdentity() || !report.hasInventory()
                || identity.getWorldRank() != participant.worldRank() || identity.getWorldSize() != participants.size()
                || !identity.getBootId().equals(participant.bootId())
                || !report.getInventory().getNamespace().equals(namespace)
                || !report.getInventory().getWorkerId().equals(participant.workerId())) {
            throw new IllegalStateException("topology participant incarnation changed");
        }
    }

    public boolean confirmed(List<EngineRpcService.CacheStatusPB> reports) {
        return matchesReceipts(reports, java.util.Set.of(EngineRpcService.PrefixTopologyReceiptPB.State.CONFIRMED));
    }

    public List<Integer> confirmationTargets(List<EngineRpcService.CacheStatusPB> reports) {
        if (!matchesReceipts(reports, java.util.Set.of(EngineRpcService.PrefixTopologyReceiptPB.State.CONFIRMED,
                EngineRpcService.PrefixTopologyReceiptPB.State.RELEASED))
                || reports.stream().noneMatch(report -> report.getTopologyReceipt().getState()
                    == EngineRpcService.PrefixTopologyReceiptPB.State.CONFIRMED)) { return List.of(); }
        var targets = new ArrayList<Integer>();
        for (int rank = 0; rank < reports.size(); rank++) {
            if (reports.get(rank).getTopologyReceipt().getState() == EngineRpcService.PrefixTopologyReceiptPB.State.RELEASED) {
                targets.add(rank);
            }
        }
        return List.copyOf(targets);
    }

    public String confirmationPayload(int rank, long timeoutMillis) {
        if (rank < 0 || rank >= participants.size() || timeoutMillis < 1 || timeoutMillis > 300000) {
            throw new IllegalArgumentException("invalid topology confirmation target");
        }
        var participant = participants.get(rank);
        var payload = new HashMap<String, Object>();
        payload.put("action", "switch_parallel_topology");
        payload.put("switch_id", switchId);
        payload.put("tp_size", tpSize);
        payload.put("dp_size", dpSize);
        payload.put("timeout_ms", timeoutMillis);
        payload.put("internal", true);
        payload.put("resident_cache", residentCache);
        payload.put("resident_leader_merge", residentLeaderMerge);
        payload.put("phase", "confirm");
        payload.put("expected_boot_id", participant.bootId());
        payload.put("expected_epoch", participant.epoch());
        payload.put("expected_world_rank", rank);
        payload.put("prefix_controller", Map.of("namespace", namespace, "controller_id", controllerId,
                "term", new BigInteger(Long.toUnsignedString(controllerTerm))));
        try { return new ObjectMapper().writeValueAsString(payload); }
        catch (java.io.IOException failure) { throw new IllegalStateException("cannot encode original topology confirmation", failure); }
    }

    private boolean matchesReceipts(List<EngineRpcService.CacheStatusPB> reports,
                                    java.util.Set<EngineRpcService.PrefixTopologyReceiptPB.State> states) {
        if (reports.size() != participants.size()) { return false; }
        for (int rank = 0; rank < participants.size(); rank++) {
            var participant = participants.get(rank);
            var report = reports.get(rank);
            try {
                validateInstance(participant, report);
                var receipt = report.getTopologyReceipt();
                var expected = new ArrayList<>(List.of(switchId, Integer.toString(tpSize), Integer.toString(dpSize),
                        participant.bootId(), participant.epoch(), Integer.toString(rank)));
                if (residentCache) { expected.add("resident_prefix_split_v1"); }
                if (residentLeaderMerge) { expected.add("resident_leader_merge_v1"); }
                var binding = new ObjectMapper().readValue(receipt.getBinding(), List.class);
                var inventory = report.getInventory();
                if (!report.hasTopologyReceipt() || !receipt.getController().equals(controller())
                        || !states.contains(receipt.getState()) || !binding.equals(expected)
                        || inventory.getEpoch().isBlank() || inventory.getEpoch().equals(participant.epoch())
                        || inventory.getCpDegree() != tpSize || inventory.getDpLane() != rank / tpSize
                        || !inventory.getSnapshotKind().equals("FULL")) {
                    return false;
                }
            } catch (RuntimeException | java.io.IOException invalid) {
                return false;
            }
        }
        return true;
    }
}
