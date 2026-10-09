package org.flexlb.sync.prefix;

import org.flexlb.dao.cache.CacheReportV2;
import org.flexlb.cache.prefix.PrefixReplicaPlanner;

import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public record PrefixTopologyPreservation(List<Required> required, boolean complete, Map<String, CacheReportV2> reports,
                                         List<PrefixReplicaPlanner.Proposal> copies, List<ResidentMerge> residentMerges) {
    public PrefixTopologyPreservation(List<Required> required, boolean complete, Map<String, CacheReportV2> reports,
                                      List<PrefixReplicaPlanner.Proposal> copies) {
        this(required, complete, reports, copies, List.of());
    }
    public PrefixTopologyPreservation(List<Required> required, boolean complete, Map<String, CacheReportV2> reports) {
        this(required, complete, reports, List.of());
    }
    public record Required(String route, PrefixTransferCoordinator.WorkerSnapshot holder, List<Long> path) {
        public Required {
            Objects.requireNonNull(holder);
            path = List.copyOf(path);
            if (route == null || route.isBlank() || route.length() > 1024 || path.isEmpty() || path.size() > 8192
                    || new HashSet<>(path).size() != path.size()) { throw new IllegalArgumentException("invalid preserved prefix path"); }
        }
    }

    public record ResidentMerge(String primaryRoute, PrefixTransferCoordinator.WorkerSnapshot primary,
                                List<Long> retainedKeys, String secondaryRoute,
                                PrefixTransferCoordinator.WorkerSnapshot secondary, List<Long> secondaryKeys) {
        public ResidentMerge {
            Objects.requireNonNull(primary);
            Objects.requireNonNull(secondary);
            retainedKeys = List.copyOf(retainedKeys);
            secondaryKeys = List.copyOf(secondaryKeys);
            if (primaryRoute == null || primaryRoute.isBlank() || primaryRoute.length() > 1024
                    || secondaryRoute == null || secondaryRoute.isBlank() || secondaryRoute.length() > 1024
                    || primaryRoute.equals(secondaryRoute) || retainedKeys.size() > 8192 || secondaryKeys.size() > 8192
                    || new HashSet<>(retainedKeys).size() != retainedKeys.size()
                    || new HashSet<>(secondaryKeys).size() != secondaryKeys.size()) {
                throw new IllegalArgumentException("invalid resident merge inventory");
            }
        }

        public List<Long> retiredKeys() {
            var retained = new HashSet<>(retainedKeys);
            return secondaryKeys.stream().filter(key -> !retained.contains(key)).toList();
        }
    }

    public PrefixTopologyPreservation {
        required = List.copyOf(required);
        reports = Map.copyOf(reports);
        copies = copies == null ? List.of() : List.copyOf(copies);
        residentMerges = residentMerges == null ? List.of() : List.copyOf(residentMerges);
        if (required.size() > 4096 || reports.size() > 4096 || copies.size() > 4096 || residentMerges.size() > 2048
                || (!complete && !reports.isEmpty())) {
            throw new IllegalArgumentException("invalid preservation evidence bound");
        }
    }

    public void validate(PrefixTopologyPlan plan) {
        for (var copy : copies) {
            if (!List.of("intra", "inter").contains(copy.kind()) || copy.source().equals(copy.destination())
                    || copy.destinationPhysicalBytes() < 1 || copy.estimatedWireBytes() < 1
                    || copy.missingBlocks().isEmpty() || !new HashSet<>(copy.path()).containsAll(copy.missingBlocks())
                    || required.stream().noneMatch(item -> item.route().equals(copy.destination()) && item.path().equals(copy.path()))) {
                throw new IllegalArgumentException("copy does not belong to immutable preservation paths");
            }
        }
        var affected = PrefixTopologyAdmission.workers(plan);
        var usedRoutes = new HashSet<String>();
        var holderKeys = new HashMap<String, HashSet<Long>>();
        if (plan.residentLeaderMerge()) {
            if (residentMerges.size() != plan.dpSize() || !required.isEmpty() || !copies.isEmpty()) {
                throw new IllegalArgumentException("resident merge requires complete lane replacement evidence");
            }
            var groups = new HashSet<Integer>();
            for (var merge : residentMerges) {
                int rank = Math.multiplyExact(merge.primary().dpLane(), merge.primary().rankCount());
                if (rank < 0 || rank >= plan.participants().size() || rank % plan.tpSize() != 0 || !groups.add(rank)
                        || !matches(plan.participants().get(rank), merge.primary())
                        || !matches(plan.participants().get(rank + plan.tpSize() / 2), merge.secondary())
                        || !usedRoutes.add(merge.primaryRoute()) || !usedRoutes.add(merge.secondaryRoute())) {
                    throw new IllegalArgumentException("resident merge evidence differs from planned lanes");
                }
                if (complete) {
                    validateMergeReport(plan, merge.primaryRoute(), merge.primary(), merge.retainedKeys());
                    validateMergeReport(plan, merge.secondaryRoute(), merge.secondary(), merge.secondaryKeys());
                }
            }
        } else if (!residentMerges.isEmpty()) {
            throw new IllegalArgumentException("merge replacement evidence requires resident merge mode");
        }
        for (var item : required) {
            if (affected.contains(item.holder().workerId()) && (!plan.residentCache()
                    || plan.participants().stream().noneMatch(participant -> participant.worldRank() % participant.cpDegree() == 0
                        && participant.workerId().equals(item.holder().workerId())
                        && participant.dpLane() == item.holder().dpLane() && participant.epoch().equals(item.holder().epoch())
                        && participant.cpDegree() == item.holder().rankCount()))) {
                throw new IllegalArgumentException("preservation holder lacks resident split plan");
            }
            usedRoutes.add(item.route());
            if (!complete) { continue; }
            var report = reports.get(item.route());
            var holder = item.holder();
            if (report == null || !plan.namespace().equals(report.namespace()) || !holder.workerId().equals(report.workerId())
                    || !holder.epoch().equals(report.epoch()) || holder.dpLane() != report.dpLane()
                    || !holder.layoutFingerprint().equals(report.layoutFingerprint()) || holder.rankCount() != report.cpDegree()
                    || report.version() < holder.version() || report.blockSize() < 1 || report.physicalBlockBytes() < 1
                    || report.rankReadinessGeneration() < 1 || !report.deviceReady() || !"FULL".equals(report.snapshotKind())
                    || !holderKeys.computeIfAbsent(item.route(), route -> new HashSet<>(report.deviceKeys())).containsAll(item.path())) {
                throw new IllegalStateException("preservation requires fresh holder FULL proof");
            }
        }
        if (complete && !usedRoutes.equals(reports.keySet())) { throw new IllegalArgumentException("unrelated preservation proof"); }
    }

    public void validateResidentAfter(PrefixTopologyPlan plan, List<org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB> outcomes) {
        if (!plan.residentCache() && !plan.residentLeaderMerge()) { return; }
        if (!complete || !plan.confirmed(outcomes)) { throw new IllegalStateException("resident preservation lacks confirmed outcome"); }
        for (var merge : residentMerges) {
            int rank = Math.multiplyExact(merge.primary().dpLane(), merge.primary().rankCount());
            var inventory = outcomes.get(rank).getInventory();
            var actual = new HashSet<>(inventory.getDeviceKeysList());
            if (!inventory.getDeviceReady() || inventory.getRankReadinessGeneration() < 1
                    || !actual.equals(new HashSet<>(merge.retainedKeys()))) {
                throw new IllegalStateException("resident merge did not replace secondary cache with confirmed primary inventory");
            }
        }
        var affected = PrefixTopologyAdmission.workers(plan);
        var keysByRank = new HashMap<Integer, HashSet<Long>>();
        for (var item : required) {
            if (!affected.contains(item.holder().workerId())) { continue; }
            int degree = item.holder().rankCount();
            int first = Math.multiplyExact(item.holder().dpLane(), degree);
            for (int rank = first; rank < first + degree; rank += plan.tpSize()) {
                var inventory = outcomes.get(rank).getInventory();
                if (!inventory.getDeviceReady() || inventory.getRankReadinessGeneration() < 1
                        || !"FULL".equals(inventory.getSnapshotKind())
                        || !keysByRank.computeIfAbsent(rank, value -> new HashSet<>(outcomes.get(value).getInventory().getDeviceKeysList()))
                            .containsAll(item.path())) {
                    throw new IllegalStateException("resident split lost a required prefix on a new lane");
                }
            }
        }
    }

    private static boolean matches(PrefixTopologyPlan.Participant participant, PrefixTransferCoordinator.WorkerSnapshot holder) {
        return participant.worldRank() % participant.cpDegree() == 0 && participant.workerId().equals(holder.workerId())
                && participant.dpLane() == holder.dpLane() && participant.epoch().equals(holder.epoch())
                && participant.cpDegree() == holder.rankCount() && participant.controlEndpoint().equals(holder.controlEndpoint());
    }

    private void validateMergeReport(PrefixTopologyPlan plan, String route,
                                     PrefixTransferCoordinator.WorkerSnapshot holder, List<Long> keys) {
        var report = reports.get(route);
        if (report == null || !plan.namespace().equals(report.namespace()) || !holder.workerId().equals(report.workerId())
                || !holder.epoch().equals(report.epoch()) || holder.dpLane() != report.dpLane()
                || !holder.layoutFingerprint().equals(report.layoutFingerprint()) || holder.rankCount() != report.cpDegree()
                || report.version() < holder.version() || !report.deviceReady() || !"FULL".equals(report.snapshotKind())
                || (!keys.isEmpty() && report.rankReadinessGeneration() < 1)
                || !new HashSet<>(report.deviceKeys()).equals(new HashSet<>(keys))) {
            throw new IllegalStateException("resident merge requires fresh complete old lane inventories");
        }
    }
}
