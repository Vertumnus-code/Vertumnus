package org.flexlb.sync.prefix;

import org.flexlb.dao.cache.CacheReportV2;
import org.flexlb.engine.grpc.EngineRpcService;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class PrefixTopologyAdmission {
    private PrefixTopologyAdmission() { }

    public static Set<String> workers(PrefixTopologyPlan plan) {
        return plan.participants().stream().map(PrefixTopologyPlan.Participant::workerId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    public static void validateContinuation(List<EngineRpcService.CacheStatusPB> recorded,
                                            List<EngineRpcService.CacheStatusPB> current) {
        if (recorded.size() != current.size()) { throw new IllegalStateException("topology evidence rank count changed"); }
        for (int rank = 0; rank < current.size(); rank++) {
            var before = recorded.get(rank).getInventory();
            var after = current.get(rank).getInventory();
            if (!before.getEpoch().equals(after.getEpoch()) || after.getVersion() < before.getVersion()
                    || after.getRankReadinessGeneration() < before.getRankReadinessGeneration()
                    || !before.getLayoutFingerprint().equals(after.getLayoutFingerprint())
                    || before.getBlockSize() != after.getBlockSize() || before.getPhysicalBlockBytes() != after.getPhysicalBlockBytes()) {
                throw new IllegalStateException("confirmed topology inventory generation changed or regressed");
            }
        }
    }

    public static Map<String, CacheReportV2> validate(PrefixTopologyPlan plan,
                                                    List<PrefixWorkerRegistry.Registration> registrations,
                                                    List<EngineRpcService.CacheStatusPB> reports) {
        if (!plan.confirmed(reports) || registrations.size() != plan.dpSize()) {
            throw new IllegalStateException("topology admission requires all-rank confirmation and every new lane");
        }
        new PrefixWorkerRegistry(registrations);
        var lanes = new HashSet<Integer>();
        var bases = new HashSet<String>();
        var result = new HashMap<String, CacheReportV2>();
        for (var registration : registrations) {
            int lane = registration.dpLane();
            if (lane >= plan.dpSize() || !lanes.add(lane)) { throw new IllegalArgumentException("invalid admitted lane"); }
            String base = org.flexlb.cache.prefix.ConfirmedCacheDirectory.routeBase(registration.route());
            String expectedRoute = plan.dpSize() == 1 ? base : base + "#dp" + lane;
            if (!registration.route().equals(expectedRoute)) { throw new IllegalArgumentException("non-canonical frontend lane route"); }
            bases.add(base);
            if (bases.size() > 1) { throw new IllegalArgumentException("topology admission requires one frontend per world"); }
            int leader = lane * plan.tpSize();
            var participant = plan.participants().get(leader);
            var inventory = reports.get(leader).getInventory();
            if (!registration.workerId().equals(participant.workerId())
                    || !registration.controlEndpoint().equals(participant.controlEndpoint())
                    || !inventory.getDeviceReady() || inventory.getLayoutFingerprint().isBlank()
                    || inventory.getBlockSize() < 1 || inventory.getPhysicalBlockBytes() < 1
                    || inventory.getPhysicalFreeBytes() < 0 || inventory.getVersion() < 0
                    || inventory.getRankReadinessGeneration() < 0
                    || (inventory.getDeviceKeysCount() > 0 && inventory.getRankReadinessGeneration() == 0)) {
                throw new IllegalStateException("new lane lacks authoritative ready FULL inventory");
            }
            result.put(registration.route(), new CacheReportV2(inventory.getNamespace(), inventory.getWorkerId(), lane,
                    inventory.getEpoch(), inventory.getLayoutFingerprint(), "FULL", inventory.getVersion(),
                    inventory.getBaseVersion(), inventory.getRankReadinessGeneration(), true, inventory.getCpDegree(),
                    inventory.getBlockSize(), inventory.getPhysicalFreeBytes(), inventory.getDeviceKeysList(), List.of(),
                    inventory.getDroppedReuseEvents(), inventory.getPhysicalBlockBytes(), inventory.getEvictableBytes(),
                    inventory.getActiveBytes(), inventory.getResidentBytes(), inventory.getOperationReservedBytes(),
                    inventory.getPhysicalMemoryStatsValid(), inventory.getPhysicalCapacityBytes(),
                    inventory.hasCopyLifetime() ? org.flexlb.service.grpc.EngineStatusConverter.convertCopyLifetime(inventory.getCopyLifetime()) : null));
        }
        return Map.copyOf(result);
    }
}
