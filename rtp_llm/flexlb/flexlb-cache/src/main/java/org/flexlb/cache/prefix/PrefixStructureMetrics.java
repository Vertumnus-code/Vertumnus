package org.flexlb.cache.prefix;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public record PrefixStructureMetrics(long sampledMillis, long logicalNodes, long workerLanes,
                                     long completeReplicaBlocks, long logicalOrphanNodes,
                                     long physicalOrphanBlocks, long unresolvedPresenceBlocks) {
    static PrefixStructureMetrics captureFromHolders(long sampledMillis,
            Map<PrefixObserver.Node, PrefixObserver.Node> parents,
            Map<String, ConfirmedCacheDirectory.Snapshot> inventory,
            Map<PrefixObserver.Node, List<String>> completeHolders) {
        var lanes = new HashMap<ConfirmedCacheDirectory.Lane, ConfirmedCacheDirectory.Snapshot>();
        var routeLanes = new HashMap<String, ConfirmedCacheDirectory.Lane>();
        var completeHashes = new HashMap<ConfirmedCacheDirectory.Lane, Set<Long>>();
        inventory.forEach((route, snapshot) -> {
            if (!snapshot.report().deviceReady()) { return; }
            var lane = new ConfirmedCacheDirectory.Lane(snapshot.report().workerId(), snapshot.report().dpLane());
            lanes.put(lane, snapshot);
            routeLanes.put(route, lane);
            completeHashes.putIfAbsent(lane, new HashSet<>());
        });
        var known = new HashSet<Long>();
        long heldNodes = 0;
        for (var node : parents.keySet()) {
            Long hash = node.hash();
            known.add(hash);
            boolean held = false;
            for (String route : completeHolders.getOrDefault(node, List.of())) {
                var lane = routeLanes.get(route);
                if (lane == null) { continue; }
                completeHashes.get(lane).add(hash);
                held = true;
            }
            if (held) { heldNodes++; }
        }
        long completeReplicaBlocks = 0;
        long physicalOrphanBlocks = 0;
        long unresolvedPresenceBlocks = 0;
        for (var entry : lanes.entrySet()) {
            var complete = completeHashes.get(entry.getKey());
            completeReplicaBlocks += complete.size();
            for (Long key : entry.getValue().keys()) {
                if (!known.contains(key)) { unresolvedPresenceBlocks++; }
                else if (!complete.contains(key)) { physicalOrphanBlocks++; }
            }
        }
        return new PrefixStructureMetrics(sampledMillis, parents.size(), lanes.size(), completeReplicaBlocks,
                parents.size() - heldNodes, physicalOrphanBlocks, unresolvedPresenceBlocks);
    }

    public static PrefixStructureMetrics capture(long sampledMillis, Map<PrefixObserver.Node, PrefixObserver.Node> parents,
                                                  Map<String, ConfirmedCacheDirectory.Snapshot> inventory) {
        var nodes = parents.keySet().stream().sorted(Comparator.comparingInt(PrefixObserver.Node::depth)).toList();
        Set<Long> known = new HashSet<>();
        nodes.forEach(node -> known.add(node.hash()));
        var lanes = new HashMap<ConfirmedCacheDirectory.Lane, ConfirmedCacheDirectory.Snapshot>();
        inventory.values().stream().filter(snapshot -> snapshot.report().deviceReady()).forEach(snapshot -> {
            var lane = new ConfirmedCacheDirectory.Lane(snapshot.report().workerId(), snapshot.report().dpLane());
            lanes.put(lane, snapshot);
        });
        Set<PrefixObserver.Node> held = new HashSet<>();
        long completeReplicaBlocks = 0;
        long physicalOrphanBlocks = 0;
        long unresolvedPresenceBlocks = 0;
        for (var snapshot : lanes.values()) {
            Set<PrefixObserver.Node> complete = new HashSet<>();
            Set<Long> completeHashes = new HashSet<>();
            for (var node : nodes) {
                var parent = parents.get(node);
                boolean ancestorsReady = parent == null ? node.depth() == 1 : complete.contains(parent);
                if (ancestorsReady && snapshot.keys().contains(node.hash())) {
                    complete.add(node);
                    completeHashes.add(node.hash());
                }
            }
            held.addAll(complete);
            completeReplicaBlocks += completeHashes.size();
            for (long key : snapshot.keys()) {
                if (!known.contains(key)) { unresolvedPresenceBlocks++; }
                else if (!completeHashes.contains(key)) { physicalOrphanBlocks++; }
            }
        }
        return new PrefixStructureMetrics(sampledMillis, nodes.size(), lanes.size(), completeReplicaBlocks,
                nodes.size() - held.size(), physicalOrphanBlocks, unresolvedPresenceBlocks);
    }
}
