package org.flexlb.cache.prefix;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.flexlb.dao.cache.CacheReportV2;

final class PrefixRoutingHistory {
    private record Identity(String worker, int lane, String epoch, String layout, int degree) {
        static Identity of(CacheReportV2 report) {
            return new Identity(report.workerId(), report.dpLane(), report.epoch(),
                    report.layoutFingerprint(), report.cpDegree());
        }
    }
    private record History(Identity identity, LinkedHashMap<Long, Boolean> keys) { }
    private final int capacity;
    private final Map<String, History> routes = new HashMap<>();

    PrefixRoutingHistory(int capacity) {
        if (capacity < 1) { throw new IllegalArgumentException("history capacity must be positive"); }
        this.capacity = capacity;
    }

    int capacity() { return capacity; }

    synchronized void remember(String route, ConfirmedCacheDirectory.Snapshot snapshot,
                               List<PrefixObserver.Node> path) {
        if (!snapshot.report().deviceReady()) { return; }
        var identity = Identity.of(snapshot.report());
        var history = routes.compute(route, (ignored, previous) ->
                previous != null && previous.identity().equals(identity) ? previous
                        : new History(identity, new LinkedHashMap<>(16, 0.75f, true)));
        for (var node : path) {
            if (snapshot.unavailableKeys().contains(node.hash())) { break; }
            history.keys().put(node.hash(), Boolean.TRUE);
            while (history.keys().size() > capacity) {
                history.keys().pollFirstEntry();
            }
        }
    }

    synchronized void merge(Map<String, Integer> predictions,
                            Map<String, ConfirmedCacheDirectory.Snapshot> view,
                            List<Long> keys, List<String> candidates) {
        routes.entrySet().removeIf(entry -> {
            var snapshot = view.get(entry.getKey());
            return snapshot == null || !snapshot.report().deviceReady()
                    || !entry.getValue().identity().equals(Identity.of(snapshot.report()));
        });
        for (var route : candidates) {
            var history = routes.get(route);
            var snapshot = view.get(route);
            if (history == null || snapshot == null) { continue; }
            int matched = 0;
            for (var key : keys) {
                if (!history.keys().containsKey(key) || snapshot.unavailableKeys().contains(key)) { break; }
                matched++;
            }
            predictions.merge(route, matched, Math::max);
        }
    }

    synchronized void invalidate(String route) { routes.remove(route); }
    synchronized void clear() { routes.clear(); }
}
