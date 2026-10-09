package org.flexlb.cache.prefix;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.flexlb.dao.cache.PrefixCopyLifetime;

public record PrefixCopyMetrics(long sampledMillis, List<Lane> lanes) {
    public record Lane(String workerId, int dpLane, String epoch, boolean valid, String unavailableReason,
                       BigInteger publishedBytes, BigInteger reusedBytes, BigInteger unusedRetiredBytes,
                       BigInteger unknownRetiredBytes, BigInteger rightCensoredBytes, BigInteger unknownLiveBytes,
                       boolean individualHistoryComplete, PrefixCopyLifetime raw) { }

    public static PrefixCopyMetrics capture(long sampledMillis,
                                            Map<String, ConfirmedCacheDirectory.Snapshot> inventory) {
        var unique = new HashMap<ConfirmedCacheDirectory.Lane, ConfirmedCacheDirectory.Snapshot>();
        inventory.values().forEach(snapshot -> unique.put(
                new ConfirmedCacheDirectory.Lane(snapshot.report().workerId(), snapshot.report().dpLane()), snapshot));
        List<Lane> lanes = new ArrayList<>();
        for (var snapshot : unique.values()) {
            var report = snapshot.report();
            var raw = report.copyLifetime();
            String reason = !report.deviceReady() || !report.memoryStatsConsistent() ? "WORKER_NOT_MEASURED"
                    : raw == null ? "LEGACY_OR_MISSING" : !raw.consistent() ? "INCONSISTENT_OR_SATURATED" : "";
            if (!reason.isEmpty()) {
                lanes.add(new Lane(report.workerId(), report.dpLane(), report.epoch(), false, reason,
                        null, null, null, null, null, null, false, raw));
                continue;
            }
            long censored = raw.records().stream().filter(copy -> !copy.retired() && !copy.reused() && copy.trackingValid()).count();
            long unknown = raw.records().stream().filter(copy -> !copy.retired() && !copy.reused() && !copy.trackingValid()).count();
            var bytes = BigInteger.valueOf(report.physicalBlockBytes());
            lanes.add(new Lane(report.workerId(), report.dpLane(), report.epoch(), true, "",
                    bytes.multiply(BigInteger.valueOf(raw.publishedBlocks())), bytes.multiply(BigInteger.valueOf(raw.reusedBlocks())),
                    bytes.multiply(BigInteger.valueOf(raw.wastedBlocks())), bytes.multiply(BigInteger.valueOf(raw.unknownRetiredBlocks())),
                    bytes.multiply(BigInteger.valueOf(censored)), bytes.multiply(BigInteger.valueOf(unknown)),
                    raw.droppedTerminalRecords() == 0, raw));
        }
        lanes.sort(java.util.Comparator.comparing(Lane::workerId).thenComparingInt(Lane::dpLane));
        return new PrefixCopyMetrics(sampledMillis, List.copyOf(lanes));
    }
}
