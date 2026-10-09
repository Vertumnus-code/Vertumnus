package org.flexlb.dao.cache;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record CacheReportV2(@JsonProperty("namespace") String namespace,
                            @JsonProperty("worker_id") String workerId,
                            @JsonProperty("dp_lane") int dpLane,
                            @JsonProperty("epoch") String epoch,
                            @JsonProperty("layout_fingerprint") String layoutFingerprint,
                            @JsonProperty("snapshot_kind") String snapshotKind,
                            @JsonProperty("version") long version,
                            @JsonProperty("base_version") long baseVersion,
                            @JsonProperty("rank_readiness_generation") long rankReadinessGeneration,
                            @JsonProperty("device_ready") boolean deviceReady,
                            @JsonProperty("cp_degree") int cpDegree,
                            @JsonProperty("block_size") int blockSize,
                            @JsonProperty("physical_free_bytes") long physicalFreeBytes,
                            @JsonProperty("device_keys") List<Long> deviceKeys,
                            @JsonProperty("reuse_events") List<PrefixReuseEvent> reuseEvents,
                            @JsonProperty("dropped_reuse_events") long droppedReuseEvents,
                            @JsonProperty("physical_block_bytes") long physicalBlockBytes,
                            @JsonProperty("evictable_bytes") long evictableBytes,
                            @JsonProperty("active_bytes") long activeBytes,
                            @JsonProperty("resident_bytes") long residentBytes,
                            @JsonProperty("operation_reserved_bytes") long operationReservedBytes,
                            @JsonProperty("physical_memory_stats_valid") boolean physicalMemoryStatsValid,
                            @JsonProperty("physical_capacity_bytes") long physicalCapacityBytes,
                            @JsonProperty("copy_lifetime") PrefixCopyLifetime copyLifetime) {
    public CacheReportV2(String namespace, String workerId, int dpLane, String epoch, String layoutFingerprint,
                         String snapshotKind, long version, long baseVersion, long rankReadinessGeneration,
                         boolean deviceReady, int cpDegree, int blockSize, long physicalFreeBytes,
                         List<Long> deviceKeys, List<PrefixReuseEvent> reuseEvents, long droppedReuseEvents,
                         long physicalBlockBytes, long evictableBytes, long activeBytes, long residentBytes,
                         long operationReservedBytes, boolean physicalMemoryStatsValid, long physicalCapacityBytes) {
        this(namespace, workerId, dpLane, epoch, layoutFingerprint, snapshotKind, version, baseVersion,
                rankReadinessGeneration, deviceReady, cpDegree, blockSize, physicalFreeBytes, deviceKeys, reuseEvents,
                droppedReuseEvents, physicalBlockBytes, evictableBytes, activeBytes, residentBytes,
                operationReservedBytes, physicalMemoryStatsValid, physicalCapacityBytes, null);
    }
    public CacheReportV2(String namespace, String workerId, int dpLane, String epoch, String layoutFingerprint,
                         String snapshotKind, long version, long baseVersion, long rankReadinessGeneration,
                         boolean deviceReady, int cpDegree, int blockSize, long physicalFreeBytes,
                         List<Long> deviceKeys, List<PrefixReuseEvent> reuseEvents, long droppedReuseEvents,
                         long physicalBlockBytes) {
        this(namespace, workerId, dpLane, epoch, layoutFingerprint, snapshotKind, version, baseVersion,
                rankReadinessGeneration, deviceReady, cpDegree, blockSize, physicalFreeBytes, deviceKeys, reuseEvents,
                droppedReuseEvents, physicalBlockBytes, 0, 0, 0, 0, false, 0);
    }

    public CacheReportV2 {
        deviceKeys = deviceKeys == null ? List.of() : List.copyOf(deviceKeys);
        reuseEvents = reuseEvents == null ? List.of() : List.copyOf(reuseEvents);
    }

    public boolean memoryStatsConsistent() {
        if (!physicalMemoryStatsValid || physicalBlockBytes <= 0 || physicalCapacityBytes < 0) { return false; }
        long total = 0;
        try {
            for (long bytes : new long[] {physicalFreeBytes, evictableBytes, activeBytes, residentBytes,
                    operationReservedBytes}) {
                if (bytes < 0 || bytes % physicalBlockBytes != 0) { return false; }
                total = Math.addExact(total, bytes);
            }
        } catch (ArithmeticException overflow) { return false; }
        return total == physicalCapacityBytes;
    }

    public record PrefixReuseEvent(@JsonProperty("sequence") long sequence,
                                   @JsonProperty("request_id") long requestId,
                                   @JsonProperty("device_reuse_tokens") long deviceReuseTokens,
                                   @JsonProperty("input_tokens") long inputTokens,
                                   @JsonProperty("prefill") boolean prefill,
                                   @JsonProperty("attempt_id") String attemptId) { }
}
