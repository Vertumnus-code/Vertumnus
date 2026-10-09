package org.flexlb.dao.master;

import lombok.Data;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/** Snapshot of one logical DP replica behind a physical RTP frontend. */
@Data
public class DpRankStatus {
    private int dpRank;
    private String address;
    private boolean alive;
    private long availableConcurrency;
    private double stepLatencyMs;
    private long iterateCount;
    private Map<String, TaskInfo> waitingTaskInfo;
    private Map<String, TaskInfo> runningTaskInfo;
    private Map<String, TaskInfo> finishedTaskInfo;
    private final AtomicBoolean resourceAvailable = new AtomicBoolean(true);

    public int getInflightTaskCount() {
        return size(waitingTaskInfo) + size(runningTaskInfo);
    }

    public int getWaitingTaskCount() {
        return size(waitingTaskInfo);
    }

    public boolean updateResourceAvailabilityWithHysteresis(
            long currentMetric, long upperThreshold, long hysteresisBias) {
        long lowerThreshold = Math.max(
                0L, upperThreshold - Math.round(upperThreshold * hysteresisBias / 100.0));
        if (currentMetric >= upperThreshold) {
            resourceAvailable.compareAndSet(true, false);
        } else if (currentMetric <= lowerThreshold) {
            resourceAvailable.compareAndSet(false, true);
        }
        return resourceAvailable.get();
    }

    private static int size(Map<?, ?> value) {
        return value == null ? 0 : value.size();
    }
}
