package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

@Data
@Slf4j
public class WorkerStatus {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private String role;
    private String group;
    private String ip;
    private int port;
    private String site;
    private Long availableConcurrency;
    private boolean alive;
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong usedKvCacheTokens = new AtomicLong();
    private CacheStatus cacheStatus;
    private AtomicLong runningQueueTime = new AtomicLong();
    private Map<String, TaskInfo> waitingTaskList;
    private Map<String, TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1L);

    private ConcurrentHashMap<Long/*requestId*/, TaskInfo> localTaskMap = new ConcurrentHashMap<>();
    private double stepLatencyMs;
    private long iterateCount;
    private long dpSize;
    private long tpSize;
    private volatile long cpSize = -1L;
    private long worldSize = -1L;
    private long localWorldSize = -1L;
    private volatile String profileName;
    private String cpRotateMethod;
    private volatile long cpEpoch = -1L;
    private volatile long cpPrepareEpoch = -1L;
    private volatile String cpPhase;
    private volatile long cpTargetSize = -1L;
    private volatile long logicalLaneCount = -1L;
    /** Ranks behind one endpoint, a launch constant that does not follow the CP degree. */
    private volatile long cpRanksPerEndpoint = -1L;
    /**
     * Whether this endpoint is the CP lane's ingress at the ACTIVE degree, i.e. the one
     * a router may address. A CP lane admits external traffic only at its leader and
     * every other endpoint of the lane answers 503; which endpoints lead is a function
     * of the degree, so it changes under the router while the endpoint set cannot.
     *
     * <p>{@code null} means the worker published no claim -- an older engine, or one
     * that could not resolve its own rank. It MUST read as "serving": a boxed Boolean
     * rather than a {@code boolean} exists precisely so that absent and false are
     * distinguishable, because defaulting a silent worker to false would empty the
     * candidate set and take the whole pool out of rotation.
     */
    private volatile Boolean cpAcceptsExternal;
    private volatile long routeFanout = -1L;
    private ConcurrentHashMap<Integer, DpRankStatus> dpRankStatusMap = new ConcurrentHashMap<>();
    private ConcurrentHashMap<Integer, CacheStatus> dpRankCacheStatusMap = new ConcurrentHashMap<>();
    /** Scheduler-selected DP lane. This decouples dispatch from requestId % dpSize. */
    private ConcurrentHashMap<Long, Integer> requestDpRankMap = new ConcurrentHashMap<>();
    /** Rotates equal-load tie breaking so idle sequential traffic uses every DP lane. */
    private AtomicLong nextDpRank = new AtomicLong();

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1); // Last status update time (microseconds)
    private AtomicLong statusUpdateIntervalUs = new AtomicLong(0); // Actual interval between last two status updates (microseconds)
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1); // Last cache status update time
    private AtomicLong cacheKeysFetchTime = new AtomicLong(0); // Last key-bearing decode cache poll (ms)
    private AtomicLong lastSelectedTime = new AtomicLong(-1); // Last selection time
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true); // Resource availability state
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false); // Status check in progress flag
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false); // Cache check in progress flag
    private AtomicLong statusVersion = new AtomicLong(-1L);
    @lombok.Setter(lombok.AccessLevel.NONE)
    private volatile PrefixTopology prefixTopology;

    public record PrefixTopology(String switchId, String namespace, int degree, int lanes,
                                 Map<Integer, String> workers, Map<Integer, String> epochs) {
        public PrefixTopology {
            workers = Map.copyOf(workers);
            epochs = Map.copyOf(epochs);
            if (!java.util.UUID.fromString(switchId).toString().equals(switchId) || namespace.isBlank()
                    || degree < 1 || lanes < 1 || Math.multiplyExact(degree, lanes) > 4096
                    || workers.size() != lanes || epochs.size() != lanes) {
                throw new IllegalArgumentException("invalid admitted frontend topology");
            }
            for (int lane = 0; lane < lanes; lane++) {
                if (workers.get(lane) == null || workers.get(lane).isBlank()
                        || epochs.get(lane) == null || epochs.get(lane).isBlank()) {
                    throw new IllegalArgumentException("incomplete admitted frontend lanes");
                }
            }
        }
    }

    public long getCpSize() { return prefixTopology == null ? cpSize : prefixTopology.degree(); }
    public long getTpSize() { return prefixTopology == null ? tpSize : prefixTopology.degree(); }
    public long getDpSize() { return prefixTopology == null ? dpSize : prefixTopology.lanes(); }
    public long getWorldSize() { return prefixTopology == null ? worldSize : (long) prefixTopology.degree() * prefixTopology.lanes(); }
    public boolean isNativeCpTopology() {
        return "native".equalsIgnoreCase(System.getenv("FLEXLB_CP_SWITCH_MODE"));
    }
    public String getCpSwitchDomain() { return isNativeCpTopology() ? getIpPort() : getIp(); }
    public long getCpEpoch() {
        if (!isNativeCpTopology()) { return cpEpoch; }
        return prefixTopology == null ? 0L
                : java.util.UUID.fromString(prefixTopology.switchId()).getMostSignificantBits() & Long.MAX_VALUE;
    }
    public long getCpPrepareEpoch() { return isNativeCpTopology() ? getCpEpoch() : cpPrepareEpoch; }
    public String getCpPhase() { return isNativeCpTopology() ? "commit" : cpPhase; }
    public long getLogicalLaneCount() { return isNativeCpTopology() ? getDpSize() : logicalLaneCount; }
    public long getRouteFanout() { return isNativeCpTopology() ? 1L : routeFanout; }
    public String getProfileName() {
        return prefixTopology == null ? profileName : "cp" + prefixTopology.degree() + "-tp" + prefixTopology.degree();
    }

    public void installPrefixTopology(PrefixTopology topology) {
        java.util.Objects.requireNonNull(topology);
        lock.lock();
        try {
            if (topology.equals(prefixTopology)) { return; }
            if ((prefixTopology != null && prefixTopology.switchId().equals(topology.switchId())) || !localTaskMap.isEmpty()) {
                throw new IllegalStateException("frontend topology changed identity or still has local requests");
            }
            dpRankStatusMap.clear();
            dpRankCacheStatusMap.clear();
            requestDpRankMap.clear();
            waitingTaskList = Map.of();
            runningTaskList = Map.of();
            cacheStatus = null;
            nextDpRank.set(0);
            runningQueueTime.set(0);
            statusVersion.set(-1);
            latestFinishedTaskVersion.set(-1);
            prefixTopology = topology;
        } finally { lock.unlock(); }
    }

    public boolean acceptsPrefixInventory(org.flexlb.dao.cache.CacheReportV2 report) {
        var topology = prefixTopology;
        return topology == null || (report != null && topology.namespace().equals(report.namespace())
                && topology.degree() == report.cpDegree() && java.util.Objects.equals(report.workerId(), topology.workers().get(report.dpLane()))
                && java.util.Objects.equals(report.epoch(), topology.epochs().get(report.dpLane()))
                && topology.epochs().containsKey(report.dpLane()));
    }

    public String getProfileKey() {
        if (prefixTopology != null) { return getProfileName(); }
        if (!isBlank(profileName)) {
            return profileName.trim();
        }
        if (cpSize > 0 && tpSize > 0) {
            return "cp" + cpSize + "-tp" + tpSize;
        }
        if (cpSize > 0) {
            return "cp" + cpSize;
        }
        if (tpSize > 0) {
            return "tp" + tpSize;
        }
        return "default";
    }

    public boolean matchesProfile(String profile) {
        if (isBlank(profile)) {
            return true;
        }
        String expectedProfile = profile.trim();
        return expectedProfile.equals(getProfileKey())
                || (!isBlank(getProfileName()) && expectedProfile.equals(getProfileName().trim()));
    }

    public boolean isCpRoutingReady() {
        return isCpPhaseCommitted() && isCpExternalIngress();
    }

    /** The reported geometry is the one the worker is actually serving. */
    public boolean isCpPhaseCommitted() {
        return cpPhase == null || cpPhase.isBlank() || "commit".equalsIgnoreCase(cpPhase);
    }

    /**
     * Whether a router may address this endpoint at the degree it currently reports.
     *
     * <p>This is deliberately a ROUTING predicate only. Composition reconstruction
     * (EngineWorkerStatus#liveCpWorkers and everything reached through cpWorkers())
     * reads the raw status maps and must keep seeing every endpoint: a CP4 lane on an
     * 8-rank world is two endpoints and one worker, so filtering the follower out of
     * the composition would report a 4-GPU pool and halve every capacity comparison.
     * Routing needs the opposite view -- one candidate per lane -- and the two are
     * reconciled here rather than by giving the follower a second, fake identity.
     *
     * <p>Unknown means yes. A worker that publishes no claim is either an engine
     * without the field or one that could not resolve its own rank, and neither is a
     * reason to stop routing to it.
     */
    public boolean isCpExternalIngress() {
        return cpAcceptsExternal == null || cpAcceptsExternal;
    }

    public int getInflightTaskCount() {
        int remoteInflight = mapSize(waitingTaskList) + mapSize(runningTaskList);
        return Math.max(localTaskMap.size(), remoteInflight);
    }

    public int resolveDpRank(long requestId) {
        Integer selectedRank = requestDpRankMap.get(requestId);
        if (selectedRank != null) {
            return selectedRank;
        }
        int fanout = (int) Math.max(1L, getRouteFanout());
        int logicalReplicas = (int) Math.max(1L, getLogicalLaneCount() > 0 ? getLogicalLaneCount() : getDpSize() / fanout);
        return Math.floorMod(requestId, logicalReplicas) * fanout;
    }

    public int selectAndPinDpRank(long requestId) {
        int fanout = (int) Math.max(1L, getRouteFanout());
        int logicalReplicas = (int) Math.max(1L, getLogicalLaneCount() > 0 ? getLogicalLaneCount() : getDpSize() / fanout);
        int firstLogicalRank = Math.floorMod(nextDpRank.getAndIncrement(), logicalReplicas);
        int firstRank = firstLogicalRank * fanout;
        int bestRank = firstRank;
        int bestInflight = Integer.MAX_VALUE;
        boolean foundAvailableRank = false;
        for (int offset = 0; offset < logicalReplicas; offset++) {
            int rank = ((firstLogicalRank + offset) % logicalReplicas) * fanout;
            DpRankStatus remote = dpRankStatusMap.get(rank);
            if (remote != null && !remote.getResourceAvailable().get()) {
                continue;
            }
            foundAvailableRank = true;
            int remoteInflight = remote == null ? 0 : remote.getInflightTaskCount();
            final int candidateRank = rank;
            int localInflight = (int) localTaskMap.values().stream()
                    .filter(task -> task.getDpRank() == candidateRank)
                    .count();
            int inflight = Math.max(remoteInflight, localInflight);
            if (inflight < bestInflight) {
                bestInflight = inflight;
                bestRank = rank;
            }
        }
        if (!foundAvailableRank) {
            bestRank = firstRank;
        }
        requestDpRankMap.put(requestId, bestRank);
        return bestRank;
    }

    public void pinDpRank(long requestId, int dpRank) {
        int replicas = (int) Math.max(1L, getDpSize());
        if (dpRank < 0 || dpRank >= replicas) {
            throw new IllegalArgumentException("invalid dpRank=" + dpRank + ", dpSize=" + replicas);
        }
        requestDpRankMap.put(requestId, dpRank);
    }

    public void unpinDpRank(long requestId) {
        requestDpRankMap.remove(requestId);
    }

    public DpRankStatus getDpRankStatusForRequest(long requestId) {
        return dpRankStatusMap.get(resolveDpRank(requestId));
    }

    public CacheStatus getDpRankCacheStatusForRequest(long requestId) {
        return dpRankCacheStatusMap.get(resolveDpRank(requestId));
    }

    public String getDpRankCacheKey(long requestId) {
        return getDpRankCacheKey(resolveDpRank(requestId));
    }

    /**
     * Cache-index key for one DP lane. A worker with a single lane owns its whole cache,
     * so it keys on the plain address the engine itself reports under.
     */
    public String getDpRankCacheKey(int dpRank) {
        return getDpSize() <= 1 ? getIpPort() : getIpPort() + "#dp" + dpRank;
    }

    public int getInflightTaskCountForRequest(long requestId) {
        int dpRank = resolveDpRank(requestId);
        DpRankStatus remote = dpRankStatusMap.get(dpRank);
        int remoteInflight = remote == null ? 0 : remote.getInflightTaskCount();
        int localInflight = (int) localTaskMap.values().stream()
                .filter(task -> task.getDpRank() == dpRank)
                .count();
        return Math.max(localInflight, remoteInflight);
    }

    public void replaceDpRankStatuses(Map<Integer, DpRankStatus> statuses) {
        if (statuses == null) {
            return;
        }
        statuses.forEach((rank, status) -> {
            DpRankStatus previous = dpRankStatusMap.get(rank);
            if (previous != null) {
                status.getResourceAvailable().set(previous.getResourceAvailable().get());
            }
        });
        dpRankStatusMap.keySet().retainAll(statuses.keySet());
        dpRankStatusMap.putAll(statuses);
        dpRankCacheStatusMap.keySet().retainAll(statuses.keySet());
    }

    public void updateDpRankCacheStatus(int dpRank, CacheStatus cacheStatus) {
        if (cacheStatus != null) {
            dpRankCacheStatusMap.put(dpRank, cacheStatus);
        }
    }

    private static int mapSize(Map<?, ?> value) {
        return value == null ? 0 : value.size();
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    /**
     * Add task to local running queue
     * @param requestId Request ID
     * @param taskInfo Task information
     */
    public void putLocalTask(Long requestId, TaskInfo taskInfo) {
        if (taskInfo.getDpRank() < 0) {
            taskInfo.setDpRank(resolveDpRank(requestId));
        } else {
            pinDpRank(requestId, Math.toIntExact(taskInfo.getDpRank()));
        }
        taskInfo.setEnqueueTimeUs(System.nanoTime() / 1000);
        localTaskMap.put(requestId, taskInfo);
        taskInfo.updateTaskState(TaskStateEnum.IN_TRANSIT);

        // Local incremental queue time update
        this.addRunningQueueTime(taskInfo.estimatePrefillTime());
        // Local incremental KV cache tokens update
        long needNewKvCacheLen = taskInfo.getInputLength() - taskInfo.getPrefixLength();
        this.decKvCacheFree(needNewKvCacheLen);
        this.addKvCacheUsed(needNewKvCacheLen);

        lastSelectedTime.set(System.nanoTime() / 1000);
        Logger.debug("Task {} added to local queue with state: {}", requestId, TaskStateEnum.IN_TRANSIT);
    }

    /**
     * Remove task from local running queue
     * @param requestId Request ID
     */
    public void removeLocalTask(Long requestId) {
        TaskInfo taskInfo = localTaskMap.get(requestId);
        if (taskInfo != null) {
            safeDecrementQueueTime(runningQueueTime, taskInfo.estimatePrefillTime());
            long needNewKvCacheLen = taskInfo.getInputLength() - taskInfo.getPrefixLength();
            decKvCacheFree(-needNewKvCacheLen);
            addKvCacheUsed(-needNewKvCacheLen);
            localTaskMap.remove(requestId);
        }
        requestDpRankMap.remove(requestId);
    }

    /**
     * Add estimated execution time to running queue
     * @param len Estimated execution time to add
     */
    public void addRunningQueueTime(long len) {
        runningQueueTime.addAndGet(len);
    }

    public void addKvCacheUsed(long len) {
        usedKvCacheTokens.addAndGet(len);
    }

    public void decKvCacheFree(long len) {
        availableKvCacheTokens.accumulateAndGet(len, (current, decrement) ->
                Math.max(0, current - decrement));
    }

    /**
     * Update task states
     * Check for lost tasks, update running/waiting tasks, and clean up finished tasks
     */
    public void updateTaskStates(Map<String, TaskInfo> waitingTaskInfo, Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        Iterator<Map.Entry<Long, TaskInfo>> iterator = localTaskMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Long, TaskInfo> entry = iterator.next();
            Long requestId = entry.getKey();
            TaskInfo localTask = entry.getValue();
            String requestIdStr = String.valueOf(requestId);

            TaskInfo finishedTask = finishedTaskInfo != null ? finishedTaskInfo.get(requestIdStr) : null;
            if (finishedTask != null) {
                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                localTask.updateTaskState(TaskStateEnum.FINISHED);

                if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
                    long delta = finishedTask.estimatePrefillTime();
                    safeDecrementQueueTime(runningQueueTime, delta);
                }
                Logger.debug("Task {} finished and removed", requestId);
                requestDpRankMap.remove(requestId);
                iterator.remove();
                continue;
            }

            TaskInfo runningTask = runningTaskInfo != null ? runningTaskInfo.get(requestIdStr) : null;
            if (runningTask != null) {
                localTask.setLastActiveTimeUs(System.nanoTime() / 1000);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                if (localTask.getTaskState() != TaskStateEnum.RUNNING) {
                    localTask.updateTaskState(TaskStateEnum.RUNNING);
                }

                localTask.setPrefixLength(runningTask.getPrefixLength());
                localTask.setPrefillTime(runningTask.getPrefillTime());
                localTask.setInputLength(runningTask.getInputLength());
                localTask.setWaitingTime(runningTask.getWaitingTime());
                localTask.setIterateCount(runningTask.getIterateCount());
                localTask.setEndTimeMs(runningTask.getEndTimeMs());
                localTask.setDpRank(requestDpRankMap.containsKey(requestId)
                        ? resolveDpRank(requestId)
                        : runningTask.getDpRank());

                continue;
            }

            TaskInfo waitingTask = waitingTaskInfo != null ? waitingTaskInfo.get(requestIdStr) : null;
            if (waitingTask != null) {
                localTask.setLastActiveTimeUs(System.nanoTime() / 1000);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker (waiting)", requestId);
                }

                localTask.setPrefixLength(waitingTask.getPrefixLength());
                localTask.setInputLength(waitingTask.getInputLength());
                localTask.setWaitingTime(waitingTask.getWaitingTime());
                localTask.setDpRank(requestDpRankMap.containsKey(requestId)
                        ? resolveDpRank(requestId)
                        : waitingTask.getDpRank());

                continue;
            }

            if (localTask.getTaskState() == TaskStateEnum.CONFIRMED || localTask.getTaskState() == TaskStateEnum.RUNNING) {
                localTask.updateTaskState(TaskStateEnum.LOST);
                logger.warn("Task {} marked as LOST - not in waiting, running or finished list", requestId);
            }
        }
    }

    /**
     * Update total queue time for running queue
     */
    public void updateRunningQueueTime() {
        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            runningQueueTime.getAndSet(0);
            return;
        }
        long rectifiedEstimateRunningTime = 0;
        for (Entry<Long, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // Recalculate based on accurate cache hit count, rectify local task running queue time
            rectifiedEstimateRunningTime += taskInfo.estimatePrefillTime();
        }
        if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
            // Only update when rectified time is less than estimated time, because engine layer returned running_list may include queuing tasks where prefixLength=0
            if (runningQueueTime.get() > rectifiedEstimateRunningTime) {
                runningQueueTime.getAndSet(rectifiedEstimateRunningTime);
            }
        }
    }

    public void updateKvCacheTokens(long latestUsedKvCacheTokens, long latestAvailableKvCacheTokens) {

        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
            availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);
            return;
        }

        long inTransitTaskCacheUsed = 0;
        for (Map.Entry<Long, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // Calculate tokens occupied by in-transit task cache miss portion
            if (taskInfo.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                inTransitTaskCacheUsed = inTransitTaskCacheUsed + taskInfo.getInputLength() - taskInfo.getPrefixLength();
            }
        }
        // Rectify KV cache tokens affected by in-transit tasks
        latestUsedKvCacheTokens += inTransitTaskCacheUsed;
        latestAvailableKvCacheTokens -= inTransitTaskCacheUsed;

        usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
        availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);

    }

    /**
     * Safely decrement total queue time for running queue, ensuring it never becomes negative
     *
     * @param runningQueueTime Total queue time for running queue
     * @param timeToReduce Time to reduce
     */
    public static void safeDecrementQueueTime(AtomicLong runningQueueTime, long timeToReduce) {
        if (timeToReduce <= 0) {
            logger.warn("Invalid tokens to reduce: {}", timeToReduce);
            return;
        }
        runningQueueTime.accumulateAndGet(timeToReduce, (currentRunningQueueTime, reductionAmount) -> {
            // Ensure reduction amount is positive, calculate new value, but not less than 0
            long newRunningQueueTime = currentRunningQueueTime - reductionAmount;

            // If result is negative, set to 0, ensuring token count never goes below 0
            return Math.max(newRunningQueueTime, 0L);
        });
    }

    /**
     * Update resource availability with hysteresis to prevent state oscillation.
     * <p>
     * Hysteresis uses two thresholds: upper and lower (calculated as upper - hysteresisBias%).
     * This creates a band where state doesn't change, preventing rapid toggling.
     * <p>
     * State transitions:
     * - AVAILABLE → UNAVAILABLE: when current metric EXCEEDS upper threshold
     * - UNAVAILABLE → AVAILABLE: when current metric FALLS BELOW lower threshold
     *
     * @param currentMetric current resource metric value
     * @param upperThreshold upper threshold for disabling availability
     * @param hysteresisBias bias percentage for calculating lower threshold (lower = upper - upper * bias / 100)
     * @return the new resource availability state
     */
    public boolean updateResourceAvailabilityWithHysteresis(long currentMetric, long upperThreshold, long hysteresisBias) {
        long lowerThreshold = Math.max(0, upperThreshold - (long)(upperThreshold * hysteresisBias / 100.0));

        if (currentMetric >= upperThreshold) {
            resourceAvailable.compareAndSet(true, false);
        } else if (currentMetric <= lowerThreshold) {
            resourceAvailable.compareAndSet(false, true);
        }
        return resourceAvailable.get();
    }

    /**
     * Get IP:PORT format address
     *
     * @return IP:PORT string
     */
    public String getIpPort() {
        if (ip == null) {
            return null;
        }
        return ip + ":" + port;
    }
}
