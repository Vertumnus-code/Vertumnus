package org.flexlb.sync.status;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Comparator;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Slf4j
@Data
@Component
public class EngineWorkerStatus {

    public static final ModelWorkerStatus MODEL_ROLE_WORKER_STATUS = new ModelWorkerStatus();
    private static final long DEFAULT_SWITCH_TIMEOUT_MS = 600_000L;
    private static final long DEFAULT_SWITCH_POLL_INTERVAL_MS = 200L;

    public final ModelMetaConfig modelMetaConfig;
    private final Object profileSwitchLock = new Object();
    private volatile boolean profileSwitchInProgress;
    private volatile String pendingProfileName;
    private volatile String drainingProfileName;
    private final Object cpSwitchLock = new Object();
    private volatile boolean cpSwitchInProgress;
    private volatile boolean cpSwitchBlocked;
    /**
     * The endpoints the actuator is moving in THIS transition. A resident switch changes
     * exactly one host at a time and the other hosts keep
     * serving at their old degree, so pausing the whole PREFILL role for the length of
     * the transition is not a safety property there -- it is a total routing outage that
     * lasts longer the more gradual the walk is. When the set is non-empty, admission
     * stays open and only these endpoints are withheld.
     */
    private volatile Set<String> cpSwitchParticipants = Set.of();
    private volatile CpSoftQuiesceState cpSoftQuiesceState = CpSoftQuiesceState.idle(0L);
    private volatile long cpSoftQuiesceVersion;
    private volatile long pendingCpSize = -1L;
    private volatile long pendingCpEpoch = -1L;
    /** Compatibility view of the last participant published, never the mixed fleet version. */
    private volatile long committedCpSize = -1L;
    private volatile long committedCpEpoch = -1L;
    /** Manager-owned version of the published fleet composition, not a host-local engine epoch. */
    private volatile long compositionEpoch;
    private volatile Map<Integer, Integer> publishedCpComposition = Map.of();
    private volatile long preSwitchCpSize = -1L;
    private volatile long preSwitchCpEpoch = -1L;
    private volatile String lastCpSwitchError;
    private volatile String lastCpReconcileOutcome;
    private volatile String lastCpReconcileDetail;
    private volatile long lastCpReconcileAtMs;
    private volatile long cpReconcileAttemptCount;

    public EngineWorkerStatus(ModelMetaConfig modelMetaConfig) {
        this.modelMetaConfig = modelMetaConfig;
    }

    public Map<String/*ipPort*/, WorkerStatus> selectModelWorkerStatus(RoleType roleType, String group) {

        if (isSchedulingPaused(roleType)) {
            log.info("Worker composition switch is in progress or blocked, pause new scheduling requests, role: {}, group: {}", roleType, group);
            return Map.of();
        }

        Map<String/*ip:port*/, WorkerStatus> roleStatusMap =
                MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType, MODEL_ROLE_WORKER_STATUS.getActiveProfileName());

        if (roleStatusMap == null) {
            return Map.of();
        }

        return filterByGroup(roleStatusMap, group);
    }

    public Map<String/*ipPort*/, WorkerStatus> selectAllProfileModelWorkerStatus(RoleType roleType, String group) {

        if (isSchedulingPaused(roleType)) {
            log.info("Worker composition switch is in progress or blocked, pause new scheduling requests, role: {}, group: {}", roleType, group);
            return Map.of();
        }

        Map<String/*ip:port*/, WorkerStatus> roleStatusMap =
                MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);

        if (roleStatusMap == null) {
            return Map.of();
        }

        return filterByGroup(roleStatusMap, group);
    }

    private Map<String/*ipPort*/, WorkerStatus> filterByGroup(Map<String/*ipPort*/, WorkerStatus> roleStatusMap, String group) {
        Set<String> participants = cpSwitchParticipants;
        CpSoftQuiesceState softQuiesce = cpSoftQuiesceState;
        return roleStatusMap.entrySet()
                .stream()
                .filter(entry -> {
                    WorkerStatus workerStatus = entry.getValue();
                    return workerStatus != null
                            && !participants.contains(entry.getKey())
                            && !(softQuiesce.active()
                            && softQuiesce.participants().contains(entry.getKey()))
                            && workerStatus.isCpRoutingReady()
                            && (group == null || (workerStatus.getGroup() != null && workerStatus.getGroup().equals(group)));
                })
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    public String getActiveProfileName() {
        return MODEL_ROLE_WORKER_STATUS.getActiveProfileName();
    }

    public void switchActiveProfile(String profileName) {
        String normalizedProfile = normalizeProfile(profileName);
        MODEL_ROLE_WORKER_STATUS.setActiveProfileName(normalizedProfile);
        log.info("Switch active worker profile to {}", normalizedProfile);
    }

    public Map<String, Object> switchActiveProfileAfterDrain(String profileName) throws InterruptedException {
        return switchActiveProfileAfterDrain(profileName, DEFAULT_SWITCH_TIMEOUT_MS, DEFAULT_SWITCH_POLL_INTERVAL_MS);
    }

    public Map<String, Object> switchActiveProfileAfterDrain(String profileName, long timeoutMs, long pollIntervalMs)
            throws InterruptedException {
        String targetProfile = normalizeProfile(profileName);
        long deadlineMs = System.currentTimeMillis() + Math.max(timeoutMs, 0L);
        long intervalMs = Math.max(pollIntervalMs, 50L);

        synchronized (profileSwitchLock) {
            if (cpSwitchInProgress || cpSwitchBlocked || cpSoftQuiesceState.active()) {
                throw new IllegalStateException("CP composition switch is active; profile switch is not allowed");
            }
            String currentProfile = getActiveProfileName();
            if (sameProfile(currentProfile, targetProfile)) {
                log.info("Skip worker profile switch, active profile already {}", targetProfile);
                return profileSnapshot();
            }

            profileSwitchInProgress = true;
            pendingProfileName = targetProfile;
            drainingProfileName = currentProfile;
            log.info("Start draining worker profile before switch, from {} to {}", currentProfile, targetProfile);

            try {
                while (true) {
                    int inflightTaskCount = getInflightTaskCountForProfile(currentProfile);
                    if (inflightTaskCount == 0) {
                        switchActiveProfile(targetProfile);
                        log.info("Finish worker profile drain and switch to {}", targetProfile);
                        profileSwitchInProgress = false;
                        pendingProfileName = null;
                        drainingProfileName = null;
                        return profileSnapshot();
                    }

                    long waitMs = Math.min(intervalMs, deadlineMs - System.currentTimeMillis());
                    if (waitMs <= 0L) {
                        throw new IllegalStateException("Timeout waiting for profile " + currentProfile
                                + " to drain, inflight_task_count=" + inflightTaskCount);
                    }
                    profileSwitchLock.wait(waitMs);
                }
            } finally {
                profileSwitchInProgress = false;
                pendingProfileName = null;
                drainingProfileName = null;
            }
        }
    }

    public Map<String, Object> profileSnapshot() {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("active_profile_name", getActiveProfileName());
        snapshot.put("known_profiles", MODEL_ROLE_WORKER_STATUS.getKnownProfiles());
        snapshot.put("worker_total_count", MODEL_ROLE_WORKER_STATUS.getWorkerTotalCount());
        snapshot.put("switch_in_progress", profileSwitchInProgress);
        snapshot.put("pending_profile_name", pendingProfileName);
        snapshot.put("draining_profile_name", drainingProfileName);
        snapshot.put("draining_inflight_task_count", profileSwitchInProgress ? getInflightTaskCountForProfile(drainingProfileName) : 0);
        snapshot.put("cp_switch_in_progress", cpSwitchInProgress);
        snapshot.put("cp_switch_blocked", cpSwitchBlocked);
        snapshot.put("cp_soft_quiesce", cpSoftQuiesceSnapshot());
        snapshot.put("pending_cp_size", pendingCpSize);
        snapshot.put("pending_cp_epoch", pendingCpEpoch);
        snapshot.put("committed_cp_size", getCommittedCpSize());
        snapshot.put("committed_cp_epoch", getCommittedCpEpoch());
        snapshot.put("last_published_participant_cp_size", getCommittedCpSize());
        snapshot.put("last_published_participant_host_epoch", getCommittedCpEpoch());
        snapshot.put("composition_epoch", compositionEpoch);
        snapshot.put("published_cp_composition", publishedCpComposition);
        snapshot.put("last_cp_switch_error", lastCpSwitchError);
        snapshot.put("pre_switch_cp_size", preSwitchCpSize);
        snapshot.put("pre_switch_cp_epoch", preSwitchCpEpoch);
        snapshot.put("cp_reconcile", cpReconcileSnapshot());
        snapshot.put("workers", workerSnapshots());
        return snapshot;
    }

    private List<Map<String, Object>> workerSnapshots() {
        List<Map<String, Object>> workers = new ArrayList<>();
        appendWorkerSnapshots(workers, RoleType.PDFUSION, MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap());
        appendWorkerSnapshots(workers, RoleType.PREFILL, MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap());
        appendWorkerSnapshots(workers, RoleType.DECODE, MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap());
        appendWorkerSnapshots(workers, RoleType.VIT, MODEL_ROLE_WORKER_STATUS.getVitStatusMap());
        return workers;
    }

    private void appendWorkerSnapshots(List<Map<String, Object>> workers, RoleType roleType,
                                       Map<String, WorkerStatus> statusMap) {
        statusMap.forEach((ipPort, workerStatus) -> {
            if (workerStatus == null) {
                return;
            }
            Map<String, Object> worker = new LinkedHashMap<>();
            worker.put("ip_port", ipPort);
            worker.put("role", roleType);
            worker.put("group", workerStatus.getGroup());
            worker.put("alive", workerStatus.isAlive());
            worker.put("profile_name", workerStatus.getProfileKey());
            worker.put("raw_profile_name", workerStatus.getProfileName());
            worker.put("dp_size", workerStatus.getDpSize());
            worker.put("tp_size", workerStatus.getTpSize());
            worker.put("cp_size", workerStatus.getCpSize());
            worker.put("world_size", workerStatus.getWorldSize());
            worker.put("local_world_size", workerStatus.getLocalWorldSize());
            worker.put("cp_rotate_method", workerStatus.getCpRotateMethod());
            worker.put("cp_epoch", workerStatus.getCpEpoch());
            worker.put("cp_prepare_epoch", workerStatus.getCpPrepareEpoch());
            worker.put("cp_phase", workerStatus.getCpPhase());
            worker.put("cp_target_size", workerStatus.getCpTargetSize());
            worker.put("logical_lane_count", workerStatus.getLogicalLaneCount());
            worker.put("cp_ranks_per_endpoint", workerStatus.getCpRanksPerEndpoint());
            worker.put("cp_accepts_external", workerStatus.getCpAcceptsExternal());
            worker.put("route_fanout", workerStatus.getRouteFanout());
            worker.put("cp_routing_ready", workerStatus.isCpRoutingReady());
            worker.put("available_concurrency", workerStatus.getAvailableConcurrency());
            worker.put("available_kv_cache_tokens", workerStatus.getAvailableKvCacheTokens().get());
            worker.put("used_kv_cache_tokens", workerStatus.getUsedKvCacheTokens().get());
            worker.put("inflight_task_count", workerStatus.getInflightTaskCount());
            worker.put("status_version", workerStatus.getStatusVersion().get());
            workers.add(worker);
        });
    }

    private int getInflightTaskCountForProfile(String profileName) {
        return allWorkers()
                .filter(workerStatus -> workerStatus != null && workerStatus.matchesProfile(profileName))
                .mapToInt(WorkerStatus::getInflightTaskCount)
                .sum();
    }

    private Stream<WorkerStatus> allWorkers() {
        return Stream.of(
                MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap(),
                MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap(),
                MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap(),
                MODEL_ROLE_WORKER_STATUS.getVitStatusMap())
                .flatMap(statusMap -> statusMap.values().stream());
    }

    /** Gate scheduling before the external actuator starts quiescing the workers. */
    public boolean beginCpSwitch(long targetCpSize) {
        return false;
    }

    /**
     * Gate scheduling before the external actuator starts quiescing the workers.
     *
     * <p>{@code participants} is the endpoint set this transition actually touches. Pass
     * it whenever the actuator moves a subset of the pool: admission then stays open for
     * every other endpoint and only the participants are withheld, which is what makes a
     * host-at-a-time walk serve traffic throughout. An empty participant set is rejected:
     * this controller does not permit a whole-pool switch.
     */
    public boolean beginCpSwitch(long targetCpSize, Collection<String> participants) {
        synchronized (profileSwitchLock) {
            synchronized (cpSwitchLock) {
                if (targetCpSize <= 0 || participants == null || participants.isEmpty()
                        || profileSwitchInProgress || cpSwitchInProgress || cpSwitchBlocked
                        || cpSoftQuiesceState.active()) {
                    return false;
                }
                Set<String> selected = Set.copyOf(participants);
                CpAdoptionView before = cpAdoptionView(selected);
                if (!before.unanimous()) {
                    log.error("Refuse CP switch because participant geometry is not settled: {}",
                            before.detail());
                    return false;
                }
                cpSwitchParticipants = selected;
                cpSwitchInProgress = true;
                pendingCpSize = targetCpSize;
                pendingCpEpoch = -1L;
                preSwitchCpSize = before.cpSize();
                preSwitchCpEpoch = before.epoch();
                lastCpSwitchError = null;
                log.info("Start CP composition switch, from CP{} to CP{}, participants={}",
                        preSwitchCpSize, targetCpSize,
                        cpSwitchParticipants);
                return true;
            }
        }
    }

    /**
     * Withhold one resident host before the hard switch while the other hosts remain
     * schedulable. The returned version identifies cancellation attempts belonging to
     * this operation and prevents their results from leaking into a later conversion.
     */
    public boolean beginCpSoftQuiesce(long targetCpSize, Collection<String> participants,
                                      String host, String operationId) {
        synchronized (profileSwitchLock) {
            synchronized (cpSwitchLock) {
                Set<String> selected = participants == null ? Set.of() : Set.copyOf(participants);
                if (targetCpSize <= 0 || selected.isEmpty() || isBlank(host)
                        || isBlank(operationId) || profileSwitchInProgress
                        || cpSwitchInProgress || cpSwitchBlocked) {
                    return false;
                }
                CpSoftQuiesceState current = cpSoftQuiesceState;
                if (current.active()) {
                    return current.matches(targetCpSize, selected, host, operationId);
                }
                CpAdoptionView before = cpAdoptionView(selected);
                if (!before.unanimous()) {
                    log.error("Refuse CP soft quiesce because participant geometry is not settled: {}",
                            before.detail());
                    return false;
                }
                long version = ++cpSoftQuiesceVersion;
                cpSoftQuiesceState = new CpSoftQuiesceState(true, version,
                        targetCpSize, selected, host.trim(), operationId.trim());
                log.info("Start CP soft quiesce version={} operation={} host={} target=CP{} participants={}",
                        version, operationId, host, targetCpSize, selected);
                return true;
            }
        }
    }

    /** Promote the already-drained participant gate to the hard switch state. */
    public boolean promoteCpSoftQuiesce(long targetCpSize, Collection<String> participants,
                                        String host, String operationId) {
        synchronized (profileSwitchLock) {
            synchronized (cpSwitchLock) {
                Set<String> selected = participants == null ? Set.of() : Set.copyOf(participants);
                CpSoftQuiesceState current = cpSoftQuiesceState;
                if (profileSwitchInProgress || cpSwitchInProgress || cpSwitchBlocked
                        || !current.active()
                        || !current.matches(targetCpSize, selected, host, operationId)) {
                    return false;
                }
                CpParticipantInflightSnapshot inflight = cpParticipantInflightSnapshot(selected);
                if (!inflight.drained()) {
                    log.warn("Refuse to promote CP soft quiesce before drain: counts={} missing={}",
                            inflight.endpointCounts(), inflight.missingEndpoints());
                    return false;
                }
                CpAdoptionView before = cpAdoptionView(selected);
                if (!before.unanimous()) {
                    return false;
                }
                cpSwitchParticipants = selected;
                cpSwitchInProgress = true;
                pendingCpSize = targetCpSize;
                pendingCpEpoch = -1L;
                preSwitchCpSize = before.cpSize();
                preSwitchCpEpoch = before.epoch();
                lastCpSwitchError = null;
                cpSoftQuiesceState = CpSoftQuiesceState.idle(current.version());
                log.info("Promote CP soft quiesce version={} to switch, from CP{} to CP{}, participants={}",
                        current.version(), preSwitchCpSize, targetCpSize, selected);
                return true;
            }
        }
    }

    public boolean cancelCpSoftQuiesce(String reason) {
        synchronized (cpSwitchLock) {
            CpSoftQuiesceState current = cpSoftQuiesceState;
            if (!current.active()) {
                return false;
            }
            cpSoftQuiesceState = CpSoftQuiesceState.idle(current.version());
            cpSwitchLock.notifyAll();
            log.info("Cancel CP soft quiesce version={} operation={} reason={}",
                    current.version(), current.operationId(), reason);
            return true;
        }
    }

    public record CpSoftQuiesceState(boolean active, long version, long targetCpSize,
                                     Set<String> participants, String host,
                                     String operationId) {
        public CpSoftQuiesceState {
            participants = participants == null ? Set.of() : Set.copyOf(participants);
            host = host == null ? "" : host;
            operationId = operationId == null ? "" : operationId;
        }

        private static CpSoftQuiesceState idle(long version) {
            return new CpSoftQuiesceState(false, version, -1L, Set.of(), "", "");
        }

        private boolean matches(long targetCpSize, Set<String> participants,
                                String host, String operationId) {
            return this.targetCpSize == targetCpSize
                    && this.participants.equals(participants)
                    && this.host.equals(host == null ? "" : host.trim())
                    && this.operationId.equals(operationId == null ? "" : operationId.trim());
        }
    }

    public record CpParticipantInflightSnapshot(long total,
                                                Map<String, Integer> endpointCounts,
                                                Set<String> missingEndpoints) {
        public boolean drained() {
            return total == 0L && missingEndpoints.isEmpty();
        }
    }

    public CpParticipantInflightSnapshot cpParticipantInflightSnapshot(
            Collection<String> endpoints) {
        Set<String> expected = endpoints == null ? Set.of() : Set.copyOf(endpoints);
        Map<String, WorkerStatus> workers = cpWorkerMap();
        Map<String, Integer> counts = new TreeMap<>();
        Set<String> missing = new java.util.TreeSet<>();
        long total = 0L;
        for (String endpoint : expected) {
            WorkerStatus worker = workers.get(endpoint);
            if (worker == null || !worker.isAlive()) {
                missing.add(endpoint);
                continue;
            }
            int count = worker.getInflightTaskCount();
            counts.put(endpoint, count);
            total += count;
        }
        return new CpParticipantInflightSnapshot(total, Map.copyOf(counts), Set.copyOf(missing));
    }

    public record CpReceivedTask(String endpoint, String ip, int httpPort, long requestId) {
    }

    /** Snapshot requests the engine still reports as cancellable RECEIVED work. */
    public List<CpReceivedTask> cpReceivedTasks(Collection<String> endpoints) {
        Set<String> expected = endpoints == null ? Set.of() : Set.copyOf(endpoints);
        Map<String, WorkerStatus> workers = cpWorkerMap();
        List<CpReceivedTask> result = new ArrayList<>();
        for (String endpoint : expected) {
            WorkerStatus worker = workers.get(endpoint);
            if (worker == null || !worker.isAlive() || worker.getWaitingTaskList() == null) {
                continue;
            }
            worker.getWaitingTaskList().values().stream()
                    .filter(Objects::nonNull)
                    .filter(task -> "TASK_PHASE_RECEIVED".equalsIgnoreCase(task.getPhase())
                            || "RECEIVED".equalsIgnoreCase(task.getPhase()))
                    .map(WorkerStatusTask::new)
                    .collect(Collectors.toMap(WorkerStatusTask::requestId,
                            task -> task, (left, right) -> left, LinkedHashMap::new))
                    .values().forEach(task -> {
                        String ip = worker.getIp();
                        int port = worker.getPort();
                        if (worker.isNativeCpTopology() && worker.getDpSize() > 1) {
                            var lane = worker.getDpRankStatusMap().get(Math.toIntExact(task.dpRank()));
                            if (lane == null || lane.getAddress() == null) { return; }
                            var address = java.net.URI.create("http://" + lane.getAddress());
                            if (address.getHost() == null || address.getPort() <= 1) { return; }
                            ip = address.getHost();
                            port = org.flexlb.util.CommonUtils.toHttpPort(address.getPort());
                        }
                        result.add(new CpReceivedTask(endpoint, ip, port, task.requestId()));
                    });
        }
        return List.copyOf(result);
    }

    private record WorkerStatusTask(long requestId, long dpRank) {
        private WorkerStatusTask(org.flexlb.dao.master.TaskInfo task) {
            this(task.getRequestId(), task.getDpRank());
        }
    }

    public Map<String, Object> cpSoftQuiesceSnapshot() {
        CpSoftQuiesceState state = cpSoftQuiesceState;
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("active", state.active());
        snapshot.put("version", state.version());
        snapshot.put("target_cp_size", state.targetCpSize());
        snapshot.put("participants", state.participants());
        snapshot.put("host", state.host());
        snapshot.put("operation_id", state.operationId());
        return snapshot;
    }

    /** Record the participant host's committed epoch while keeping it unpublished. */
    public void markCpSwitchCommitted(long targetCpSize, long epoch) {
        synchronized (cpSwitchLock) {
            pendingCpSize = targetCpSize;
            pendingCpEpoch = epoch;
        }
    }

    /** Publish the new composition to scheduling only after actuator/adoption success. */
    public void finishCpSwitch(long targetCpSize, long epoch,
                               Map<Integer, Integer> composition) {
        synchronized (cpSwitchLock) {
            committedCpSize = targetCpSize;
            committedCpEpoch = epoch;
            publishCpCompositionLocked(composition);
            pendingCpSize = -1L;
            pendingCpEpoch = -1L;
            cpSwitchInProgress = false;
            cpSwitchBlocked = false;
            cpSwitchParticipants = Set.of();
            cpSoftQuiesceState = CpSoftQuiesceState.idle(cpSoftQuiesceVersion);
            lastCpSwitchError = null;
            cpSwitchLock.notifyAll();
            log.info("Published CP composition {} at manager epoch {} after host-local CP{} epoch {}",
                    publishedCpComposition, compositionEpoch, targetCpSize, epoch);
        }
    }

    /** A pre-commit failure leaves the old geometry valid and can reopen admission. */
    public void abortCpSwitch(String error) {
        synchronized (cpSwitchLock) {
            pendingCpSize = -1L;
            pendingCpEpoch = -1L;
            cpSwitchInProgress = false;
            cpSwitchParticipants = Set.of();
            cpSoftQuiesceState = CpSoftQuiesceState.idle(cpSoftQuiesceVersion);
            lastCpSwitchError = error;
            cpSwitchLock.notifyAll();
            log.error("CP composition switch aborted before commit: {}", error);
        }
    }

    /**
     * A post-commit verification failure must not resume traffic blindly: the old
     * geometry is no longer authoritative. Keep admission closed for inspection.
     */
    public void blockCpSwitch(String error) {
        synchronized (cpSwitchLock) {
            cpSwitchInProgress = false;
            cpSwitchBlocked = true;
            lastCpSwitchError = error;
            cpSwitchLock.notifyAll();
            log.error("CP composition committed but verification failed; participant endpoints remain blocked: {}", error);
        }
    }

    /**
     * What the exact participant endpoints report about their host-local CP geometry.
     */
    public record CpAdoptionView(int workerCount, long cpSize, long epoch,
                                 boolean unanimous, String detail) {
    }

    public CpAdoptionView cpAdoptionView() {
        Set<String> participants = cpSwitchParticipants;
        return participants.isEmpty()
                ? new CpAdoptionView(0, -1L, -1L, false,
                        "no CP switch participants are recorded")
                : cpAdoptionView(participants);
    }

    public CpAdoptionView cpAdoptionView(Collection<String> endpoints) {
        Set<String> expected = endpoints == null ? Set.of() : Set.copyOf(endpoints);
        if (expected.isEmpty()) {
            return new CpAdoptionView(0, -1L, -1L, false,
                    "no CP switch participants were supplied");
        }
        Map<String, WorkerStatus> byEndpoint = cpWorkerMap();
        List<String> missing = expected.stream().filter(endpoint -> !byEndpoint.containsKey(endpoint)).toList();
        List<WorkerStatus> workers = expected.stream()
                .map(byEndpoint::get)
                .filter(Objects::nonNull)
                .toList();
        long dead = workers.stream().filter(worker -> !worker.isAlive()).count();
        if (!missing.isEmpty() || dead > 0) {
            return new CpAdoptionView(workers.size(), -1L, -1L, false,
                    "expected " + expected.size() + " participant endpoints, missing=" + missing
                            + ", dead=" + dead);
        }
        if (workers.isEmpty()) {
            return new CpAdoptionView(0, -1L, -1L, false, "no participant reports a geometry");
        }
        Map<String, Long> countBySignature = workers.stream()
                .collect(Collectors.groupingBy(EngineWorkerStatus::geometrySignature,
                        LinkedHashMap::new, Collectors.counting()));
        WorkerStatus first = workers.get(0);
        long prepareAhead = workers.stream()
                .filter(worker -> worker.getCpPrepareEpoch() > worker.getCpEpoch())
                .count();
        boolean settled = countBySignature.size() == 1
                && prepareAhead == 0
                && first.getCpSize() > 0
                && first.getCpEpoch() >= 0
                && "commit".equalsIgnoreCase(first.getCpPhase());
        return new CpAdoptionView(workers.size(),
                settled ? first.getCpSize() : -1L,
                settled ? first.getCpEpoch() : -1L,
                settled,
                workers.size() + " participant workers " + countBySignature
                        + (prepareAhead > 0 ? ", " + prepareAhead + " mid-prepare" : ""));
    }

    /**
     * Adopt a geometry that both the actuator and every live worker agree on, which is
     * the only evidence that resolves the ambiguity a blocked switch left behind.
     * Returns false when the caller no longer owns the decision.
     */
    public boolean reconcileCpSwitch(long observedCpSize, long observedEpoch,
                                     Map<Integer, Integer> composition,
                                     String outcome, String detail) {
        synchronized (cpSwitchLock) {
            if (!cpSwitchBlocked || cpSwitchInProgress
                    || observedCpSize <= 0 || observedEpoch < 0) {
                return false;
            }
            long target = pendingCpSize;
            committedCpSize = observedCpSize;
            committedCpEpoch = observedEpoch;
            publishCpCompositionLocked(composition);
            pendingCpSize = -1L;
            pendingCpEpoch = -1L;
            preSwitchCpSize = -1L;
            preSwitchCpEpoch = -1L;
            cpSwitchBlocked = false;
            cpSwitchParticipants = Set.of();
            cpSoftQuiesceState = CpSoftQuiesceState.idle(cpSoftQuiesceVersion);
            lastCpSwitchError = null;
            lastCpReconcileOutcome = outcome;
            lastCpReconcileDetail = detail;
            lastCpReconcileAtMs = System.currentTimeMillis();
            cpReconcileAttemptCount++;
            cpSwitchLock.notifyAll();
            log.warn("Reconciled blocked CP composition switch, target=CP{} observed=CP{} epoch={} outcome={}, {}",
                    target, observedCpSize, observedEpoch, outcome, detail);
            return true;
        }
    }

    /** Record an inconclusive reconciliation so an operator can see why traffic is still gated. */
    public void noteCpReconcileAttempt(String outcome, String detail) {
        synchronized (cpSwitchLock) {
            lastCpReconcileOutcome = outcome;
            lastCpReconcileDetail = detail;
            lastCpReconcileAtMs = System.currentTimeMillis();
            cpReconcileAttemptCount++;
        }
        log.warn("CP composition switch stays blocked after reconciliation, outcome={}, {}", outcome, detail);
    }

    public Map<String, Object> cpReconcileSnapshot() {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("attempt_count", cpReconcileAttemptCount);
        snapshot.put("last_outcome", lastCpReconcileOutcome);
        snapshot.put("last_detail", lastCpReconcileDetail);
        snapshot.put("last_at_ms", lastCpReconcileAtMs);
        return snapshot;
    }

    private static String geometrySignature(WorkerStatus worker) {
        return "cp" + worker.getCpSize() + "/epoch" + worker.getCpEpoch()
                + "/" + (worker.getCpPhase() == null ? "unknown" : worker.getCpPhase());
    }

    public long getCommittedCpSize() {
        return committedCpSize > 0 ? committedCpSize : inferActiveCpSize();
    }

    public long getCommittedCpEpoch() {
        if (committedCpEpoch >= 0) {
            return committedCpEpoch;
        }
        long activeCp = getCommittedCpSize();
        return cpWorkers()
                .filter(Objects::nonNull)
                .filter(WorkerStatus::isAlive)
                .filter(worker -> worker.getCpSize() == activeCp)
                .filter(worker -> "commit".equalsIgnoreCase(worker.getCpPhase()))
                .mapToLong(WorkerStatus::getCpEpoch)
                .max()
                .orElse(-1L);
    }

    public long getPublishedUniformCpSize() {
        Map<Integer, Integer> composition = publishedCpComposition;
        return composition.size() == 1 ? composition.keySet().iterator().next() : -1L;
    }

    public long inferActiveCpSize() {
        return cpWorkers()
                .filter(Objects::nonNull)
                .filter(WorkerStatus::isAlive)
                .filter(worker -> worker.getCpSize() > 0)
                .collect(Collectors.groupingBy(WorkerStatus::getCpSize, Collectors.counting()))
                .entrySet().stream()
                .max(Comparator.<Map.Entry<Long, Long>>comparingLong(Map.Entry::getValue)
                        .thenComparingLong(Map.Entry::getKey))
                .map(Map.Entry::getKey)
                .orElse(-1L);
    }

    /** The live workers that carry a CP degree, the population a composition is built from. */
    public List<WorkerStatus> liveCpWorkers() {
        return cpWorkers()
                .filter(Objects::nonNull)
                .filter(WorkerStatus::isAlive)
                .filter(worker -> worker.getCpSize() > 0)
                .toList();
    }

    public boolean isCpEpochVisible(long cpSize, long epoch) {
        return isCpEpochVisible(cpSize, epoch, cpSwitchParticipants);
    }

    public boolean isCpEpochVisible(long cpSize, long epoch, Collection<String> participants) {
        CpAdoptionView adoption = cpAdoptionView(participants);
        return adoption.unanimous() && adoption.cpSize() == cpSize && adoption.epoch() == epoch;
    }

    public void initializeCpComposition(Map<Integer, Integer> composition) {
        synchronized (cpSwitchLock) {
            if (publishedCpComposition.isEmpty()) {
                publishedCpComposition = normalizedComposition(composition);
            }
        }
    }

    public boolean isSchedulingPaused() {
        return profileSwitchInProgress || cpSwitchInProgress || cpSwitchBlocked;
    }

    /**
     * A CP transition only redefines the geometry of the roles that carry a CP degree,
     * so decode and vit keep serving the requests that are already past prefill.
     *
     * <p>A PARTICIPANT-SCOPED transition does not pause the CP-carrying roles at all: the
     * endpoints whose geometry is changing are withheld by {@link #filterByGroup}, and the
     * rest of the pool is still serving at a degree it has committed. Pausing the role
     * instead would empty the candidate set for the whole transition, and a rolling walk
     * makes that window LONGER than the whole-pool switch it was meant to replace (312 s
     * per host measured, against 132 s all-at-once) -- so the more gradual the switch, the
     * more requests it sheds. A blocked host remains withheld by the same participant set;
     * it does not make unrelated prefill hosts or decode workers unavailable.
     */
    public boolean isSchedulingPaused(RoleType roleType) {
        if (profileSwitchInProgress) {
            return true;
        }
        if (!cpSwitchInProgress && !cpSwitchBlocked) {
            return false;
        }
        if (!cpSwitchParticipants.isEmpty()) {
            return false;
        }
        return roleType == null || RoleType.PREFILL.equals(roleType) || RoleType.PDFUSION.equals(roleType);
    }

    private Stream<WorkerStatus> cpWorkers() {
        return Stream.of(
                MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap(),
                MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap())
                .flatMap(statusMap -> statusMap.values().stream());
    }

    private Map<String, WorkerStatus> cpWorkerMap() {
        Map<String, WorkerStatus> workers = new LinkedHashMap<>();
        MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().forEach(workers::put);
        MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().forEach(workers::put);
        return workers;
    }

    private void publishCpCompositionLocked(Map<Integer, Integer> composition) {
        Map<Integer, Integer> normalized = normalizedComposition(composition);
        if (!normalized.equals(publishedCpComposition)) {
            publishedCpComposition = normalized;
            compositionEpoch++;
        }
    }

    private static Map<Integer, Integer> normalizedComposition(Map<Integer, Integer> composition) {
        if (composition == null || composition.isEmpty()) {
            return Map.of();
        }
        Map<Integer, Integer> normalized = new TreeMap<>();
        composition.forEach((cp, count) -> {
            if (cp != null && cp > 0 && count != null && count > 0) {
                normalized.put(cp, count);
            }
        });
        return Map.copyOf(normalized);
    }

    private static boolean sameProfile(String left, String right) {
        String normalizedLeft = normalizeProfile(left);
        String normalizedRight = normalizeProfile(right);
        if (normalizedLeft == null) {
            return normalizedRight == null;
        }
        return normalizedLeft.equals(normalizedRight);
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private static String normalizeProfile(String profileName) {
        return profileName == null || profileName.trim().isEmpty() ? null : profileName.trim();
    }

}
