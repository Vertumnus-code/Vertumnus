package org.flexlb.balance.strategy;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Divides the serving timeline into consecutive, non-overlapping observation
 * windows, estimates the aggregate workload demand of each one, and moves the
 * worker composition to an adjacent state when the reconfiguration condition has
 * persisted.
 */
@Component
public class CpCompositionController {

    private static final String ENABLED_ENV = "FLEXLB_CP_COMPOSITION_ENABLED";
    private final EngineWorkerStatus engineWorkerStatus;
    private final CpSwitchActuator actuator;
    private final boolean enabled;
    private final boolean policyReady;
    private final long observationWindowMs;
    private final long longRequestTokens;
    private final long switchTimeoutMs;
    private final long statusSyncTimeoutMs;
    private final long reconcileIntervalMs;
    private final long softDrainTimeoutMs;
    private final long softDrainStableMs;
    private final long softDrainPollMs;
    private final long receivedMigrationPollMs;
    private final int expectedGpuCount;
    private final int startupGraceWindows;
    private final List<Integer> supportedCpSizes;
    private final CpCapacityProfile capacityProfile;
    private final CpCompositionInitializer.HistoricalWorkload historicalWorkload;
    private final CpPrefixAccessTracker accessTracker;
    private final CpCompositionPlanner planner;
    private final CpCompositionPolicy policy;
    private final Object windowLock = new Object();
    private final Object scheduledSwitchLock = new Object();
    private final Map<Long, Long> recentlyObserved = new HashMap<>();
    private WindowAccumulator accumulator;
    private ScheduledExecutorService scheduler;
    private volatile CpCompositionPolicy.Window lastWindow;
    private volatile CpCompositionPolicy.Decision lastDecision;
    private volatile CpComposition lastComposition = CpComposition.empty();
    private volatile CpCompositionInitializer.Selection initialSelection;
    private volatile double lastPendingQueueDemandSeconds;
    private volatile ExpectedTransition pendingTransition;
    private volatile long completedObservationWindows;
    private volatile SoftDrainObservation softDrainObservation = SoftDrainObservation.idle();
    private CpReceivedRequestMigrator receivedRequestMigrator;

    @Autowired
    public CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                                   CpSwitchActuator actuator,
                                   CpReceivedRequestMigrator receivedRequestMigrator) {
        this(engineWorkerStatus, actuator,
                readBooleanEnv(ENABLED_ENV, false),
                readLongEnv("FLEXLB_CP_OBSERVATION_WINDOW_MS", 10_000L),
                readLongEnv("FLEXLB_CP_LONG_REQUEST_TOKENS", 16_384L),
                readLongEnv("FLEXLB_CP_SWITCH_TIMEOUT_MS", 120_000L),
                readLongEnv("FLEXLB_CP_STATUS_SYNC_TIMEOUT_MS", 5_000L),
                parseCpSizes(readStringEnv("FLEXLB_CP_SUPPORTED_SIZES", "4,8")),
                CpCapacityProfile.load(System.getenv("FLEXLB_CP_CAPACITY_PROFILE"),
                        System.getenv("FLEXLB_CP_PROFILED_CAPACITY"),
                        readLongEnv("FLEXLB_CP_OBSERVATION_WINDOW_MS", 10_000L)),
                new CpCompositionPolicy.Config(
                        readDoubleEnv("FLEXLB_CP_UPPER_UTILIZATION", 0.85),
                        readDoubleEnv("FLEXLB_CP_LOWER_UTILIZATION", 0.40),
                        readDoubleEnv("FLEXLB_CP_STABLE_DEMAND_RELATIVE_DELTA", 0.10),
                        readDoubleEnv("FLEXLB_CP_LONG_REQUEST_FRACTION_DELTA", 0.10),
                        readIntEnv("FLEXLB_CP_CONFIRMATION_WINDOWS", 3),
                        readIntEnv("FLEXLB_CP_COOLDOWN_WINDOWS", 3),
                        readIntEnv("FLEXLB_CP_MIN_REQUESTS_PER_WINDOW", 1),
                        readDoubleEnv("FLEXLB_CP_MERGE_CAPACITY_HEADROOM", 0.90),
                        readDoubleEnv("FLEXLB_CP_SPLIT_CAPACITY_ALPHA", 0.50),
                        readDoubleEnv("FLEXLB_CP_MERGE_CAPACITY_MARGIN", 0.10),
                        readIntEnv("FLEXLB_CP_EDGE_CONFIRMATION_WINDOWS",
                                readIntEnv("FLEXLB_CP_CONFIRMATION_WINDOWS", 3))),
                CpCompositionInitializer.HistoricalWorkload.load(
                        System.getenv("FLEXLB_CP_HISTORICAL_WORKLOAD")),
                new CpCompositionPlanner.Config(
                        readDoubleEnv("FLEXLB_CP_SPLIT_KV_COEFFICIENT", 1.0),
                        readDoubleEnv("FLEXLB_CP_MERGE_KV_COEFFICIENT", 1.0),
                        readDoubleEnv("FLEXLB_CP_SPLIT_DUPLICATION_MIN_ACCESSES", 2.0)),
                readIntEnv("FLEXLB_CP_EXPECTED_GPU_COUNT", 40),
                readIntEnv("FLEXLB_CP_STARTUP_GRACE_WINDOWS", 2),
                readLongEnv("FLEXLB_CP_SOFT_DRAIN_TIMEOUT_MS", 120_000L),
                readLongEnv("FLEXLB_CP_SOFT_DRAIN_STABLE_MS", 500L),
                readLongEnv("FLEXLB_CP_SOFT_DRAIN_POLL_MS", 100L));
        this.receivedRequestMigrator = Objects.requireNonNull(receivedRequestMigrator);
    }

    CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                            CpSwitchActuator actuator,
                            boolean enabled,
                            long observationWindowMs,
                            long longRequestTokens,
                            long switchTimeoutMs,
                            long statusSyncTimeoutMs,
                            List<Integer> supportedCpSizes,
                            CpCapacityProfile capacityProfile,
                            CpCompositionPolicy.Config policyConfig) {
        this(engineWorkerStatus, actuator, enabled, observationWindowMs, longRequestTokens,
                switchTimeoutMs, statusSyncTimeoutMs, supportedCpSizes, capacityProfile,
                policyConfig, CpCompositionInitializer.HistoricalWorkload.absent(),
                new CpCompositionPlanner.Config(1.0, 1.0, 2.0), 0, 0);
    }

    CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                            CpSwitchActuator actuator,
                            boolean enabled,
                            long observationWindowMs,
                            long longRequestTokens,
                            long switchTimeoutMs,
                            long statusSyncTimeoutMs,
                            List<Integer> supportedCpSizes,
                            CpCapacityProfile capacityProfile,
                            CpCompositionPolicy.Config policyConfig,
                            CpCompositionInitializer.HistoricalWorkload historicalWorkload,
                            CpCompositionPlanner.Config plannerConfig) {
        this(engineWorkerStatus, actuator, enabled, observationWindowMs, longRequestTokens,
                switchTimeoutMs, statusSyncTimeoutMs, supportedCpSizes, capacityProfile,
                policyConfig, historicalWorkload, plannerConfig, 0, 0);
    }

    CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                            CpSwitchActuator actuator,
                            boolean enabled,
                            long observationWindowMs,
                            long longRequestTokens,
                            long switchTimeoutMs,
                            long statusSyncTimeoutMs,
                            List<Integer> supportedCpSizes,
                            CpCapacityProfile capacityProfile,
                            CpCompositionPolicy.Config policyConfig,
                            CpCompositionInitializer.HistoricalWorkload historicalWorkload,
                            CpCompositionPlanner.Config plannerConfig,
                            int expectedGpuCount) {
        this(engineWorkerStatus, actuator, enabled, observationWindowMs, longRequestTokens,
                switchTimeoutMs, statusSyncTimeoutMs, supportedCpSizes, capacityProfile,
                policyConfig, historicalWorkload, plannerConfig, expectedGpuCount, 0);
    }

    CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                            CpSwitchActuator actuator,
                            boolean enabled,
                            long observationWindowMs,
                            long longRequestTokens,
                            long switchTimeoutMs,
                            long statusSyncTimeoutMs,
                            List<Integer> supportedCpSizes,
                            CpCapacityProfile capacityProfile,
                            CpCompositionPolicy.Config policyConfig,
                            CpCompositionInitializer.HistoricalWorkload historicalWorkload,
                            CpCompositionPlanner.Config plannerConfig,
                            int expectedGpuCount,
                            int startupGraceWindows) {
        this(engineWorkerStatus, actuator, enabled, observationWindowMs, longRequestTokens,
                switchTimeoutMs, statusSyncTimeoutMs, supportedCpSizes, capacityProfile,
                policyConfig, historicalWorkload, plannerConfig, expectedGpuCount,
                startupGraceWindows, 5_000L, 0L, 10L);
    }

    CpCompositionController(EngineWorkerStatus engineWorkerStatus,
                            CpSwitchActuator actuator,
                            boolean enabled,
                            long observationWindowMs,
                            long longRequestTokens,
                            long switchTimeoutMs,
                            long statusSyncTimeoutMs,
                            List<Integer> supportedCpSizes,
                            CpCapacityProfile capacityProfile,
                            CpCompositionPolicy.Config policyConfig,
                            CpCompositionInitializer.HistoricalWorkload historicalWorkload,
                            CpCompositionPlanner.Config plannerConfig,
                            int expectedGpuCount,
                            int startupGraceWindows,
                            long softDrainTimeoutMs,
                            long softDrainStableMs,
                            long softDrainPollMs) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.actuator = actuator;
        this.enabled = enabled;
        this.observationWindowMs = Math.max(100L, observationWindowMs);
        this.longRequestTokens = Math.max(1L, longRequestTokens);
        this.switchTimeoutMs = Math.max(1_000L, switchTimeoutMs);
        this.statusSyncTimeoutMs = Math.max(100L, statusSyncTimeoutMs);
        this.reconcileIntervalMs = Math.max(200L,
                readLongEnv("FLEXLB_CP_RECONCILE_INTERVAL_MS", 2_000L));
        this.softDrainTimeoutMs = Math.max(1L, softDrainTimeoutMs);
        this.softDrainStableMs = Math.max(0L, softDrainStableMs);
        this.softDrainPollMs = Math.max(1L, softDrainPollMs);
        this.receivedMigrationPollMs = Math.max(this.softDrainPollMs,
                readLongEnv("FLEXLB_CP_CANCEL_RECEIVED_POLL_MS", 500L));
        this.expectedGpuCount = Math.max(0, expectedGpuCount);
        this.startupGraceWindows = Math.max(0, startupGraceWindows);
        this.supportedCpSizes = List.copyOf(supportedCpSizes);
        this.capacityProfile = capacityProfile;
        this.historicalWorkload = historicalWorkload;
        this.policyReady = enabled && actuator.isConfigured()
                && capacityProfile.covers(this.supportedCpSizes);
        this.accessTracker = new CpPrefixAccessTracker(
                readIntEnv("FLEXLB_CP_PREFIX_TRACKER_MAX_KEYS", 262_144),
                readDoubleEnv("FLEXLB_CP_PREFIX_TRACKER_DECAY", 0.5));
        this.planner = new CpCompositionPlanner(this.supportedCpSizes, accessTracker, plannerConfig);
        this.policy = new CpCompositionPolicy(policyConfig);
        this.accumulator = new WindowAccumulator(System.currentTimeMillis());
    }

    @PostConstruct
    public void start() {
        if (!enabled) {
            Logger.info("CP composition controller is disabled; set {}=true to enable", ENABLED_ENV);
            return;
        }
        if (!actuator.isConfigured()) {
            Logger.error("CP composition controller is enabled but no actuator URL is configured");
            return;
        }
        if (!capacityProfile.covers(supportedCpSizes)) {
            Logger.error("CP composition controller requires profiled capacity for every supported CP size; supported={}, profile={}",
                    supportedCpSizes, capacityProfile.snapshot());
            return;
        }
        capacityProfile.warnOnWindowMismatch(observationWindowMs);
        reportInitialComposition();
        scheduler = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "cp-composition-controller");
            thread.setDaemon(true);
            return thread;
        });
        // A resident switch can outlast several windows. Do not catch up with
        // millisecond samples that inflate demand and consume cooldown instantly.
        scheduler.scheduleWithFixedDelay(this::closeWindowSafely,
                observationWindowMs, observationWindowMs, TimeUnit.MILLISECONDS);
        scheduler.scheduleWithFixedDelay(this::reconcileSafely,
                reconcileIntervalMs, reconcileIntervalMs, TimeUnit.MILLISECONDS);
        Logger.info("CP composition controller started windowMs={} longTokens={} supported={} profile={} reconcileMs={} startupGraceWindows={} expectedGpuCount={}",
                observationWindowMs, longRequestTokens, supportedCpSizes,
                capacityProfile.snapshot(), reconcileIntervalMs, startupGraceWindows,
                expectedGpuCount);
    }

    @PreDestroy
    public void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
        }
    }

    /** Called once after a request has a concrete cache hit and route decision. */
    public void observeRequest(long requestId, long sequenceLength, long prefixHitTokens,
                               long selectedCpSize, Collection<Long> blockCacheKeys) {
        if (!policyReady) {
            return;
        }
        if (selectedCpSize <= 0) {
            Logger.warn("Skip CP workload observation for request {} because its CP degree is unknown",
                    requestId);
            return;
        }
        long now = System.currentTimeMillis();
        synchronized (windowLock) {
            Long previous = recentlyObserved.putIfAbsent(requestId, now);
            if (previous != null) {
                return;
            }
            long hit = Math.max(0L, Math.min(prefixHitTokens, sequenceLength));
            long uncached = Math.max(0L, sequenceLength - hit);
            accumulator.requestCount++;
            if (sequenceLength >= longRequestTokens) {
                accumulator.longRequestCount++;
            }
            accumulator.demandSeconds += CpAwareTtftStrategy.prefillSeconds(
                    selectedCpSize, uncached, hit);
        }
        accessTracker.record(blockCacheKeys);
    }

    public Map<String, Object> snapshot() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("enabled", enabled);
        result.put("ready", policyReady);
        result.put("observation_window_ms", observationWindowMs);
        result.put("long_request_tokens", longRequestTokens);
        result.put("supported_cp_sizes", supportedCpSizes);
        result.put("capacity_profile", capacityProfile.snapshot());
        result.put("capacity_policy", policy.configSnapshot());
        result.put("historical_workload", historicalWorkload.snapshot());
        result.put("initial_composition", initialSelection == null ? null : initialSelection.snapshot());
        result.put("composition", lastComposition.snapshot(capacityProfile));
        result.put("composition_epoch", engineWorkerStatus.getCompositionEpoch());
        result.put("published_cp_composition", engineWorkerStatus.getPublishedCpComposition());
        result.put("prefix_access_keys", accessTracker.size());
        result.put("reconcile_interval_ms", reconcileIntervalMs);
        result.put("soft_drain_timeout_ms", softDrainTimeoutMs);
        result.put("soft_drain_stable_ms", softDrainStableMs);
        result.put("soft_drain_poll_ms", softDrainPollMs);
        result.put("cp_soft_drain", softDrainObservation.snapshot());
        result.put("expected_gpu_count", expectedGpuCount);
        result.put("startup_grace_windows", startupGraceWindows);
        result.put("completed_observation_windows", completedObservationWindows);
        result.put("startup_grace_remaining_windows", Math.max(0L,
                startupGraceWindows - completedObservationWindows));
        result.put("committed_cp_size", engineWorkerStatus.getPublishedUniformCpSize());
        result.put("committed_cp_epoch", engineWorkerStatus.getCompositionEpoch());
        result.put("last_published_participant_cp_size", engineWorkerStatus.getCommittedCpSize());
        result.put("last_published_participant_host_epoch", engineWorkerStatus.getCommittedCpEpoch());
        result.put("cp_switch_in_progress", engineWorkerStatus.isCpSwitchInProgress());
        result.put("cp_switch_blocked", engineWorkerStatus.isCpSwitchBlocked());
        result.put("cp_soft_quiesce", engineWorkerStatus.cpSoftQuiesceSnapshot());
        result.put("last_cp_switch_error", engineWorkerStatus.getLastCpSwitchError());
        result.put("cp_reconcile", engineWorkerStatus.cpReconcileSnapshot());
        result.put("last_window", windowSnapshot(lastWindow));
        result.put("last_decision", decisionSnapshot(lastDecision));
        result.put("pending_queue_demand_seconds", lastPendingQueueDemandSeconds);
        return result;
    }

    /**
     * Operator entry point for a switch that ended ambiguously. It never clears the
     * block on request: the block is only lifted by a settled, epoch-consistent report
     * from every participant endpoint.
     */
    public Map<String, Object> reconcileNow() {
        Map<String, Object> result = new LinkedHashMap<>();
        boolean blocked = engineWorkerStatus.isCpSwitchBlocked();
        result.put("was_blocked", blocked);
        result.put("recovered", blocked && reconcileBlockedSwitch());
        result.put("blocked", engineWorkerStatus.isCpSwitchBlocked());
        result.put("cp_composition", snapshot());
        return result;
    }

    /**
     * Execute an externally scheduled, one-host operation through the normal
     * reconfiguration protocol.  This is an experiment-only decision override: it
     * chooses the time and host, but deliberately reuses the same participant gate,
     * soft drain, resident actuator, adoption check, and manager publication as the
     * online policy.
     */
    public Map<String, Object> executeScheduledSwitch(String host, int targetCp) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("requested_host", host);
        result.put("requested_target_cp", targetCp);
        result.put("controller_enabled", enabled);
        if (enabled) {
            result.put("ok", false);
            result.put("error", "scheduled switching requires the online controller to be disabled");
            return result;
        }
        if (host == null || host.isBlank() || !supportedCpSizes.contains(targetCp)) {
            result.put("ok", false);
            result.put("error", "a non-empty host and supported target CP are required");
            return result;
        }

        synchronized (scheduledSwitchLock) {
            if (engineWorkerStatus.isCpSwitchInProgress()
                    || engineWorkerStatus.isCpSwitchBlocked()
                    || engineWorkerStatus.getCpSoftQuiesceState().active()) {
                result.put("ok", false);
                result.put("error", "another CP transition is active or blocked");
                return result;
            }
            List<WorkerStatus> beforeWorkers = engineWorkerStatus.liveCpWorkers();
            CpComposition before = CpComposition.fromWorkers(beforeWorkers);
            result.put("composition_before", before.snapshot(capacityProfile));
            if (expectedGpuCount > 0 && before.gpuCount() != expectedGpuCount) {
                result.put("ok", false);
                result.put("error", "observed GPU count " + before.gpuCount()
                        + " does not match required " + expectedGpuCount);
                return result;
            }

            List<CpCompositionPlanner.Operation> matches = new ArrayList<>();
            for (CpCompositionPolicy.Action action : List.of(
                    CpCompositionPolicy.Action.SPLIT, CpCompositionPolicy.Action.MERGE)) {
                planner.enumerate(action, before).stream()
                        .filter(operation -> operation.host().equals(host)
                                && operation.targetCp() == targetCp)
                        .forEach(matches::add);
            }
            if (matches.size() != 1) {
                result.put("ok", false);
                result.put("error", "expected exactly one legal adjacent operation, found "
                        + matches.size());
                return result;
            }

            CpCompositionPlanner.Operation operation = matches.get(0);
            CpComposition expected = operation.applyTo(before);
            result.put("operation", operation.snapshot());
            result.put("composition_expected", expected.snapshot(capacityProfile));
            engineWorkerStatus.initializeCpComposition(before.vector());
            lastComposition = before;
            executeSwitch(targetCp, operation);

            CpComposition after = CpComposition.fromWorkers(engineWorkerStatus.liveCpWorkers());
            lastComposition = after;
            boolean success = after.vector().equals(expected.vector())
                    && !engineWorkerStatus.isCpSwitchInProgress()
                    && !engineWorkerStatus.isCpSwitchBlocked()
                    && engineWorkerStatus.getPublishedCpComposition().equals(expected.vector());
            result.put("ok", success);
            result.put("composition_after", after.snapshot(capacityProfile));
            result.put("published_cp_composition",
                    engineWorkerStatus.getPublishedCpComposition());
            result.put("composition_epoch", engineWorkerStatus.getCompositionEpoch());
            result.put("last_cp_switch_error", engineWorkerStatus.getLastCpSwitchError());
            result.put("error", success ? "" : "scheduled switch did not reach and publish the expected composition");
            return result;
        }
    }

    /**
     * Resolve a blocked switch from the exact participant endpoints. The central
     * actuator's status describes a mixed fleet, so it cannot provide one active CP or
     * host-local epoch for this decision. Other prefill hosts may legitimately remain
     * at different CP degrees and host-local epochs.
     *
     * <ul>
     *   <li>all at the target host epoch: the switch did land, publish it and reopen;</li>
     *   <li>all at the pre-switch epoch: keep the host blocked because a delayed status
     *       poll is not proof that an ambiguous POST did not land;</li>
     *   <li>mixed, missing or mid-protocol: stay blocked and record why.</li>
     * </ul>
     */
    boolean reconcileBlockedSwitch() {
        if (!engineWorkerStatus.isCpSwitchBlocked() || engineWorkerStatus.isCpSwitchInProgress()) {
            return false;
        }
        EngineWorkerStatus.CpAdoptionView adoption = engineWorkerStatus.cpAdoptionView();
        if (adoption.workerCount() == 0) {
            engineWorkerStatus.noteCpReconcileAttempt("workers_missing", adoption.detail());
            return false;
        }
        if (!adoption.unanimous()) {
            engineWorkerStatus.noteCpReconcileAttempt("workers_mixed", adoption.detail());
            return false;
        }

        long targetCp = engineWorkerStatus.getPendingCpSize();
        long preSwitchCp = engineWorkerStatus.getPreSwitchCpSize();
        long preSwitchEpoch = engineWorkerStatus.getPreSwitchCpEpoch();
        long pendingEpoch = engineWorkerStatus.getPendingCpEpoch();
        long observedCp = adoption.cpSize();
        ExpectedTransition expected = pendingTransition;
        boolean cancelled = false;
        boolean epochMatches;
        try {
            cancelled = expected != null && pendingEpoch < 0 && observedCp == preSwitchCp
                    && adoption.epoch() == preSwitchEpoch
                    && actuator.confirmsCancelled(expected.host(), Math.toIntExact(targetCp), preSwitchCp, preSwitchEpoch);
            epochMatches = pendingEpoch >= 0 ? adoption.epoch() == pendingEpoch
                    : cancelled || actuator.confirmsRecoveredEpoch(Math.toIntExact(targetCp), adoption.epoch(),
                            preSwitchEpoch, statusSyncTimeoutMs);
        } catch (Exception error) {
            engineWorkerStatus.noteCpReconcileAttempt("actuator_unconfirmed", error.getMessage());
            return false;
        }
        if (targetCp <= 0 || (!cancelled && observedCp != targetCp) || !epochMatches) {
            engineWorkerStatus.noteCpReconcileAttempt("participant_epoch_ambiguous",
                    "target=CP" + targetCp + " preSwitch=CP" + preSwitchCp
                            + "/epoch" + preSwitchEpoch + " pendingEpoch=" + pendingEpoch
                            + ", " + adoption.detail());
            return false;
        }
        String outcome = cancelled ? "switch_cancelled" : "switch_confirmed";
        String detail = "target=CP" + targetCp + " preSwitch=CP" + preSwitchCp
                + "/epoch" + preSwitchEpoch + " pendingEpoch=" + pendingEpoch
                + ", " + adoption.detail();

        CpComposition observedComposition = CpComposition.fromWorkers(
                engineWorkerStatus.liveCpWorkers());
        String validationError = validateObservedTransition(observedComposition,
                Math.toIntExact(observedCp), adoption.epoch(), cancelled);
        if (!validationError.isBlank()) {
            engineWorkerStatus.noteCpReconcileAttempt("composition_validation_failed",
                    validationError);
            return false;
        }
        boolean recovered = engineWorkerStatus.reconcileCpSwitch(observedCp, adoption.epoch(),
                observedComposition.vector(), outcome, detail);
        if (recovered) {
            lastComposition = observedComposition;
            pendingTransition = null;
        }
        return recovered;
    }

    private void reconcileSafely() {
        try {
            if (engineWorkerStatus.isCpSwitchBlocked()) {
                reconcileBlockedSwitch();
            }
        } catch (Throwable throwable) {
            Logger.error("CP composition reconciliation failed", throwable);
        }
    }

    void closeWindowForTest() {
        closeWindow();
    }

    CpCompositionPolicy.Window getLastWindowForTest() {
        return lastWindow;
    }

    CpCompositionPolicy.Decision getLastDecisionForTest() {
        return lastDecision;
    }

    CpCompositionInitializer.Selection getInitialCompositionForTest() {
        reportInitialComposition();
        return initialSelection;
    }

    /**
     * Selects the composition the historical workload calls for and reports how the
     * running fleet compares with it.
     */
    private void reportInitialComposition() {
        CpComposition observed = CpComposition.fromWorkers(engineWorkerStatus.liveCpWorkers());
        lastComposition = observed;
        engineWorkerStatus.initializeCpComposition(observed.vector());
        CpCompositionInitializer.Selection selection = CpCompositionInitializer.select(
                supportedCpSizes, observed.gpuCount(), capacityProfile, historicalWorkload);
        initialSelection = selection;
        if (selection.composition() == null) {
            Logger.info("CP initial composition is undetermined: {}, observed={}",
                    selection.reason(), observed.describe());
            return;
        }
        Logger.info("CP initial composition selected={} reason={} capacitySeconds={} requiredSeconds={} observed={} gpus={}",
                selection.composition().describe(), selection.reason(),
                selection.aggregateCapacitySeconds(), selection.requiredDemandSeconds(),
                observed.describe(), observed.gpuCount());
    }

    private static Map<String, Object> windowSnapshot(CpCompositionPolicy.Window window) {
        if (window == null) {
            return null;
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("duration_ms", window.durationMs());
        result.put("request_count", window.requestCount());
        result.put("long_request_count", window.longRequestCount());
        result.put("long_request_fraction", window.longRequestFraction());
        result.put("demand_seconds", window.demandSeconds());
        result.put("demand_rate", window.demandRate());
        result.put("backlog_demand_seconds", window.backlogDemandSeconds());
        result.put("admitted_demand_seconds", window.admittedDemandSeconds());
        return result;
    }

    private static Map<String, Object> decisionSnapshot(CpCompositionPolicy.Decision decision) {
        if (decision == null) {
            return null;
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("action", decision.action());
        result.put("composition", decision.composition());
        result.put("target_cp", decision.targetCp());
        result.put("operation", decision.operation() == null ? null : decision.operation().snapshot());
        result.put("reason", decision.reason());
        result.put("persistence", decision.persistence());
        result.put("current_utilization", decision.currentUtilization());
        result.put("long_request_fraction", decision.longRequestFraction());
        result.put("long_request_fraction_delta", decision.longRequestFractionDelta());
        result.put("workload_demand_rate", decision.workloadDemandRate());
        result.put("capacity_before_rate", decision.capacityBeforeRate());
        result.put("capacity_after_rate", decision.capacityAfterRate());
        return result;
    }

    private void closeWindowSafely() {
        try {
            closeWindow();
        } catch (Throwable throwable) {
            Logger.error("CP composition observation window failed", throwable);
        }
    }

    private void closeWindow() {
        long now = System.currentTimeMillis();
        CpComposition observedComposition = CpComposition.fromWorkers(engineWorkerStatus.liveCpWorkers());
        double pendingQueueDemandSeconds = estimatePendingQueueDemandSeconds(observedComposition);
        lastPendingQueueDemandSeconds = pendingQueueDemandSeconds;
        CpCompositionPolicy.Window window;
        synchronized (windowLock) {
            long durationMs = Math.max(1L, now - accumulator.startedAtMs);
            double admittedDemandSeconds = accumulator.demandSeconds;
            window = new CpCompositionPolicy.Window(durationMs, accumulator.requestCount,
                    accumulator.longRequestCount, admittedDemandSeconds, 0.0);
            accumulator = new WindowAccumulator(now);
            long retentionMs = Math.max(observationWindowMs * 10L, switchTimeoutMs * 2L);
            recentlyObserved.entrySet().removeIf(entry -> now - entry.getValue() >= retentionMs);
        }
        lastWindow = window;
        long completedWindow = ++completedObservationWindows;

        if (engineWorkerStatus.isCpSwitchBlocked()) {
            accessTracker.decay();
            reconcileBlockedSwitch();
            return;
        }

        CpComposition composition = observedComposition;
        lastComposition = composition;
        // A host disappearing from status must not turn the remaining workers into a
        // seemingly valid smaller composition.  During an already committed resident
        // switch the participant may be temporarily absent; outside that protocol,
        // hold and clear confirmation until the expected GPU topology returns.
        if (expectedGpuCount > 0 && composition.gpuCount() != expectedGpuCount
                && !engineWorkerStatus.isCpSwitchInProgress()
                && !engineWorkerStatus.getCpSoftQuiesceState().active()) {
            CpCompositionPolicy.Decision incomplete = policy.holdForIncompleteComposition(
                    composition, window, capacityProfile);
            lastDecision = incomplete;
            accessTracker.decay();
            Logger.warn("CP composition incomplete: observed={} gpus={} expectedGpuCount={}, holding decisions",
                    composition.describe(), composition.gpuCount(), expectedGpuCount);
            return;
        }
        List<CpCompositionPlanner.Operation> splitCandidates =
                onlineCandidates(CpCompositionPolicy.Action.SPLIT, composition);
        List<CpCompositionPlanner.Operation> mergeCandidates =
                onlineCandidates(CpCompositionPolicy.Action.MERGE, composition);
        CpCompositionPolicy.Decision decision = completedWindow <= startupGraceWindows
                ? policy.holdForStartupGrace(composition, window, capacityProfile)
                : policy.evaluate(composition, window, capacityProfile,
                        splitCandidates, mergeCandidates);
        lastDecision = decision;
        accessTracker.decay();
        Logger.info("CP composition window composition={} gpus={} requests={} longFraction={} demandSeconds={} admittedDemandSeconds={} backlogDemandSeconds={} demandRate={} capacityRate={} splitCandidates={} mergeCandidates={} decision={} target=CP{} operation={} reason={} persistence={} workloadRate={} capacityBeforeRate={} capacityAfterRate={}",
                composition.describe(), composition.gpuCount(), window.requestCount(),
                window.longRequestFraction(), window.demandSeconds(), window.admittedDemandSeconds(),
                window.backlogDemandSeconds(), window.demandRate(),
                composition.profiledCapacityRate(capacityProfile),
                splitCandidates.size(), mergeCandidates.size(), decision.action(),
                decision.targetCp(),
                decision.operation() == null ? "none" : decision.operation().id(),
                decision.reason(), decision.persistence(), decision.workloadDemandRate(),
                decision.capacityBeforeRate(), decision.capacityAfterRate());
        if (decision.operation() != null) {
            List<CpCompositionPlanner.Operation> ranked =
                    decision.operation().action() == CpCompositionPolicy.Action.SPLIT
                            ? splitCandidates : mergeCandidates;
            Logger.info("CP operation ranking action={} chosen={} ranked={}",
                    decision.operation().action(), decision.operation().id(), ranked.stream()
                            .map(candidate -> candidate.host()
                                    + ":drain=" + candidate.drainSeconds()
                                    + ":kv=" + candidate.kvTraffic()
                                    + ":kvEstimated=" + candidate.kvTrafficEstimated())
                            .toList());
        }
        if (decision.action() == CpCompositionPolicy.Action.HOLD) {
            clearUnconfirmedSoftQuiesce(decision);
            return;
        }
        executeSwitch(decision);
    }

    /**
     * Online reconfiguration may not remove the last worker at a supported CP degree.
     * Explicit scheduled switches remain unrestricted so setup and profiling can still
     * construct homogeneous fleets while the online controller is disabled.
     */
    private List<CpCompositionPlanner.Operation> onlineCandidates(
            CpCompositionPolicy.Action action, CpComposition composition) {
        boolean currentIsAdmissible = retainsAllSupportedDegrees(composition);
        return planner.enumerate(action, composition).stream()
                .filter(operation -> onlineTransitionIsAdmissible(
                        currentIsAdmissible, operation.applyTo(composition)))
                .toList();
    }

    boolean onlineTransitionIsAdmissible(CpComposition before, CpComposition after) {
        return onlineTransitionIsAdmissible(retainsAllSupportedDegrees(before), after);
    }

    private boolean onlineTransitionIsAdmissible(
            boolean currentIsAdmissible, CpComposition after) {
        return !currentIsAdmissible || retainsAllSupportedDegrees(after);
    }

    private boolean retainsAllSupportedDegrees(CpComposition composition) {
        return supportedCpSizes.stream().allMatch(cp -> composition.workerCount(cp) >= 1);
    }

    /**
     * Estimate work already admitted to CP workers but not yet removed from their
     * status queues. A CP request is reported by multiple ranks, so request ids are
     * deduplicated together with the degree of the worker currently serving it.
     */
    private double estimatePendingQueueDemandSeconds(CpComposition composition) {
        if (composition.isEmpty()) {
            return 0.0;
        }
        Map<Long, PendingTask> pending = new HashMap<>();
        for (WorkerStatus worker : engineWorkerStatus.liveCpWorkers()) {
            long cpSize = worker.getCpSize() > 0 ? worker.getCpSize() : worker.getTpSize();
            if (cpSize <= 0) {
                continue;
            }
            if (worker.getLocalTaskMap() != null) {
                worker.getLocalTaskMap().values().forEach(task -> addPendingTask(pending, task, cpSize));
            }
            if (worker.getWaitingTaskList() != null) {
                worker.getWaitingTaskList().values().forEach(task -> addPendingTask(pending, task, cpSize));
            }
            if (worker.getRunningTaskList() != null) {
                worker.getRunningTaskList().values().forEach(task -> addPendingTask(pending, task, cpSize));
            }
        }
        double total = 0.0;
        for (PendingTask pendingTask : pending.values()) {
            TaskInfo task = pendingTask.task();
            if (task == null || task.getInputLength() <= 0) {
                continue;
            }
            long input = task.getInputLength();
            long hit = Math.max(0L, Math.min(task.getPrefixLength(), input));
            total += CpAwareTtftStrategy.prefillSeconds(
                    pendingTask.cpSize(), input - hit, hit);
        }
        return Double.isFinite(total) && total > 0.0 ? total : 0.0;
    }

    private static void addPendingTask(Map<Long, PendingTask> pending, TaskInfo task, long cpSize) {
        if (task != null && task.getRequestId() > 0L) {
            pending.putIfAbsent(task.getRequestId(), new PendingTask(task, cpSize));
        }
    }

    private record PendingTask(TaskInfo task, long cpSize) {
    }

    private void clearUnconfirmedSoftQuiesce(CpCompositionPolicy.Decision decision) {
        // Persistence is part of admission control: a worker remains schedulable until
        // the policy has emitted a final SPLIT/MERGE decision. Starting quiesce for a
        // *_pending signal would remove an entire resident host for one observation
        // window even if the signal disappears. executeSwitch establishes the gate,
        // reroutes queued work, and drains the selected host after confirmation.
        if (engineWorkerStatus.cancelCpSoftQuiesce("decision=" + decision.reason())) {
            softDrainObservation = softDrainObservation.finished(
                    "cancelled", false, softDrainObservation.waitMs(),
                    softDrainObservation.lastCount(), softDrainObservation.endpointCounts(),
                    softDrainObservation.missingEndpoints());
        }
    }

    private void executeSwitch(CpCompositionPolicy.Decision decision) {
        executeSwitch(decision.targetCp(), decision.operation());
    }

    private void executeSwitch(int targetCp, CpCompositionPlanner.Operation operation) {
        if (operation == null) {
            engineWorkerStatus.cancelCpSoftQuiesce("final decision has no operation");
            Logger.error("Refuse CP{} switch because the policy named no host-wide operation", targetCp);
            policy.applyCooldown();
            return;
        }
        // The endpoints this operation touches, so admission stays open for the rest of
        // the pool while the actuator walks them. The planner already names them; passing
        // an empty set here would pause the whole PREFILL role for the length of the walk.
        // Keyed off the endpoints themselves, not WorkerGroup#id: a group id degrades to
        // "ip:port+N" once a group spans more than one endpoint, which matches no key in
        // the status map and would silently withhold nothing.
        List<String> participants = operationParticipants(operation);
        List<String> participantHosts = operationHosts(operation);
        if (participants.isEmpty() || participantHosts.size() != 1) {
            engineWorkerStatus.cancelCpSoftQuiesce("invalid final operation");
            Logger.error("Refuse CP{} switch because one resident host is required, participants={} hosts={}",
                    targetCp, participants, participantHosts);
            policy.applyCooldown();
            return;
        }
        List<WorkerStatus> beforeWorkers = engineWorkerStatus.liveCpWorkers();
        CpComposition before = CpComposition.fromWorkers(beforeWorkers);
        CpComposition expected = operation.applyTo(before);
        String participantHost = participantHosts.get(0);
        Set<String> participantSet = Set.copyOf(participants);
        Set<String> actualHostEndpoints = endpointsOnHost(beforeWorkers, participantHost);
        if (!participantSet.equals(actualHostEndpoints)) {
            engineWorkerStatus.cancelCpSoftQuiesce("final operation no longer covers host");
            Logger.error("Refuse CP{} switch because operation does not cover the whole resident host {}: expectedEndpoints={} operationEndpoints={}",
                    targetCp, participantHost, actualHostEndpoints, participantSet);
            policy.applyCooldown();
            return;
        }
        if (expected.gpuCount() != before.gpuCount()
                || (expectedGpuCount > 0 && before.gpuCount() != expectedGpuCount)) {
            engineWorkerStatus.cancelCpSoftQuiesce("final GPU topology validation failed");
            Logger.error("Refuse CP{} switch because the observed GPU topology is invalid: before={} expected={} requiredGpuCount={}",
                    targetCp, before.describe(), expected.describe(), expectedGpuCount);
            policy.applyCooldown();
            return;
        }
        EngineWorkerStatus.CpSoftQuiesceState soft = engineWorkerStatus.getCpSoftQuiesceState();
        boolean matchingSoft = soft.active()
                && soft.targetCpSize() == targetCp
                && soft.host().equals(participantHost)
                && soft.operationId().equals(operation.id())
                && soft.participants().equals(participantSet);
        if (soft.active() && !matchingSoft) {
            engineWorkerStatus.cancelCpSoftQuiesce("final operation changed to " + operation.id());
            policy.applyCooldown();
            Logger.error("Refuse CP{} switch because final operation {} does not match active soft quiesce {}",
                    targetCp, operation.id(), soft.operationId());
            return;
        }
        if (!matchingSoft && !engineWorkerStatus.beginCpSoftQuiesce(
                targetCp, participantSet, participantHost, operation.id())) {
            Logger.warn("Skip CP{} switch because its participant gate could not be established",
                    targetCp);
            policy.applyCooldown();
            return;
        }
        if (!waitForSoftDrain(targetCp, participantHost, operation.id(), participantSet)) {
            return;
        }
        boolean switchStarted = engineWorkerStatus.promoteCpSoftQuiesce(
                targetCp, participantSet, participantHost, operation.id());
        if (!switchStarted) {
            engineWorkerStatus.cancelCpSoftQuiesce("could not promote drained operation");
            softDrainObservation = softDrainObservation.finished(
                    "promotion_failed", false, softDrainObservation.waitMs(),
                    softDrainObservation.lastCount(), softDrainObservation.endpointCounts(),
                    softDrainObservation.missingEndpoints());
            Logger.warn("Skip CP{} switch because the drained participant gate could not be promoted",
                    targetCp);
            policy.applyCooldown();
            return;
        }
        pendingTransition = ExpectedTransition.capture(participantHost, participantSet,
                before, expected, beforeWorkers);
        try {
            Logger.info("CP composition transition {} participants={} drainSeconds={} kvTraffic={} kvTrafficEstimated={}",
                    operation.id(),
                    operation.participants().stream().map(CpComposition.WorkerGroup::id).toList(),
                    operation.drainSeconds(), operation.kvTraffic(), operation.kvTrafficEstimated());
            CpSwitchActuator.Result result =
                    actuator.switchTo(targetCp, switchTimeoutMs, participantHosts);
            boolean targetCommitted = result.committed() && result.activeCp() == targetCp
                    && result.epoch() >= 0;
            if (!targetCommitted) {
                engineWorkerStatus.abortCpSwitch(nonBlank(result.error(),
                        "actuator did not commit the target geometry"));
                pendingTransition = null;
                return;
            }

            engineWorkerStatus.markCpSwitchCommitted(targetCp, result.epoch());
            if (!result.confirms(participantHost, targetCp)) {
                blockAndReconcile("actuator commit response did not confirm requested host "
                        + participantHost + " at CP" + targetCp + ": participant="
                        + result.participantHost() + " composition=" + result.composition());
                return;
            }
            if (!result.success()) {
                blockAndReconcile(nonBlank(result.error(),
                        "target committed but rank adoption was not verified"));
                return;
            }
            if (!waitForManagerEpoch(targetCp, result.epoch(), participants)) {
                blockAndReconcile("worker status did not expose CP" + targetCp
                        + " epoch " + result.epoch() + " within " + statusSyncTimeoutMs + "ms");
                return;
            }
            CpComposition observed = CpComposition.fromWorkers(engineWorkerStatus.liveCpWorkers());
            String validationError = validateObservedTransition(observed, targetCp, result.epoch());
            if (!validationError.isBlank()) {
                blockAndReconcile(validationError);
                return;
            }
            engineWorkerStatus.finishCpSwitch(targetCp, result.epoch(), observed.vector());
            lastComposition = observed;
            pendingTransition = null;
        } catch (Exception e) {
            // Once the POST has been sent, a timeout is ambiguous: the workers may
            // have committed even though FlexLB never received the response. Do not
            // reopen admission under the old geometry without an authoritative read.
            blockAndReconcile(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    /**
     * Close admission on the ambiguity, then try to resolve it immediately. The
     * periodic tick retries whatever this first attempt cannot settle.
     */
    private void blockAndReconcile(String error) {
        engineWorkerStatus.blockCpSwitch(error);
        reconcileBlockedSwitch();
    }

    private String validateObservedTransition(CpComposition observed, int targetCp, long epoch) {
        return validateObservedTransition(observed, targetCp, epoch, false);
    }

    private String validateObservedTransition(CpComposition observed, int targetCp, long epoch, boolean cancelled) {
        ExpectedTransition expected = pendingTransition;
        if (expected == null) {
            return "no expected host-wide transition is recorded";
        }
        List<WorkerStatus> workers = engineWorkerStatus.liveCpWorkers();
        var expectedVector = cancelled ? expected.beforeVector() : expected.expectedVector();
        if (!observed.vector().equals(expectedVector)) {
            return "observed composition " + observed.vector() + " does not match expected "
                    + expectedVector;
        }
        if (observed.gpuCount() != expected.gpuCount()
                || (expectedGpuCount > 0 && observed.gpuCount() != expectedGpuCount)) {
            return "observed GPU count " + observed.gpuCount() + " does not match baseline "
                    + expected.gpuCount() + " and required " + expectedGpuCount;
        }
        Set<String> hostEndpoints = endpointsOnHost(workers, expected.host());
        if (!hostEndpoints.equals(expected.participants())) {
            return "participant host " + expected.host() + " endpoints changed: expected="
                    + expected.participants() + " observed=" + hostEndpoints;
        }
        EngineWorkerStatus.CpAdoptionView adoption =
                engineWorkerStatus.cpAdoptionView(expected.participants());
        if (!adoption.unanimous() || adoption.cpSize() != targetCp || adoption.epoch() != epoch) {
            return "participant host " + expected.host() + " did not adopt CP" + targetCp
                    + " epoch " + epoch + ": " + adoption.detail();
        }
        Map<String, String> nonParticipants = nonParticipantGeometry(workers, expected.host());
        if (!nonParticipants.equals(expected.nonParticipantGeometry())) {
            return "non-participant topology changed: expected="
                    + expected.nonParticipantGeometry() + " observed=" + nonParticipants;
        }
        return "";
    }

    private static Set<String> endpointsOnHost(Collection<WorkerStatus> workers, String host) {
        return workers.stream()
                .filter(worker -> Objects.equals(host, worker.getCpSwitchDomain()))
                .map(WorkerStatus::getIpPort)
                .filter(Objects::nonNull)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    private static Map<String, String> nonParticipantGeometry(
            Collection<WorkerStatus> workers, String participantHost) {
        Map<String, String> result = new java.util.TreeMap<>();
        workers.stream()
                .filter(worker -> !Objects.equals(participantHost, worker.getCpSwitchDomain()))
                .forEach(worker -> result.put(worker.getIpPort(), geometrySignature(worker)));
        return Map.copyOf(result);
    }

    private static String geometrySignature(WorkerStatus worker) {
        return "host=" + worker.getIp() + "/cp=" + worker.getCpSize()
                + "/epoch=" + worker.getCpEpoch()
                + "/prepare=" + worker.getCpPrepareEpoch()
                + "/phase=" + worker.getCpPhase()
                + "/world=" + worker.getWorldSize()
                + "/localWorld=" + worker.getLocalWorldSize()
                + "/ranksPerEndpoint=" + worker.getCpRanksPerEndpoint();
    }

    private record ExpectedTransition(String host,
                                      Set<String> participants,
                                      Map<Integer, Integer> beforeVector,
                                      Map<Integer, Integer> expectedVector,
                                      int gpuCount,
                                      Map<String, String> nonParticipantGeometry) {

        private static ExpectedTransition capture(String host, Set<String> participants,
                                                  CpComposition before, CpComposition expected,
                                                  Collection<WorkerStatus> workers) {
            return new ExpectedTransition(host, Set.copyOf(participants),
                    Map.copyOf(before.vector()), Map.copyOf(expected.vector()), before.gpuCount(),
                    CpCompositionController.nonParticipantGeometry(workers, host));
        }
    }

    private boolean waitForSoftDrain(int targetCp, String host, String operationId,
                                     Set<String> participants) {
        long startedNs = System.nanoTime();
        long zeroSinceNs = -1L;
        long lastMigrationNs = Long.MIN_VALUE;
        long lastLoggedCount = Long.MIN_VALUE;
        Logger.info("Wait for CP soft drain operation={} host={} target=CP{} participants={} timeoutMs={} stableMs={} pollMs={}",
                operationId, host, targetCp, participants, softDrainTimeoutMs,
                softDrainStableMs, softDrainPollMs);
        while (true) {
            EngineWorkerStatus.CpSoftQuiesceState gate =
                    engineWorkerStatus.getCpSoftQuiesceState();
            if (!gate.active() || gate.targetCpSize() != targetCp
                    || !gate.host().equals(host)
                    || !gate.operationId().equals(operationId)
                    || !gate.participants().equals(participants)) {
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedNs);
                softDrainObservation = new SoftDrainObservation("gate_changed", false,
                        host, operationId, targetCp, participants, elapsedMs, -1L,
                        softDrainTimeoutMs, softDrainStableMs, Map.of(), Set.of());
                Logger.error("Abort CP soft drain because participant gate changed operation={} host={} waitMs={}",
                        operationId, host, elapsedMs);
                policy.applyCooldown();
                return false;
            }

            long beforeSnapshotNs = System.nanoTime();
            if (lastMigrationNs == Long.MIN_VALUE
                    || TimeUnit.NANOSECONDS.toMillis(beforeSnapshotNs - lastMigrationNs)
                    >= receivedMigrationPollMs) {
                migrateReceived(gate);
                lastMigrationNs = beforeSnapshotNs;
            }

            EngineWorkerStatus.CpParticipantInflightSnapshot inflight =
                    engineWorkerStatus.cpParticipantInflightSnapshot(participants);
            long nowNs = System.nanoTime();
            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(nowNs - startedNs);
            if (inflight.drained()) {
                if (zeroSinceNs < 0L) {
                    zeroSinceNs = nowNs;
                }
            } else {
                zeroSinceNs = -1L;
            }
            softDrainObservation = new SoftDrainObservation("waiting", false,
                    host, operationId, targetCp, participants, elapsedMs, inflight.total(),
                    softDrainTimeoutMs, softDrainStableMs, inflight.endpointCounts(),
                    inflight.missingEndpoints());
            if (lastLoggedCount != inflight.total() || !inflight.missingEndpoints().isEmpty()) {
                Logger.info("CP soft drain progress operation={} host={} waitMs={} inflight={} endpointCounts={} missing={}",
                        operationId, host, elapsedMs, inflight.total(),
                        inflight.endpointCounts(), inflight.missingEndpoints());
                lastLoggedCount = inflight.total();
            }

            long stableForMs = zeroSinceNs < 0L ? 0L
                    : TimeUnit.NANOSECONDS.toMillis(nowNs - zeroSinceNs);
            if (inflight.drained() && stableForMs >= softDrainStableMs) {
                softDrainObservation = new SoftDrainObservation("drained", false,
                        host, operationId, targetCp, participants, elapsedMs, 0L,
                        softDrainTimeoutMs, softDrainStableMs, inflight.endpointCounts(), Set.of());
                Logger.info("CP soft drain complete operation={} host={} waitMs={} stableMs={}",
                        operationId, host, elapsedMs, stableForMs);
                return true;
            }
            if (elapsedMs >= softDrainTimeoutMs) {
                engineWorkerStatus.cancelCpSoftQuiesce("soft drain timeout after " + elapsedMs + "ms");
                softDrainObservation = new SoftDrainObservation("timed_out", true,
                        host, operationId, targetCp, participants, elapsedMs, inflight.total(),
                        softDrainTimeoutMs, softDrainStableMs, inflight.endpointCounts(),
                        inflight.missingEndpoints());
                policy.applyCooldown();
                Logger.error("CP soft drain timed out operation={} host={} waitMs={} inflight={} endpointCounts={} missing={}; actuator was not called",
                        operationId, host, elapsedMs, inflight.total(),
                        inflight.endpointCounts(), inflight.missingEndpoints());
                return false;
            }
            try {
                Thread.sleep(Math.min(softDrainPollMs,
                        Math.max(1L, softDrainTimeoutMs - elapsedMs)));
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                engineWorkerStatus.cancelCpSoftQuiesce("soft drain interrupted");
                softDrainObservation = new SoftDrainObservation("interrupted", false,
                        host, operationId, targetCp, participants, elapsedMs, inflight.total(),
                        softDrainTimeoutMs, softDrainStableMs, inflight.endpointCounts(),
                        inflight.missingEndpoints());
                policy.applyCooldown();
                Logger.error("CP soft drain interrupted operation={} host={} waitMs={}",
                        operationId, host, elapsedMs);
                return false;
            }
        }
    }

    private void migrateReceived(EngineWorkerStatus.CpSoftQuiesceState gate) {
        if (receivedRequestMigrator == null || gate == null || !gate.active()) {
            return;
        }
        try {
            CpReceivedRequestMigrator.MigrationSummary summary =
                    receivedRequestMigrator.migrate(gate.participants(), gate.version());
            if (summary.didWork()) {
                Logger.info("CP soft drain reroute operation={} host={} version={} candidates={} cancelled={} alreadyCancelled={} becameActive={} notFound={} skipped={} errors={}",
                        gate.operationId(), gate.host(), gate.version(), summary.candidates(),
                        summary.cancelled(), summary.alreadyCancelled(), summary.becameActive(),
                        summary.notFound(), summary.skipped(), summary.errors());
            }
        } catch (RuntimeException error) {
            // Keep the participant gate closed. The regular drain fence will either
            // observe the requests finish or time out without invoking the actuator.
            Logger.error("Could not migrate RECEIVED requests for CP soft drain operation={} host={} version={}",
                    gate.operationId(), gate.host(), gate.version(), error);
        }
    }

    private record SoftDrainObservation(String status,
                                        boolean timedOut,
                                        String host,
                                        String operationId,
                                        int targetCp,
                                        Set<String> participants,
                                        long waitMs,
                                        long lastCount,
                                        long timeoutMs,
                                        long stableMs,
                                        Map<String, Integer> endpointCounts,
                                        Set<String> missingEndpoints) {

        private SoftDrainObservation {
            participants = participants == null ? Set.of() : Set.copyOf(participants);
            endpointCounts = endpointCounts == null ? Map.of() : Map.copyOf(endpointCounts);
            missingEndpoints = missingEndpoints == null ? Set.of() : Set.copyOf(missingEndpoints);
        }

        private static SoftDrainObservation idle() {
            return new SoftDrainObservation("idle", false, "", "", -1, Set.of(),
                    0L, 0L, 0L, 0L, Map.of(), Set.of());
        }

        private static SoftDrainObservation pending(String host, String operationId,
                                                    int targetCp, Set<String> participants,
                                                    long timeoutMs, long stableMs) {
            return new SoftDrainObservation("pending", false, host, operationId, targetCp,
                    participants, 0L, -1L, timeoutMs, stableMs, Map.of(), Set.of());
        }

        private SoftDrainObservation finished(String status, boolean timedOut, long waitMs,
                                              long lastCount,
                                              Map<String, Integer> endpointCounts,
                                              Set<String> missingEndpoints) {
            return new SoftDrainObservation(status, timedOut, host, operationId, targetCp,
                    participants, waitMs, lastCount, timeoutMs, stableMs,
                    endpointCounts, missingEndpoints);
        }

        private Map<String, Object> snapshot() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("status", status);
            result.put("timed_out", timedOut);
            result.put("host", host);
            result.put("operation", operationId);
            result.put("target_cp", targetCp);
            result.put("participants", participants.stream().sorted().toList());
            result.put("wait_ms", waitMs);
            result.put("last_count", lastCount);
            result.put("timeout_ms", timeoutMs);
            result.put("stable_ms", stableMs);
            result.put("endpoint_counts", new java.util.TreeMap<>(endpointCounts));
            result.put("missing_endpoints", missingEndpoints.stream().sorted().toList());
            return result;
        }
    }

    private boolean waitForManagerEpoch(int targetCp, long epoch,
                                        Collection<String> participants) throws InterruptedException {
        if (statusSyncTimeoutMs == 0L) {
            return true;
        }
        long deadline = System.currentTimeMillis() + statusSyncTimeoutMs;
        while (System.currentTimeMillis() <= deadline) {
            if (engineWorkerStatus.isCpEpochVisible(targetCp, epoch, participants)) {
                return true;
            }
            Thread.sleep(Math.min(100L, Math.max(1L, deadline - System.currentTimeMillis())));
        }
        return false;
    }

    private static String nonBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private static List<String> operationParticipants(CpCompositionPlanner.Operation operation) {
        return operation.participants().stream()
                .flatMap(group -> group.endpoints().stream())
                .map(WorkerStatus::getIpPort)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
    }

    private static List<String> operationHosts(CpCompositionPlanner.Operation operation) {
        return operation.participants().stream()
                .map(CpComposition.WorkerGroup::host)
                .distinct()
                .toList();
    }

    static List<Integer> parseCpSizes(String raw) {
        List<Integer> values = new ArrayList<>();
        for (String item : raw.split(",")) {
            int cp = Integer.parseInt(item.trim());
            if (cp <= 0) {
                throw new IllegalArgumentException("CP size must be positive: " + item);
            }
            values.add(cp);
        }
        return values.stream().distinct().sorted().toList();
    }

    static Map<Integer, Double> parseCapacityRates(String raw) {
        if (raw == null || raw.isBlank()) {
            return Map.of();
        }
        Map<Integer, Double> values = new LinkedHashMap<>();
        for (String item : raw.split(",")) {
            String[] pair = item.trim().split(":", 2);
            if (pair.length != 2) {
                throw new IllegalArgumentException("invalid CP capacity entry: " + item);
            }
            int cp = Integer.parseInt(pair[0].trim());
            double rate = Double.parseDouble(pair[1].trim());
            if (cp <= 0 || !Double.isFinite(rate) || rate <= 0.0) {
                throw new IllegalArgumentException("invalid CP capacity entry: " + item);
            }
            values.put(cp, rate);
        }
        return values;
    }

    private static boolean readBooleanEnv(String name, boolean defaultValue) {
        String raw = System.getenv(name);
        return raw == null || raw.isBlank() ? defaultValue : Boolean.parseBoolean(raw.trim());
    }

    private static String readStringEnv(String name, String defaultValue) {
        String raw = System.getenv(name);
        return raw == null || raw.isBlank() ? defaultValue : raw.trim();
    }

    private static int readIntEnv(String name, int defaultValue) {
        return Math.toIntExact(readLongEnv(name, defaultValue));
    }

    private static long readLongEnv(String name, long defaultValue) {
        String raw = System.getenv(name);
        try {
            return raw == null || raw.isBlank() ? defaultValue : Long.parseLong(raw.trim());
        } catch (NumberFormatException e) {
            Logger.warn("Invalid {}={}, fallback to {}", name, raw, defaultValue);
            return defaultValue;
        }
    }

    private static double readDoubleEnv(String name, double defaultValue) {
        String raw = System.getenv(name);
        try {
            return raw == null || raw.isBlank() ? defaultValue : Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            Logger.warn("Invalid {}={}, fallback to {}", name, raw, defaultValue);
            return defaultValue;
        }
    }

    private static final class WindowAccumulator {
        private final long startedAtMs;
        private int requestCount;
        private int longRequestCount;
        private double demandSeconds;

        private WindowAccumulator(long startedAtMs) {
            this.startedAtMs = startedAtMs;
        }
    }
}
