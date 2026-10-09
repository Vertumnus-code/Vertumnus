package org.flexlb.balance.strategy;

import java.util.List;
import java.util.Map;

/**
 * Stateful window policy for choosing a split or merge that reaches an adjacent
 * composition.
 *
 * <p>Each request contributes predicted service time in the capacity profile's
 * common demand currency. The sum produces the aggregate workload demand D(k) of
 * one observation window, normalized to demand seconds per wall second. A
 * transition is admitted by comparing that workload with both the current
 * composition capacity and the capacity after the concrete one-host operation.
 * A split is triggered after demand crosses a configured fraction of the capacity
 * gain between the current and target compositions. A merge is useful only when
 * its target still accommodates the workload with headroom. The condition must
 * persist for several windows before a transition is initiated. The legacy
 * upper/lower fields remain in the configuration for wire/API compatibility, but
 * are deliberately not consulted.
 */
final class CpCompositionPolicy {

    enum Action {
        HOLD,
        SPLIT,
        MERGE
    }

    /**
     * The first two and the final field are retained for configuration/API
     * compatibility with earlier threshold-based builds. Capacity decisions use
     * exact before/after profile comparisons and intentionally ignore them.
     */
    record Config(double upperUtilization,
                  double lowerUtilization,
                  double stableDemandRelativeDelta,
                  double longRequestFractionDelta,
                  int confirmationWindows,
                  int cooldownWindows,
                  int minimumRequestsPerWindow,
                  double mergeCapacityHeadroom,
                  double splitCapacityAlpha,
                  double mergeCapacityMargin,
                  int edgeConfirmationWindows) {
        /** Compatibility constructor that preserves the former uniform confirmation policy. */
        Config(double upperUtilization,
               double lowerUtilization,
               double stableDemandRelativeDelta,
               double longRequestFractionDelta,
               int confirmationWindows,
               int cooldownWindows,
               int minimumRequestsPerWindow,
               double mergeCapacityHeadroom,
               double splitCapacityAlpha,
               double mergeCapacityMargin) {
            this(upperUtilization, lowerUtilization, stableDemandRelativeDelta,
                    longRequestFractionDelta, confirmationWindows, cooldownWindows,
                    minimumRequestsPerWindow, mergeCapacityHeadroom, splitCapacityAlpha,
                    mergeCapacityMargin, confirmationWindows);
        }

        /** Compatibility constructor used by the current controller/test wiring. */
        Config(double upperUtilization,
               double lowerUtilization,
               double stableDemandRelativeDelta,
               double longRequestFractionDelta,
               int confirmationWindows,
               int cooldownWindows,
               int minimumRequestsPerWindow,
               double mergeCapacityHeadroom) {
            this(upperUtilization, lowerUtilization, stableDemandRelativeDelta,
                    longRequestFractionDelta, confirmationWindows, cooldownWindows,
                    minimumRequestsPerWindow, mergeCapacityHeadroom, 0.50, 0.10);
        }

        /** Compatibility constructor used by older controller wiring. */
        Config(double upperUtilization,
               double lowerUtilization,
               double stableDemandRelativeDelta,
               double longRequestFractionDelta,
               int confirmationWindows,
               int cooldownWindows,
               double mergeCapacityHeadroom) {
            this(upperUtilization, lowerUtilization, stableDemandRelativeDelta,
                    longRequestFractionDelta, confirmationWindows, cooldownWindows,
                    1, mergeCapacityHeadroom, 0.50, 0.10);
        }

        Config {
            if (!(upperUtilization > lowerUtilization && lowerUtilization >= 0.0)) {
                throw new IllegalArgumentException("upper utilization must exceed lower utilization");
            }
            if (stableDemandRelativeDelta < 0.0 || longRequestFractionDelta < 0.0) {
                throw new IllegalArgumentException("relative deltas must be non-negative");
            }
            if (confirmationWindows < 1 || cooldownWindows < 0 || minimumRequestsPerWindow < 1) {
                throw new IllegalArgumentException("invalid window count configuration");
            }
            if (edgeConfirmationWindows < confirmationWindows) {
                throw new IllegalArgumentException(
                        "edge confirmation windows must be at least confirmation windows");
            }
            if (!(mergeCapacityHeadroom > 0.0 && mergeCapacityHeadroom <= 1.0)) {
                throw new IllegalArgumentException("merge capacity headroom must be in (0, 1]");
            }
            if (!(splitCapacityAlpha >= 0.0 && splitCapacityAlpha < 1.0)) {
                throw new IllegalArgumentException("split capacity alpha must be in [0, 1)");
            }
            if (!(mergeCapacityMargin >= 0.0 && mergeCapacityMargin < 1.0)) {
                throw new IllegalArgumentException("merge capacity margin must be in [0, 1)");
            }
        }
    }

    record Window(long durationMs,
                  int requestCount,
                  int longRequestCount,
                  double demandSeconds,
                  double backlogDemandSeconds) {
        Window(long durationMs, int requestCount, int longRequestCount,
               double demandSeconds) {
            this(durationMs, requestCount, longRequestCount, demandSeconds, 0.0);
        }

        double demandRate() {
            return durationMs > 0 ? demandSeconds * 1000.0 / durationMs : Double.NaN;
        }

        double admittedDemandSeconds() {
            return Math.max(0.0, demandSeconds - Math.max(0.0, backlogDemandSeconds));
        }

        double longRequestFraction() {
            return requestCount == 0 ? 0.0 : longRequestCount / (double) requestCount;
        }
    }

    record Decision(Action action,
                    String composition,
                    int targetCp,
                    CpCompositionPlanner.Operation operation,
                    String reason,
                    int persistence,
                    double currentUtilization,
                    double longRequestFraction,
                    double longRequestFractionDelta,
                    double workloadDemandRate,
                    double capacityBeforeRate,
                    double capacityAfterRate) {
        static Decision hold(String composition, String reason, double utilization,
                             double longFraction, double longDelta) {
            return hold(composition, reason, utilization, longFraction, longDelta,
                    Double.NaN, Double.NaN, Double.NaN);
        }

        static Decision hold(String composition, String reason, double utilization,
                             double longFraction, double longDelta, double workload,
                             double capacityBefore, double capacityAfter) {
            return new Decision(Action.HOLD, composition, -1, null, reason, 0,
                    utilization, longFraction, longDelta, workload, capacityBefore,
                    capacityAfter);
        }
    }

    private record Signal(Action action, CpCompositionPlanner.Operation operation, String reason,
                          double utilization, double longFraction, double longDelta,
                          double workloadDemandRate, double capacityBeforeRate,
                          double capacityAfterRate) {
    }

    private final Config config;
    private Action pendingAction = Action.HOLD;
    private int pendingTargetCp = -1;
    private String pendingSignalClass = "";
    private int persistence;
    private int cooldownRemaining;
    private double longFractionReference = Double.NaN;
    private double demandRateReference = Double.NaN;

    CpCompositionPolicy(Config config) {
        this.config = config;
    }

    Map<String, Object> configSnapshot() {
        return Map.of(
                "split_capacity_alpha", config.splitCapacityAlpha(),
                "merge_capacity_margin", config.mergeCapacityMargin(),
                "confirmation_windows", config.confirmationWindows(),
                "edge_confirmation_windows", config.edgeConfirmationWindows(),
                "cooldown_windows", config.cooldownWindows(),
                "stable_demand_relative_delta", config.stableDemandRelativeDelta(),
                "long_request_fraction_delta", config.longRequestFractionDelta());
    }

    /**
     * Decides whether the composition holds or moves to an adjacent state.
     *
     * @param splitCandidates eligible split operations, best-ranked first
     * @param mergeCandidates eligible merge operations, best-ranked first
     */
    synchronized Decision evaluate(CpComposition composition, Window window,
                                   CpCapacityProfile profile,
                                   List<CpCompositionPlanner.Operation> splitCandidates,
                                   List<CpCompositionPlanner.Operation> mergeCandidates) {
        String description = composition.describe();
        double longFraction = window.longRequestFraction();
        double currentDemandRate = window.demandRate();
        double currentCapacityRate = composition.profiledCapacityRate(profile);
        double utilization = finitePositive(currentCapacityRate)
                ? currentDemandRate / currentCapacityRate : Double.NaN;

        // The controller must not interpret an idle interval after a replay (or a
        // naturally quiet service interval) as evidence that the fleet should be
        // merged. This is intentionally narrower than a minimum-sample gate: any
        // real admission still contributes its profiled demand, and a window with
        // queued backlog remains actionable even when it has no new admissions.
        if (window.requestCount() == 0 && window.backlogDemandSeconds() <= 0.0) {
            resetPending();
            updateReference(currentDemandRate, longFraction);
            return Decision.hold(description, "empty_window", utilization,
                    longFraction, 0.0, currentDemandRate, currentCapacityRate,
                    Double.NaN);
        }

        // The observation interval and confirmation count already provide temporal
        // stability. Do not gate a capacity decision on an arbitrary request count:
        // a low-rate window can still carry a large request or an admitted backlog.
        if (!finitePositive(currentCapacityRate) || !Double.isFinite(currentDemandRate)) {
            resetPending();
            updateReference(currentDemandRate, longFraction);
            return Decision.hold(description, "missing_current_capacity", utilization, longFraction, 0.0,
                    currentDemandRate, currentCapacityRate, Double.NaN);
        }

        CpCompositionPlanner.Operation splitOperation = bestCapacityIncreasingSplit(
                composition, profile, currentDemandRate, currentCapacityRate, splitCandidates);
        double splitTargetCapacity = splitOperation == null ? Double.NaN
                : applyOperation(composition, splitOperation).profiledCapacityRate(profile);
        double splitTrigger = splitOperation == null ? currentCapacityRate
                : currentCapacityRate + config.splitCapacityAlpha()
                * (splitTargetCapacity - currentCapacityRate);
        boolean splitTriggered = splitOperation != null && currentDemandRate > splitTrigger;
        boolean capacityPressure = currentDemandRate > splitTrigger;
        if (cooldownRemaining > 0) {
            cooldownRemaining--;
            // A cooldown prevents a burst of reversals, but it must not erase a
            // sustained overload signal while a legal split remains.  Keep counting
            // those windows so a second host can be split one step at a time.
            if (!splitTriggered) {
                resetPending();
                updateReference(currentDemandRate, longFraction);
                return Decision.hold(description, "cooldown", utilization, longFraction, 0.0,
                        currentDemandRate, currentCapacityRate, Double.NaN);
            }
        }

        Signal signal = capacitySignal(composition, window, profile, splitOperation,
                splitTargetCapacity, splitTriggered, mergeCandidates, utilization, longFraction);
        // An overloaded composition has priority over the optional long-request
        // preference. If no split can improve capacity, do not let that preference
        // turn an already insufficient fleet into a merge.
        boolean capacityTakesPrecedence = capacityPressure;
        if (signal == null && !capacityTakesPrecedence) {
            signal = longRequestSignal(composition, window, profile, splitCandidates,
                    mergeCandidates, utilization, longFraction, currentDemandRate);
        }
        if (signal == null) {
            double longDelta = referenceDelta(longFraction);
            resetPending();
            updateReference(currentDemandRate, longFraction);
            return Decision.hold(description,
                    capacityTakesPrecedence ? "capacity_action_unavailable" : "within_hysteresis",
                    utilization,
                    longFraction, longDelta, currentDemandRate, currentCapacityRate,
                    Double.NaN);
        }

        String signalClass = signal.reason().startsWith("capacity_") ? "capacity" : "long_fraction";
        int targetCp = signal.operation().targetCp();
        if (pendingAction == signal.action() && pendingTargetCp == targetCp
                && pendingSignalClass.equals(signalClass)) {
            persistence++;
        } else {
            pendingAction = signal.action();
            pendingTargetCp = targetCp;
            pendingSignalClass = signalClass;
            persistence = 1;
        }

        CpComposition targetComposition = applyOperation(composition, signal.operation());
        int requiredConfirmationWindows = targetComposition.vector().size() == 1
                ? config.edgeConfirmationWindows() : config.confirmationWindows();
        if (persistence < requiredConfirmationWindows) {
            return new Decision(Action.HOLD, description, targetCp, signal.operation(),
                    signal.reason() + "_pending", persistence, signal.utilization(),
                    signal.longFraction(), signal.longDelta(), signal.workloadDemandRate(),
                    signal.capacityBeforeRate(), signal.capacityAfterRate());
        }

        Decision decision = new Decision(signal.action(), description, targetCp, signal.operation(),
                signal.reason(), persistence, signal.utilization(), signal.longFraction(),
                signal.longDelta(), signal.workloadDemandRate(), signal.capacityBeforeRate(),
                signal.capacityAfterRate());
        cooldownRemaining = config.cooldownWindows();
        resetPending();
        updateReference(currentDemandRate, longFraction);
        return decision;
    }

    /** Hold the current composition for a full cooldown after a transition failed. */
    synchronized void applyCooldown() {
        cooldownRemaining = Math.max(cooldownRemaining, config.cooldownWindows());
        resetPending();
    }

    /**
     * Startup observations establish the long-request reference but cannot arm a
     * transition. This keeps the resident controller's startup grace behavior while
     * reporting the same common-currency workload/capacity fields as normal windows.
     */
    synchronized Decision holdForStartupGrace(CpComposition composition, Window window,
                                               CpCapacityProfile profile) {
        double demandRate = window.demandRate();
        double capacityRate = composition.profiledCapacityRate(profile);
        double utilization = finitePositive(capacityRate)
                ? demandRate / capacityRate : Double.NaN;
        double longFraction = window.longRequestFraction();
        resetPending();
        updateReference(demandRate, longFraction);
        return Decision.hold(composition.describe(), "startup_grace", utilization,
                longFraction, 0.0, demandRate, capacityRate, Double.NaN);
    }

    /**
     * Keep a partially observed fleet out of the decision path.  A transiently
     * missing host must not be interpreted as a smaller legal composition, and any
     * confirmation accumulated before the gap must be discarded.
     */
    synchronized Decision holdForIncompleteComposition(CpComposition composition, Window window,
                                                        CpCapacityProfile profile) {
        double demandRate = window.demandRate();
        double capacityRate = composition.profiledCapacityRate(profile);
        double utilization = finitePositive(capacityRate)
                ? demandRate / capacityRate : Double.NaN;
        resetPending();
        updateReference(demandRate, window.longRequestFraction());
        return Decision.hold(composition.describe(), "composition_incomplete", utilization,
                window.longRequestFraction(), 0.0, demandRate, capacityRate, Double.NaN);
    }

    private Signal capacitySignal(CpComposition composition, Window window,
                                  CpCapacityProfile profile,
                                  CpCompositionPlanner.Operation splitOperation,
                                  double splitTargetCapacity,
                                  boolean splitTriggered,
                                  List<CpCompositionPlanner.Operation> mergeCandidates,
                                  double utilization, double longFraction) {
        double workload = window.demandRate();
        double currentCapacity = composition.profiledCapacityRate(profile);
        if (splitTriggered) {
            return new Signal(Action.SPLIT, splitOperation, "capacity_insufficient_current",
                    utilization, longFraction, referenceDelta(longFraction), workload,
                    currentCapacity, splitTargetCapacity);
        }
        // Direction check only.  The actual merge admission is based on the
        // candidate's post-merge capacity, so a lower-capacity target cannot be
        // selected merely because it is below the current-capacity boundary.
        if (workload < currentCapacity) {
            CpCompositionPlanner.Operation operation =
                    firstAdmissibleMerge(composition, window, profile, mergeCandidates);
            if (operation != null) {
                double targetCapacity = applyOperation(composition, operation)
                        .profiledCapacityRate(profile);
                return new Signal(Action.MERGE, operation, "capacity_surplus_target_fits",
                        utilization, longFraction, referenceDelta(longFraction), workload,
                        currentCapacity, targetCapacity);
            }
        }
        return null;
    }

    private Signal longRequestSignal(CpComposition composition, Window window,
                                     CpCapacityProfile profile,
                                     List<CpCompositionPlanner.Operation> splitCandidates,
                                     List<CpCompositionPlanner.Operation> mergeCandidates,
                                     double utilization, double longFraction,
                                     double currentDemandRate) {
        if (!Double.isFinite(longFractionReference) || !finitePositive(demandRateReference)) {
            updateReference(currentDemandRate, longFraction);
            return null;
        }
        double demandDelta = relativeDelta(currentDemandRate, demandRateReference);
        if (demandDelta > config.stableDemandRelativeDelta()) {
            return null;
        }

        double longDelta = longFraction - longFractionReference;
        if (longDelta >= config.longRequestFractionDelta()) {
            CpCompositionPlanner.Operation operation =
                    firstAdmissibleMerge(composition, window, profile, mergeCandidates);
            if (operation != null) {
                double targetCapacity = applyOperation(composition, operation)
                        .profiledCapacityRate(profile);
                return new Signal(Action.MERGE, operation, "long_fraction_increased",
                        utilization, longFraction, longDelta, window.demandRate(),
                        composition.profiledCapacityRate(profile), targetCapacity);
            }
        } else if (longDelta <= -config.longRequestFractionDelta()) {
            CpCompositionPlanner.Operation operation = bestCapacityIncreasingSplit(
                    composition, profile, window.demandRate(),
                    composition.profiledCapacityRate(profile), splitCandidates);
            if (operation != null) {
                double targetCapacity = applyOperation(composition, operation)
                        .profiledCapacityRate(profile);
                return new Signal(Action.SPLIT, operation, "long_fraction_decreased",
                        utilization, longFraction, longDelta, window.demandRate(),
                        composition.profiledCapacityRate(profile), targetCapacity);
            }
        }
        return null;
    }

    /**
     * Keeps a split incremental while preferring the candidate that can cover the
     * current workload in one step. If none can cover it, choose the candidate with
     * the largest capacity gain so the next window can continue the walk.
     */
    private CpCompositionPlanner.Operation bestCapacityIncreasingSplit(
            CpComposition composition, CpCapacityProfile profile, double workload,
            double currentCapacity, List<CpCompositionPlanner.Operation> candidates) {
        if (candidates == null) {
            return null;
        }
        CpCompositionPlanner.Operation bestImproving = null;
        double bestGain = Double.NEGATIVE_INFINITY;
        for (CpCompositionPlanner.Operation operation : candidates) {
            double targetCapacity = applyOperation(composition, operation).profiledCapacityRate(profile);
            if (!finitePositive(targetCapacity) || !(targetCapacity > currentCapacity)) {
                continue;
            }
            if (Double.isFinite(workload)
                    && workload <= targetCapacity * (1.0 - config.mergeCapacityMargin())) {
                return operation;
            }
            double gain = targetCapacity - currentCapacity;
            if (gain > bestGain) {
                bestGain = gain;
                bestImproving = operation;
            }
        }
        return bestImproving;
    }

    /**
     * The best-ranked merge whose resulting composition still carries the observed
     * workload. Ranking decides the order; admission decides membership.
     */
    private CpCompositionPlanner.Operation firstAdmissibleMerge(
            CpComposition composition, Window window, CpCapacityProfile profile,
            List<CpCompositionPlanner.Operation> mergeCandidates) {
        if (mergeCandidates == null) {
            return null;
        }
        for (CpCompositionPlanner.Operation operation : mergeCandidates) {
            if (mergeFits(applyOperation(composition, operation), window, profile)) {
                return operation;
            }
        }
        return null;
    }

    private boolean mergeFits(CpComposition target, Window window, CpCapacityProfile profile) {
        double targetCapacityRate = target.profiledCapacityRate(profile);
        return finitePositive(targetCapacityRate) && Double.isFinite(window.demandRate())
                && window.demandRate() < targetCapacityRate * (1.0 - config.mergeCapacityMargin());
    }

    /**
     * Production planner operations always carry their concrete host groups. A
     * few API-level callers provide an abstract operation with no participants;
     * interpret that as the one-worker split or the two-sibling merge encoded by
     * the operation's CP degrees.
     */
    private static CpComposition applyOperation(CpComposition composition,
                                                CpCompositionPlanner.Operation operation) {
        int workers = operation.participants().size();
        if (operation.action() == Action.SPLIT) {
            return composition.split(operation.parentCp(), Math.max(1, workers));
        }
        int siblings = workers >= 2 ? workers : 2;
        if ((siblings & 1) != 0) {
            siblings--;
        }
        return composition.merge(operation.childCp(), Math.max(2, siblings));
    }

    private double referenceDelta(double longFraction) {
        return Double.isFinite(longFractionReference) ? longFraction - longFractionReference : 0.0;
    }

    private void updateReference(double demandRate, double longFraction) {
        if (Double.isFinite(demandRate)) {
            demandRateReference = demandRate;
        }
        longFractionReference = longFraction;
    }

    private void resetPending() {
        pendingAction = Action.HOLD;
        pendingTargetCp = -1;
        pendingSignalClass = "";
        persistence = 0;
    }

    private static boolean finitePositive(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static double relativeDelta(double current, double reference) {
        if (!Double.isFinite(current) || !finitePositive(reference)) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.abs(current - reference) / reference;
    }
}
