package org.flexlb.balance.strategy;

import java.util.List;
import java.util.Map;

/** Executes the already-initialized resident-worker CP epoch protocol. */
public interface CpSwitchActuator {

    record Result(boolean success,
                  boolean committed,
                  int activeCp,
                  long epoch,
                  String error,
                  String responseBody,
                  String participantHost,
                  Map<String, Integer> composition) {
        public Result {
            participantHost = participantHost == null ? "" : participantHost;
            composition = composition == null ? Map.of() : Map.copyOf(composition);
        }

        public Result(boolean success, boolean committed, int activeCp, long epoch,
                      String error, String responseBody) {
            this(success, committed, activeCp, epoch, error, responseBody, "", Map.of());
        }

        public boolean confirms(String host, int targetCp) {
            return participantHost.equals(host)
                    && composition.getOrDefault(host, -1) == targetCp;
        }
    }

    /** Authoritative worker-side control record, read independently of any switch call. */
    record Status(boolean reachable,
                  boolean ok,
                  boolean busy,
                  int activeCp,
                  long epoch,
                  long committedEpoch,
                  String phase,
                  String error,
                  String responseBody) {

        static Status unreachable(String error) {
            return new Status(false, false, false, -1, -1L, -1L, "", error, "");
        }

        /** True only when the control record names one settled geometry. */
        public boolean settled() {
            return reachable && ok && !busy && activeCp > 0 && committedEpoch >= 0
                    && "commit".equalsIgnoreCase(phase);
        }

        public String describe() {
            if (!reachable) {
                return "actuator unreachable: " + error;
            }
            return "actuator ok=" + ok + " busy=" + busy + " active=CP" + activeCp
                    + " epoch=" + epoch + " committedEpoch=" + committedEpoch
                    + " phase=" + phase + (error == null || error.isBlank() ? "" : " error=" + error);
        }
    }

    boolean isConfigured();

    Result switchTo(int targetCp, long timeoutMs) throws Exception;

    /**
     * Move only {@code hosts} to {@code targetCp}, leaving the rest of the pool alone.
     *
     * <p>The router withholds exactly the endpoints it names here while the transition
     * runs, so the actuator must move exactly those and no others -- otherwise traffic is
     * routed at a host being restarted while a serving host sits withheld. Implementations
     * that cannot honour a host subset must refuse the operation; whole-pool switching
     * is deliberately unavailable while serving traffic.
     */
    default Result switchTo(int targetCp, long timeoutMs, List<String> hosts) throws Exception {
        return new Result(false, false, -1, -1L,
                "actuator does not support a single-host resident switch", "");
    }

    /** Reads the worker-side control record without mutating it. */
    Status status(long timeoutMs) throws Exception;

    /** A durable cancellation proves that an ambiguous request cannot commit later. */
    default boolean confirmsCancelled(String host, int targetCp, long previousCp, long previousEpoch) throws Exception {
        return false;
    }

    default boolean confirmsRecoveredEpoch(int targetCp, long observedEpoch, long previousEpoch, long timeoutMs)
            throws Exception {
        return observedEpoch > previousEpoch;
    }
}
