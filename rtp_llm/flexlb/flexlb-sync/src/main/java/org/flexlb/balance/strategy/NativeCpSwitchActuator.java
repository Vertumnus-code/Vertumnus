package org.flexlb.balance.strategy;

import org.flexlb.sync.prefix.PrefixTopologyControl;
import org.flexlb.sync.prefix.PrefixTransferRecoveryService;
import org.flexlb.sync.prefix.ZookeeperPrefixTopologyJournal;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.JsonUtils;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** Connects composition decisions to the durable, cache-preserving native protocol. */
@Component
@ConditionalOnProperty(name = "FLEXLB_CP_SWITCH_MODE", havingValue = "native")
public final class NativeCpSwitchActuator implements CpSwitchActuator {
    private final PrefixTransferRecoveryService recovery;
    private final EngineWorkerStatus workers;
    private final boolean configured;
    private final String token;
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    private volatile String activePlan;
    private String activeHost;
    private int activeTarget;
    private long previousCp;
    private long previousEpoch;
    private PrefixTopologyControl.Request activeRequest;
    private boolean frontendAcknowledged;

    public NativeCpSwitchActuator(PrefixTransferRecoveryService recovery, EngineWorkerStatus workers,
                                 Environment environment) {
        this.recovery = recovery;
        this.workers = workers;
        configured = environment.getProperty("prefix-cache.topology.control.enabled", Boolean.class, false);
        token = environment.getProperty("prefix-cache.topology.control.token", "");
        if (configured && !token.matches("[A-Za-z0-9_-]{32,256}")) {
            throw new IllegalArgumentException("native topology control requires a valid frontend token");
        }
    }

    @Override
    public boolean isConfigured() { return configured; }

    @Override
    public Result switchTo(int targetCp, long timeoutMs) {
        throw new IllegalArgumentException("a native switch requires one configured world endpoint");
    }

    @Override
    public synchronized Result switchTo(int targetCp, long timeoutMs, List<String> domains) throws Exception {
        if (!configured || domains == null || domains.size() != 1 || targetCp < 1) {
            throw new IllegalArgumentException("native topology control requires one world");
        }
        String domain = domains.getFirst();
        var selected = workers.liveCpWorkers().stream()
                .filter(worker -> domain.equals(worker.getCpSwitchDomain())).toList();
        if (selected.size() != 1) { throw new IllegalStateException("world has missing or aliased endpoints"); }
        var worker = selected.getFirst();
        int degree = Math.toIntExact(worker.getCpSize());
        int world = Math.multiplyExact(degree, Math.toIntExact(worker.getDpSize()));
        if (world % targetCp != 0 || (degree != 2 * targetCp && targetCp != 2 * degree)) {
            throw new IllegalArgumentException("switch must follow one adjacent topology edge");
        }
        var control = recovery.topologyControl();
        if (activePlan != null) {
            var previous = control.read(activePlan);
            // ADMITTED is a durable terminal state. Its current route availability
            // can temporarily disappear while the placement reader refreshes.
            if (previous.entry().state() != ZookeeperPrefixTopologyJournal.State.CANCELLED
                    && previous.entry().state() != ZookeeperPrefixTopologyJournal.State.ADMITTED) {
                throw new IllegalStateException("previous native plan still requires recovery: " + activePlan);
            }
            if (previous.entry().state() == ZookeeperPrefixTopologyJournal.State.ADMITTED && !frontendAcknowledged) {
                throw new IllegalStateException("previous native plan still requires frontend acknowledgement: " + activePlan);
            }
        }
        String id = UUID.randomUUID().toString();
        long deadline = System.currentTimeMillis() + Math.min(300000L, Math.max(1000L, timeoutMs));
        // Bind before the POST: an ambiguous prepare must never reuse an older cancellation.
        activePlan = id;
        frontendAcknowledged = false;
        activeHost = domain;
        activeTarget = targetCp;
        previousCp = degree;
        previousEpoch = worker.getCpEpoch();
        activeRequest = new PrefixTopologyControl.Request(id, worker.getIpPort(), targetCp, world / targetCp, deadline);
        control.prepare(activeRequest);
        // The frontend owns the ingress fence and must publish its new DP addresses
        // before admitting the durable plan. A direct control.admit() bypasses that fence.
        long remaining = Math.max(1L, deadline - System.currentTimeMillis());
        var body = Map.of("switch_id", id, "tp_size", targetCp, "dp_size", world / targetCp,
                "timeout_ms", remaining, "deadline_unix_ms", deadline);
        var request = HttpRequest.newBuilder(URI.create("http://" + worker.getIpPort() + "/switch_parallel_topology"))
                .timeout(Duration.ofMillis(remaining + 1000))
                .header("Authorization", "Bearer " + token).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JsonUtils.toString(body))).build();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        frontendAcknowledged = acknowledges(response, "ok");
        var state = control.read(id);
        while (state.entry().state() == ZookeeperPrefixTopologyJournal.State.ADMITTED
                && !state.frontendAdmitted() && System.currentTimeMillis() < deadline) {
            Thread.sleep(Math.min(100L, Math.max(1L, deadline - System.currentTimeMillis())));
            state = control.read(id);
        }
        if (state.entry().state() == ZookeeperPrefixTopologyJournal.State.ADMITTED && state.frontendAdmitted()
                && frontendAcknowledged) {
            return result(state, domain);
        }
        if (state.entry().state() == ZookeeperPrefixTopologyJournal.State.CANCELLED) {
            finishCancelledFrontend();
            return new Result(false, false, degree, worker.getCpEpoch(), "native plan cancelled", "", domain, Map.of());
        }
        throw new IllegalStateException("frontend has not admitted native plan " + id + "; HTTP " + response.statusCode());
    }

    private static Result result(PrefixTopologyControl.Status state, String domain) {
        var plan = state.entry().plan();
        long epoch = UUID.fromString(plan.switchId()).getMostSignificantBits() & Long.MAX_VALUE;
        return new Result(true, true, plan.tpSize(), epoch, "", plan.switchId(), domain, Map.of(domain, plan.tpSize()));
    }

    @Override
    public synchronized Status status(long timeoutMs) {
        String id = activePlan;
        if (id == null) { return Status.unreachable("no native plan has been submitted"); }
        var state = recovery.topologyControl().read(id);
        if (state.entry().state() != ZookeeperPrefixTopologyJournal.State.ADMITTED || !state.frontendAdmitted()) {
            return Status.unreachable("native plan has not completed cache and frontend admission: " + id);
        }
        if (!frontendAcknowledged) {
            try {
                finishFrontend("ok", Math.min(300000L, Math.max(1L, timeoutMs)));
                frontendAcknowledged = true;
            } catch (Exception failure) {
                if (failure instanceof InterruptedException) { Thread.currentThread().interrupt(); }
                return Status.unreachable("native frontend has not acknowledged admitted plan: " + id);
            }
            state = recovery.topologyControl().read(id);
            if (state.entry().state() != ZookeeperPrefixTopologyJournal.State.ADMITTED || !state.frontendAdmitted()) {
                return Status.unreachable("native placement changed during frontend acknowledgement: " + id);
            }
        }
        var plan = state.entry().plan();
        long epoch = UUID.fromString(id).getMostSignificantBits() & Long.MAX_VALUE;
        return new Status(true, true, false, plan.tpSize(), epoch, epoch, "commit", "", id);
    }

    @Override
    public synchronized boolean confirmsCancelled(String host, int targetCp, long originalCp, long originalEpoch) throws Exception {
        if (activePlan == null || !host.equals(activeHost) || targetCp != activeTarget
                || originalCp != previousCp || originalEpoch != previousEpoch) { return false; }
        if (recovery.topologyControl().read(activePlan).entry().state() != ZookeeperPrefixTopologyJournal.State.CANCELLED) {
            return false;
        }
        finishCancelledFrontend();
        return true;
    }

    private void finishCancelledFrontend() throws Exception {
        finishFrontend("cancelled", 5000);
    }

    private void finishFrontend(String expectedStatus, long timeoutMs) throws Exception {
        // A timed-out frontend retains its shared gate. Replay only the original
        // terminal identity so the frontend can durably release that gate.
        var body = Map.of("switch_id", activeRequest.switchId(), "tp_size", activeRequest.tpSize(),
                "dp_size", activeRequest.dpSize(), "deadline_unix_ms", activeRequest.deadlineUnixMs(), "timeout_ms", timeoutMs);
        var request = HttpRequest.newBuilder(URI.create("http://" + activeRequest.route() + "/switch_parallel_topology"))
                .timeout(Duration.ofMillis(timeoutMs + 1000)).header("Authorization", "Bearer " + token)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JsonUtils.toString(body))).build();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (!acknowledges(response, expectedStatus)) {
            throw new IllegalStateException("frontend has not released native plan " + activePlan);
        }
    }

    private boolean acknowledges(HttpResponse<String> response, String expectedStatus) {
        if (response.statusCode() != 200) { return false; }
        try {
            var result = JsonUtils.toTreeNode(response.body());
            return result != null && expectedStatus.equals(result.path("status").asText())
                    && activeRequest.switchId().equals(result.path("switch_id").asText());
        } catch (RuntimeException invalid) { return false; }
    }

    @Override
    public boolean confirmsRecoveredEpoch(int targetCp, long observedEpoch, long previousEpoch, long timeoutMs) {
        var state = status(timeoutMs);
        return state.settled() && state.activeCp() == targetCp && state.epoch() == observedEpoch;
    }
}
