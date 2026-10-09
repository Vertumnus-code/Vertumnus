package org.flexlb.sync.prefix;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.status.EngineWorkerStatus;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

public final class PrefixTopologyFrontend {
    private final Function<String, CacheStatus> fetch;
    private final Function<String, List<WorkerStatus>> lookup;

    public PrefixTopologyFrontend(long timeoutMillis) {
        if (timeoutMillis < 100 || timeoutMillis > 300000) { throw new IllegalArgumentException("invalid frontend proof timeout"); }
        var timeout = Duration.ofMillis(timeoutMillis);
        var http = HttpClient.newBuilder().connectTimeout(timeout).followRedirects(HttpClient.Redirect.NEVER).build();
        var mapper = new ObjectMapper();
        fetch = route -> {
            try {
                var request = HttpRequest.newBuilder(URI.create("http://" + route + "/cache_status"))
                        .timeout(timeout).GET().build();
                var response = http.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() != 200 || response.body().length() > 16 * 1024 * 1024) {
                    throw new IllegalStateException("frontend cache mapping is not available or exceeds bound");
                }
                return mapper.readValue(response.body(), CacheStatus.class);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("frontend mapping proof interrupted", interrupted);
            } catch (Exception failure) { throw new IllegalStateException("frontend mapping proof is not confirmed", failure); }
        };
        lookup = PrefixTopologyFrontend::workers;
    }

    PrefixTopologyFrontend(Function<String, CacheStatus> fetch, Function<String, List<WorkerStatus>> lookup) {
        this.fetch = Objects.requireNonNull(fetch);
        this.lookup = Objects.requireNonNull(lookup);
    }

    public void validate(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> registrations,
                         List<EngineRpcService.CacheStatusPB> reports) {
        validate(plan, registrations, reports, fetch(registrations));
    }

    CacheStatus fetch(List<PrefixWorkerRegistry.Registration> registrations) {
        return Objects.requireNonNull(fetch.apply(ConfirmedCacheDirectory.routeBase(registrations.getFirst().route())));
    }

    private void validate(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> registrations,
                          List<EngineRpcService.CacheStatusPB> reports, CacheStatus response) {
        String route = ConfirmedCacheDirectory.routeBase(registrations.getFirst().route());
        var expected = PrefixTopologyAdmission.validate(plan, registrations, reports);
        var lanes = response.getResults();
        if (lanes == null || lanes.size() != plan.dpSize()) { throw new IllegalStateException("frontend has not installed every admitted lane"); }
        var seen = new java.util.HashSet<Integer>();
        for (var lane : lanes) {
            var actual = lane == null ? null : lane.getInventory();
            if (actual == null || !seen.add(actual.dpLane())) { throw new IllegalStateException("frontend lane inventory is missing or duplicated"); }
            String key = plan.dpSize() == 1 ? route : route + "#dp" + actual.dpLane();
            var recorded = expected.get(key);
            if (recorded == null || !recorded.namespace().equals(actual.namespace()) || !recorded.workerId().equals(actual.workerId())
                    || !recorded.epoch().equals(actual.epoch()) || recorded.cpDegree() != actual.cpDegree()
                    || !recorded.layoutFingerprint().equals(actual.layoutFingerprint()) || !actual.deviceReady()
                    || !"FULL".equals(actual.snapshotKind()) || actual.version() < recorded.version()
                    || actual.rankReadinessGeneration() < recorded.rankReadinessGeneration()) {
                throw new IllegalStateException("frontend still exposes another topology incarnation");
            }
        }
        var statuses = lookup.apply(route);
        if (statuses.isEmpty()) { throw new IllegalStateException("admitted frontend is not registered with this load balancer"); }
        var topology = topology(plan, registrations, reports);
        for (var worker : statuses) {
            if (!topology.equals(worker.getPrefixTopology()) && !worker.getLocalTaskMap().isEmpty()) {
                throw new IllegalStateException("frontend still has requests from the previous topology");
            }
        }
    }

    public void reconcile(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> registrations,
                          List<EngineRpcService.CacheStatusPB> reports) {
        reconcile(plan, registrations, reports, fetch(registrations));
    }

    void reconcile(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> registrations,
                   List<EngineRpcService.CacheStatusPB> reports, CacheStatus response) {
        validate(plan, registrations, reports, response);
        var topology = topology(plan, registrations, reports);
        var route = ConfirmedCacheDirectory.routeBase(registrations.getFirst().route());
        lookup.apply(route).forEach(worker -> worker.installPrefixTopology(topology));
    }

    private static WorkerStatus.PrefixTopology topology(PrefixTopologyPlan plan, List<PrefixWorkerRegistry.Registration> registrations,
                                                        List<EngineRpcService.CacheStatusPB> reports) {
        var workers = new HashMap<Integer, String>();
        var epochs = new HashMap<Integer, String>();
        for (var registration : registrations) {
            var inventory = reports.get(registration.dpLane() * plan.tpSize()).getInventory();
            workers.put(registration.dpLane(), inventory.getWorkerId());
            epochs.put(registration.dpLane(), inventory.getEpoch());
        }
        return new WorkerStatus.PrefixTopology(plan.switchId(), plan.namespace(), plan.tpSize(), plan.dpSize(), workers, epochs);
    }

    private static List<WorkerStatus> workers(String route) {
        var result = new ArrayList<WorkerStatus>();
        var seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<WorkerStatus, Boolean>());
        for (var role : RoleType.values()) {
            var statuses = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role);
            if (statuses == null) { continue; }
            statuses.values().stream().filter(worker -> worker != null && worker.getIpPort().equals(route) && seen.add(worker)).forEach(result::add);
        }
        return List.copyOf(result);
    }
}
