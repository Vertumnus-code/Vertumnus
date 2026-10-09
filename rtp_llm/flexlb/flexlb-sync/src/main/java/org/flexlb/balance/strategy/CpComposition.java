package org.flexlb.balance.strategy;

import org.flexlb.dao.master.WorkerStatus;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

/**
 * The worker composition: how many workers run at each CP degree.
 *
 * <p>A composition is a vector n over the supported degrees, and every split or
 * merge operation preserves the aggregate GPU count sum(n_C * C). The vector is
 * derived from the CP-carrying workers the manager can see rather than from a
 * declared topology, so it also names the concrete workers an operation can act
 * on: one degree-C worker occupies C ranks, which a deployment may expose as
 * several endpoints when a rank group is fronted per tensor-parallel shard.
 */
final class CpComposition {

    /** One worker at one CP degree, together with the endpoints that front it. */
    record WorkerGroup(int cpSize, String host, String id, List<WorkerStatus> endpoints, int dpLane) {
        WorkerGroup(int cpSize, String host, String id, List<WorkerStatus> endpoints) {
            this(cpSize, host, id, endpoints, -1);
        }
    }

    private static final CpComposition EMPTY = new CpComposition(new TreeMap<>(), List.of(), Set.of());

    private final NavigableMap<Integer, Integer> workersByCp;
    private final List<WorkerGroup> groups;
    private final Set<String> incompleteHosts;

    private CpComposition(NavigableMap<Integer, Integer> workersByCp, List<WorkerGroup> groups,
                          Set<String> incompleteHosts) {
        this.workersByCp = workersByCp;
        this.groups = groups;
        this.incompleteHosts = incompleteHosts;
    }

    static CpComposition empty() {
        return EMPTY;
    }

    /** A composition known only as a vector, used when planning a deployment. */
    static CpComposition of(Map<Integer, Integer> workersByCp) {
        NavigableMap<Integer, Integer> vector = new TreeMap<>();
        workersByCp.forEach((cp, count) -> {
            if (cp == null || cp <= 0) {
                throw new IllegalArgumentException("CP degree must be positive: " + cp);
            }
            if (count == null || count < 0) {
                throw new IllegalArgumentException("worker count must be non-negative: " + count);
            }
            if (count > 0) {
                vector.put(cp, count);
            }
        });
        return new CpComposition(vector, List.of(), Set.of());
    }

    /**
     * Groups live endpoints into workers. Endpoints are first isolated by host, then
     * consumed in port/rank order until a group owns C ranks. An incomplete host is
     * retained as ineligible rather than allowing a group to cross a machine boundary.
     */
    static CpComposition fromWorkers(Collection<WorkerStatus> workers) {
        Map<Integer, List<WorkerStatus>> byDegree = new LinkedHashMap<>();
        for (WorkerStatus worker : workers) {
            if (worker == null || worker.getCpSize() <= 0) {
                continue;
            }
            byDegree.computeIfAbsent(Math.toIntExact(worker.getCpSize()), key -> new ArrayList<>())
                    .add(worker);
        }
        NavigableMap<Integer, Integer> vector = new TreeMap<>();
        List<WorkerGroup> groups = new ArrayList<>();
        Set<String> incompleteHosts = new HashSet<>();
        new TreeMap<>(byDegree).forEach((cpSize, endpoints) -> {
            Map<String, List<WorkerStatus>> byHost = new TreeMap<>();
            endpoints.forEach(endpoint -> byHost
                    .computeIfAbsent(hostOf(endpoint), ignored -> new ArrayList<>())
                    .add(endpoint));
            byHost.forEach((host, hostEndpoints) -> {
                if (!addHostGroups(cpSize, host, hostEndpoints, groups)) {
                    incompleteHosts.add(host);
                }
            });
            long count = groups.stream().filter(group -> group.cpSize() == cpSize).count();
            if (count > 0) {
                vector.put(cpSize, Math.toIntExact(count));
            }
        });
        groups.sort(Comparator.comparingInt(WorkerGroup::cpSize).thenComparing(WorkerGroup::id));
        return new CpComposition(vector, List.copyOf(groups), Set.copyOf(incompleteHosts));
    }

    private static boolean addHostGroups(int cpSize, String host,
                                         List<WorkerStatus> endpoints,
                                         List<WorkerGroup> groups) {
        if (endpoints.size() == 1 && endpoints.getFirst().isNativeCpTopology()) {
            var endpoint = endpoints.getFirst();
            int lanes = Math.toIntExact(endpoint.getDpSize());
            if (lanes < 1 || endpoint.getTpSize() != cpSize) { return false; }
            for (int lane = 0; lane < lanes; lane++) {
                groups.add(new WorkerGroup(cpSize, host, host + "#dp" + lane, List.of(endpoint), lane));
            }
            return true;
        }
        endpoints.sort(Comparator.comparingInt(WorkerStatus::getPort));
        List<WorkerStatus> pending = new ArrayList<>();
        int ranks = 0;
        for (WorkerStatus endpoint : endpoints) {
            int endpointRanks = ranksPerEndpoint(endpoint, cpSize);
            if (endpointRanks <= 0 || endpointRanks > cpSize || ranks + endpointRanks > cpSize) {
                pending.clear();
                ranks = 0;
                return false;
            }
            pending.add(endpoint);
            ranks += endpointRanks;
            if (ranks == cpSize) {
                groups.add(new WorkerGroup(cpSize, host, groupId(pending), List.copyOf(pending)));
                pending.clear();
                ranks = 0;
            }
        }
        return pending.isEmpty();
    }

    private static String hostOf(WorkerStatus worker) {
        return worker.getCpSwitchDomain() == null ? "" : worker.getCpSwitchDomain();
    }

    /**
     * Ranks a single endpoint owns. A deployment that fronts one rank group per
     * tensor-parallel shard reports that constant directly; the lane count is a second
     * best because it tracks the active degree rather than the endpoint layout, and a
     * worker that reports neither is fronted as one whole group.
     */
    private static int ranksPerEndpoint(WorkerStatus worker, int cpSize) {
        long perEndpoint = worker.getCpRanksPerEndpoint();
        if (perEndpoint > 0L) {
            return Math.toIntExact(perEndpoint);
        }
        long world = worker.getWorldSize();
        long lanes = worker.getLogicalLaneCount();
        if (world > 0L && lanes > 0L && world % lanes == 0L) {
            return Math.toIntExact(world / lanes);
        }
        return cpSize;
    }

    private static String groupId(List<WorkerStatus> endpoints) {
        WorkerStatus first = endpoints.get(0);
        String head = (first.getIp() == null ? "" : first.getIp()) + ":" + first.getPort();
        return endpoints.size() == 1 ? head : head + "+" + (endpoints.size() - 1);
    }

    boolean isEmpty() {
        return workersByCp.isEmpty();
    }

    NavigableMap<Integer, Integer> vector() {
        return new TreeMap<>(workersByCp);
    }

    List<WorkerGroup> groups() {
        return groups;
    }

    List<WorkerGroup> groupsAt(int cpSize) {
        return groups.stream().filter(group -> group.cpSize() == cpSize).toList();
    }

    List<WorkerGroup> groupsOnHost(String host) {
        return groups.stream().filter(group -> group.host().equals(host)).toList();
    }

    boolean isHostComplete(String host) {
        return !incompleteHosts.contains(host);
    }

    int workerCount(int cpSize) {
        return workersByCp.getOrDefault(cpSize, 0);
    }

    int gpuCount() {
        int total = 0;
        for (Map.Entry<Integer, Integer> entry : workersByCp.entrySet()) {
            total += entry.getKey() * entry.getValue();
        }
        return total;
    }

    /** sum(n_C * kappa_C), NaN when a present degree has no profiled capacity. */
    double profiledWindowCapacity(CpCapacityProfile profile) {
        double total = 0.0;
        for (Map.Entry<Integer, Integer> entry : workersByCp.entrySet()) {
            double kappa = profile.windowDemand(entry.getKey());
            if (!Double.isFinite(kappa)) {
                return Double.NaN;
            }
            total += kappa * entry.getValue();
        }
        return workersByCp.isEmpty() ? Double.NaN : total;
    }

    /** The same aggregate capacity expressed per wall second. */
    double profiledCapacityRate(CpCapacityProfile profile) {
        double total = 0.0;
        for (Map.Entry<Integer, Integer> entry : workersByCp.entrySet()) {
            double rate = profile.demandRate(entry.getKey());
            if (!Double.isFinite(rate)) {
                return Double.NaN;
            }
            total += rate * entry.getValue();
        }
        return workersByCp.isEmpty() ? Double.NaN : total;
    }

    /** Replaces one degree-2C worker with two degree-C workers. */
    CpComposition split(int parentCp) {
        return split(parentCp, 1);
    }

    CpComposition split(int parentCp, int parentWorkers) {
        if (parentCp % 2 != 0 || parentWorkers < 1 || workerCount(parentCp) < parentWorkers) {
            throw new IllegalArgumentException("cannot split CP" + parentCp + " in " + describe());
        }
        NavigableMap<Integer, Integer> vector = vector();
        decrement(vector, parentCp, parentWorkers);
        vector.merge(parentCp / 2, parentWorkers * 2, Integer::sum);
        return new CpComposition(vector, List.of(), Set.of());
    }

    /** Replaces two degree-C workers with one degree-2C worker. */
    CpComposition merge(int childCp) {
        return merge(childCp, 2);
    }

    CpComposition merge(int childCp, int childWorkers) {
        if (childWorkers < 2 || childWorkers % 2 != 0 || workerCount(childCp) < childWorkers) {
            throw new IllegalArgumentException("cannot merge CP" + childCp + " in " + describe());
        }
        NavigableMap<Integer, Integer> vector = vector();
        decrement(vector, childCp, childWorkers);
        vector.merge(childCp * 2, childWorkers / 2, Integer::sum);
        return new CpComposition(vector, List.of(), Set.of());
    }

    String describe() {
        if (workersByCp.isEmpty()) {
            return "none";
        }
        StringBuilder builder = new StringBuilder();
        workersByCp.forEach((cp, count) -> {
            if (builder.length() > 0) {
                builder.append('+');
            }
            builder.append(count).append("xCP").append(cp);
        });
        return builder.toString();
    }

    Map<String, Object> snapshot(CpCapacityProfile profile) {
        Map<String, Object> result = new LinkedHashMap<>();
        Map<String, Object> vector = new LinkedHashMap<>();
        workersByCp.forEach((cp, count) -> vector.put(String.valueOf(cp), count));
        result.put("workers_by_cp", vector);
        result.put("gpu_count", gpuCount());
        result.put("description", describe());
        result.put("profiled_window_capacity_seconds", profiledWindowCapacity(profile));
        result.put("profiled_capacity_rate", profiledCapacityRate(profile));
        result.put("worker_groups", groups.stream()
                .map(group -> "CP" + group.cpSize() + "@" + group.id()).toList());
        return result;
    }

    private static void decrement(NavigableMap<Integer, Integer> vector, int cpSize, int by) {
        int remaining = vector.getOrDefault(cpSize, 0) - by;
        if (remaining < 0) {
            throw new IllegalArgumentException("composition has fewer than " + by + " CP" + cpSize + " workers");
        }
        if (remaining == 0) {
            vector.remove(cpSize);
        } else {
            vector.put(cpSize, remaining);
        }
    }
}
