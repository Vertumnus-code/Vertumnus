package org.flexlb.balance.strategy;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Enumerates the topology-compatible operations that reach a composition
 * adjacent to the current one, and orders them.
 *
 * <p>Each operation changes the active geometry of exactly one resident host. It
 * replaces every degree-2C worker on that host with two degree-C workers, or performs
 * the inverse conversion for every complete sibling pair, preserving GPU count. Among
 * eligible operations the predicted drain interval is minimized first and the
 * expected KV-reconciliation traffic second, the latter estimated from the
 * missing destination copies of retained prefixes weighted by their recent
 * access frequencies.
 */
final class CpCompositionPlanner {

    /**
     * Do not rank an endpoint whose queue snapshot is no longer trustworthy as
     * the cheapest drain candidate.  A timed-out status RPC leaves the previous
     * WorkerStatus object in the scheduler; treating its absent/old queue map as
     * empty can select precisely the rank that is not responding and make the
     * soft-drain wait until timeout.
     */
    private static final long MAX_STATUS_AGE_US = 10_000_000L;

    record Config(double splitKvCoefficient,
                  double mergeKvCoefficient,
                  double splitDuplicationMinFrequency) {
        Config {
            if (splitKvCoefficient < 0.0 || mergeKvCoefficient < 0.0) {
                throw new IllegalArgumentException("KV traffic coefficients must be non-negative");
            }
            if (splitDuplicationMinFrequency < 0.0) {
                throw new IllegalArgumentException("duplication threshold must be non-negative");
            }
        }
    }

    /**
     * One host-wide split or merge. {@code participants} are the workers leaving the
     * scheduler's candidate set; {@code targetCp} is the degree they come back at.
     */
    record Operation(CpCompositionPolicy.Action action,
                     int childCp,
                     int parentCp,
                     List<CpComposition.WorkerGroup> participants,
                     double drainSeconds,
                     double kvTraffic,
                     boolean kvTrafficEstimated) {

        Operation {
            participants = List.copyOf(participants);
            if (participants.isEmpty()) {
                throw new IllegalArgumentException("a resident switch must name one host's workers");
            }
            String host = participants.get(0).host();
            if (participants.stream().anyMatch(group -> !group.host().equals(host))) {
                throw new IllegalArgumentException("a resident switch cannot span hosts");
            }
            if (action == CpCompositionPolicy.Action.MERGE && participants.size() % 2 != 0) {
                throw new IllegalArgumentException("a host-wide merge needs complete sibling pairs");
            }
        }

        int targetCp() {
            return action == CpCompositionPolicy.Action.SPLIT ? childCp : parentCp;
        }

        String id() {
            return action + ":CP" + (action == CpCompositionPolicy.Action.SPLIT ? parentCp : childCp)
                    + "->CP" + targetCp() + "@host=" + host() + "["
                    + String.join(",", participants.stream().map(CpComposition.WorkerGroup::id).toList())
                    + "]";
        }

        String host() {
            return participants.get(0).host();
        }

        CpComposition applyTo(CpComposition composition) {
            return action == CpCompositionPolicy.Action.SPLIT
                    ? composition.split(parentCp, participants.size())
                    : composition.merge(childCp, participants.size());
        }

        Map<String, Object> snapshot() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("action", action);
            result.put("child_cp", childCp);
            result.put("parent_cp", parentCp);
            result.put("target_cp", targetCp());
            result.put("host", host());
            result.put("host_wide", true);
            result.put("participants", participants.stream()
                    .map(CpComposition.WorkerGroup::id).toList());
            result.put("drain_seconds", drainSeconds);
            result.put("kv_traffic", kvTraffic);
            result.put("kv_traffic_estimated", kvTrafficEstimated);
            return result;
        }
    }

    private static final Comparator<Operation> RANKING =
            Comparator.comparingDouble(Operation::drainSeconds)
                    .thenComparingDouble(Operation::kvTraffic)
                    .thenComparing(Operation::id);

    private final List<Integer> ladder;
    private final CpPrefixAccessTracker accessTracker;
    private final Config config;

    CpCompositionPlanner(List<Integer> ladder, CpPrefixAccessTracker accessTracker, Config config) {
        this.ladder = ladder.stream().distinct().sorted().toList();
        this.accessTracker = accessTracker;
        this.config = config;
    }

    /** Eligible operations in the given direction, best first. */
    List<Operation> enumerate(CpCompositionPolicy.Action direction, CpComposition composition) {
        List<Operation> operations = new ArrayList<>();
        if (direction == CpCompositionPolicy.Action.SPLIT) {
            for (Integer parentCp : ladder) {
                int childCp = parentCp / 2;
                if (parentCp % 2 != 0 || !ladder.contains(childCp)) {
                    continue;
                }
                for (List<CpComposition.WorkerGroup> hostGroups
                        : completeHostsAt(composition, parentCp).values()) {
                    operations.add(build(CpCompositionPolicy.Action.SPLIT, childCp, parentCp,
                            hostGroups));
                }
            }
        } else if (direction == CpCompositionPolicy.Action.MERGE) {
            for (Integer childCp : ladder) {
                int parentCp = childCp * 2;
                if (!ladder.contains(parentCp)) {
                    continue;
                }
                for (List<CpComposition.WorkerGroup> siblings
                        : completeHostsAt(composition, childCp).values()) {
                    if (siblings.size() % 2 != 0) {
                        continue;
                    }
                    operations.add(build(CpCompositionPolicy.Action.MERGE, childCp, parentCp,
                            siblings));
                }
            }
        }
        // An infinite drain cost means at least one participant has no fresh,
        // trustworthy status.  Leave it out entirely; if every candidate is
        // stale the policy will HOLD until the synchronizer publishes a fresh
        // snapshot instead of guessing a host and entering a doomed drain.
        return operations.stream()
                .filter(operation -> Double.isFinite(operation.drainSeconds()))
                .sorted(RANKING)
                .toList();
    }

    /**
     * A resident control record changes the geometry of an entire host. Exclude a host
     * when its visible worker groups are mixed across degrees; acting on only the groups
     * at {@code cpSize} would still reconfigure the omitted endpoints in the engine.
     */
    private static Map<String, List<CpComposition.WorkerGroup>> completeHostsAt(
            CpComposition composition, int cpSize) {
        Map<String, List<CpComposition.WorkerGroup>> result = new LinkedHashMap<>();
        for (CpComposition.WorkerGroup group : composition.groupsAt(cpSize)) {
            result.computeIfAbsent(group.host(), ignored -> new ArrayList<>()).add(group);
        }
        result.entrySet().removeIf(entry ->
                !composition.isHostComplete(entry.getKey())
                        || composition.groupsOnHost(entry.getKey()).size() != entry.getValue().size());
        result.values().forEach(groups -> groups.sort(Comparator.comparing(
                CpComposition.WorkerGroup::id)));
        return result;
    }

    private Operation build(CpCompositionPolicy.Action action, int childCp, int parentCp,
                            List<CpComposition.WorkerGroup> participants) {
        double drainSeconds = participants.stream()
                .mapToDouble(CpCompositionPlanner::drainSeconds)
                .max().orElse(0.0);
        KvTraffic traffic = action == CpCompositionPolicy.Action.SPLIT
                ? splitTraffic(participants)
                : mergeTraffic(participants);
        return new Operation(action, childCp, parentCp, participants,
                drainSeconds, traffic.units(), traffic.estimated());
    }

    private record KvTraffic(double units, boolean estimated) {
    }

    private KvTraffic mergeTraffic(List<CpComposition.WorkerGroup> siblings) {
        double units = 0.0;
        boolean estimated = false;
        for (int i = 0; i < siblings.size(); i += 2) {
            KvTraffic pair = mergeTraffic(siblings.get(i), siblings.get(i + 1));
            units += pair.units();
            estimated |= pair.estimated();
        }
        return new KvTraffic(units, estimated);
    }

    private KvTraffic splitTraffic(List<CpComposition.WorkerGroup> parents) {
        double units = 0.0;
        boolean estimated = false;
        for (CpComposition.WorkerGroup parent : parents) {
            KvTraffic worker = splitTraffic(parent);
            units += worker.units();
            estimated |= worker.estimated();
        }
        return new KvTraffic(units, estimated);
    }

    /**
     * Seconds until the participating worker has drained. Its dp ranks retire
     * their queues concurrently, so the group is drained when its slowest rank is,
     * and a group fronted by several endpoints is drained when all of them are.
     */
    private static double drainSeconds(CpComposition.WorkerGroup group) {
        double slowest = 0.0;
        for (WorkerStatus endpoint : group.endpoints()) {
            if (endpoint == null || !endpoint.isAlive()
                    || endpoint.getStatusLastUpdateTime() == null
                    || endpoint.getStatusLastUpdateTime().get() <= 0L
                    || statusAgeUs(endpoint) > MAX_STATUS_AGE_US) {
                return Double.POSITIVE_INFINITY;
            }
            int dpSize = Math.max(1, Math.toIntExact(Math.max(1L, endpoint.getDpSize())));
            for (int dpRank = 0; dpRank < dpSize; dpRank++) {
                if (group.dpLane() >= 0 && group.dpLane() != dpRank) { continue; }
                slowest = Math.max(slowest,
                        CpAwareTtftStrategy.calculateQueueSeconds(endpoint, dpRank));
            }
        }
        return slowest;
    }

    private static long statusAgeUs(WorkerStatus endpoint) {
        long updated = endpoint.getStatusLastUpdateTime().get();
        long now = System.nanoTime() / 1_000L;
        return Math.max(0L, now - updated);
    }

    /**
     * A merge retains the union of the sibling prefixes, and every block that is
     * not already resident on both siblings is missing from the merged worker.
     */
    private KvTraffic mergeTraffic(CpComposition.WorkerGroup left, CpComposition.WorkerGroup right) {
        Set<Long> leftKeys = residentKeys(left);
        Set<Long> rightKeys = residentKeys(right);
        if (leftKeys.isEmpty() && rightKeys.isEmpty()) {
            double missing = Math.abs(residentBlockCount(left) - residentBlockCount(right));
            return new KvTraffic(config.mergeKvCoefficient() * missing, true);
        }
        Set<Long> missing = new HashSet<>(leftKeys);
        missing.addAll(rightKeys);
        Set<Long> shared = new HashSet<>(leftKeys);
        shared.retainAll(rightKeys);
        missing.removeAll(shared);
        return new KvTraffic(config.mergeKvCoefficient() * accessTracker.weight(missing), false);
    }

    /**
     * A split assigns each retained prefix to one child, so a destination copy is
     * missing only where both children demand the same prefix. A prefix whose
     * recent access frequency reaches the duplication threshold is treated as
     * demanded on both sides.
     */
    private KvTraffic splitTraffic(CpComposition.WorkerGroup parent) {
        Set<Long> keys = residentKeys(parent);
        if (keys.isEmpty()) {
            return new KvTraffic(config.splitKvCoefficient() * residentBlockCount(parent), true);
        }
        double weight = accessTracker.weightAbove(keys, config.splitDuplicationMinFrequency());
        return new KvTraffic(config.splitKvCoefficient() * weight, false);
    }

    private static Set<Long> residentKeys(CpComposition.WorkerGroup group) {
        Set<Long> keys = new LinkedHashSet<>();
        for (WorkerStatus endpoint : group.endpoints()) {
            if (group.dpLane() >= 0) {
                addKeys(keys, endpoint.getDpRankCacheStatusMap().get(group.dpLane()));
                continue;
            }
            addKeys(keys, endpoint.getCacheStatus());
            if (endpoint.getDpRankCacheStatusMap() != null) {
                endpoint.getDpRankCacheStatusMap().values().forEach(status -> addKeys(keys, status));
            }
        }
        return keys;
    }

    private static void addKeys(Set<Long> target, CacheStatus status) {
        if (status != null && status.getCachedKeys() != null) {
            target.addAll(status.getCachedKeys());
        }
    }

    private static double residentBlockCount(CpComposition.WorkerGroup group) {
        double total = 0.0;
        for (WorkerStatus endpoint : group.endpoints()) {
            if (group.dpLane() >= 0) {
                var status = endpoint.getDpRankCacheStatusMap().get(group.dpLane());
                total += status == null ? 0L : Math.max(0L, status.getCacheKeySize());
                continue;
            }
            if (endpoint.getDpRankCacheStatusMap() != null
                    && !endpoint.getDpRankCacheStatusMap().isEmpty()) {
                for (CacheStatus status : endpoint.getDpRankCacheStatusMap().values()) {
                    total += status == null ? 0L : Math.max(0L, status.getCacheKeySize());
                }
            } else if (endpoint.getCacheStatus() != null) {
                total += Math.max(0L, endpoint.getCacheStatus().getCacheKeySize());
            }
        }
        return total;
    }
}
