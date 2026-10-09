package org.flexlb.cache.prefix;

import org.flexlb.cache.prefix.PrefixReplicaPlanner.Proposal;
import org.flexlb.cache.prefix.PrefixReplicaPlanner.Worker;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiPredicate;

public final class PrefixPreservationPlanner {
    public record Limits(int maxPaths, long destinationBytes, long wireBytes, long safetyBytes) {
        public Limits {
            if (maxPaths < 1 || maxPaths > 4096 || destinationBytes < 0 || wireBytes < 0 || safetyBytes < 0) {
                throw new IllegalArgumentException("invalid preservation limits");
            }
        }
    }
    public record Retained(List<Long> path, String holder) {
        public Retained { path = List.copyOf(path); }
    }
    public record Reclaim(String route, List<Long> keys) {
        public Reclaim { keys = List.copyOf(keys); }
    }
    public record Plan(List<Retained> required, List<Proposal> copies, long destinationBytes, long wireBytes,
                       List<Reclaim> reclaims) {
        public Plan { required = List.copyOf(required); copies = List.copyOf(copies); reclaims = List.copyOf(reclaims); }
    }

    public Plan plan(List<PrefixReplicaPlanner.Demand> demands, List<Worker> workers, Set<String> affected,
                     Limits limits, BiPredicate<Worker, Worker> compatible) {
        return plan(demands, workers, affected, limits, compatible, Map.of());
    }

    public Plan plan(List<PrefixReplicaPlanner.Demand> demands, List<Worker> workers, Set<String> affected,
                     Limits limits, BiPredicate<Worker, Worker> compatible, Map<String, List<Long>> reclaimable) {
        return plan(demands, workers, affected, limits, compatible, reclaimable, Map.of());
    }

    public Plan plan(List<PrefixReplicaPlanner.Demand> demands, List<Worker> workers, Set<String> affected,
                     Limits limits, BiPredicate<Worker, Worker> compatible, Map<String, List<Long>> reclaimable,
                     Map<String, List<List<Long>>> rejectedReclaims) {
        return plan(demands, workers, affected, limits, compatible, reclaimable, rejectedReclaims, Set.of());
    }

    public Plan plan(List<PrefixReplicaPlanner.Demand> demands, List<Worker> workers, Set<String> affected,
                     Limits limits, BiPredicate<Worker, Worker> compatible, Map<String, List<Long>> reclaimable,
                     Map<String, List<List<Long>>> rejectedReclaims, Set<String> residentHolders) {
        if (workers.size() > 4096 || affected.isEmpty()) { throw new IllegalArgumentException("invalid preservation world"); }
        if (!affected.containsAll(residentHolders)) { throw new IllegalArgumentException("resident holder is outside reconfigured world"); }
        var paths = new HashSet<List<Long>>();
        var routes = new HashSet<String>();
        for (var worker : workers) {
            if (!routes.add(worker.lane()) || worker.degree() < 1 || worker.physicalBlockBytes() < 1 || worker.freeBytes() < 0
                    || !Double.isFinite(worker.workload()) || worker.workload() < 0) {
                throw new IllegalArgumentException("invalid preservation inventory");
            }
        }
        for (var demand : demands) {
            if (!Double.isFinite(demand.access()) || !Double.isFinite(demand.missed()) || demand.access() < 0 || demand.missed() < 0
                    || demand.path().size() > 8192 || new HashSet<>(demand.path()).size() != demand.path().size()) {
                throw new IllegalArgumentException("invalid preservation demand");
            }
            if ((demand.access() > 0 || demand.missed() > 0) && !demand.path().isEmpty()
                    && workers.stream().anyMatch(worker -> affected.contains(worker.lane()) && available(worker, demand.path()))) {
                paths.add(demand.path());
            }
        }
        var maximal = paths.stream().filter(path -> paths.stream().noneMatch(other -> other.size() > path.size()
                && other.subList(0, path.size()).equals(path))).sorted(Comparator.comparing(Object::toString)).toList();
        if (maximal.size() > limits.maxPaths()) { throw new IllegalStateException("valuable preservation paths exceed operation budget"); }
        var required = new ArrayList<Retained>();
        var copies = new ArrayList<Proposal>();
        var protectedKeys = maximal.stream().flatMap(List::stream).collect(java.util.stream.Collectors.toSet());
        Map<String, Set<Long>> reclaimed = new HashMap<>();
        Map<String, Set<Long>> installed = new HashMap<>();
        Map<String, Long> reserved = new HashMap<>();
        long physical = 0;
        long wire = 0;
        for (var path : maximal) {
            var existing = workers.stream().filter(worker -> (!affected.contains(worker.lane()) || residentHolders.contains(worker.lane()))
                            && available(worker, path))
                    .min(Comparator.comparingDouble(Worker::workload).thenComparing(Worker::lane));
            if (existing.isPresent()) {
                required.add(new Retained(path, existing.get().lane()));
                continue;
            }
            record Choice(Worker source, Worker destination, List<Long> missing, long bytes, List<Long> reclaim,
                          boolean fullRemainder) { }
            var choices = new ArrayList<Choice>();
            for (var destination : workers) {
                if (affected.contains(destination.lane()) || path.stream().anyMatch(destination.unavailableBlocks()::contains)) { continue; }
                var present = new HashSet<>(destination.blocks());
                present.addAll(installed.getOrDefault(destination.lane(), Set.of()));
                var missing = path.stream().filter(key -> !present.contains(key)).toList();
                long bytes = Math.multiplyExact(destination.physicalBlockBytes(), missing.size());
                var alreadyReclaimed = reclaimed.getOrDefault(destination.lane(), Set.of());
                long remaining = Math.addExact(destination.freeBytes(),
                        Math.multiplyExact(alreadyReclaimed.size(), destination.physicalBlockBytes()))
                        - reserved.getOrDefault(destination.lane(), 0L);
                long deficit = Math.max(0, Math.addExact(bytes, limits.safetyBytes()) - remaining);
                long needed = deficit / destination.physicalBlockBytes() + (deficit % destination.physicalBlockBytes() == 0 ? 0 : 1);
                var rejected = rejectedReclaims.getOrDefault(destination.lane(), List.of());
                boolean fullRemainder = needed > 0 && !rejected.isEmpty() && reclaimable.containsKey(destination.lane());
                // Required paths contain every ancestor. Their inventory complement
                // includes unknown descendants omitted by the observer's inferred order.
                if (fullRemainder && destination.unavailableBlocks().stream().anyMatch(key -> !protectedKeys.contains(key))) { continue; }
                var candidates = fullRemainder ? destination.blocks().stream().sorted().toList()
                        : reclaimable.getOrDefault(destination.lane(), List.of());
                var reclaim = candidates.stream()
                        .filter(key -> destination.blocks().contains(key) && !protectedKeys.contains(key)
                                && !destination.unavailableBlocks().contains(key) && !alreadyReclaimed.contains(key))
                        .distinct().limit(fullRemainder ? Long.MAX_VALUE : needed).toList();
                if (reclaim.size() < needed || alreadyReclaimed.size() + reclaim.size() > 8192
                        || !reclaim.isEmpty() && rejected.stream().anyMatch(keys -> Set.copyOf(keys).equals(Set.copyOf(reclaim)))) { continue; }
                for (var source : workers) {
                    if (affected.contains(source.lane()) && available(source, path) && compatible.test(source, destination)) {
                        choices.add(new Choice(source, destination, missing, bytes, reclaim, fullRemainder));
                    }
                }
            }
            var choice = choices.stream().min(Comparator.comparingInt((Choice candidate) ->
                            candidate.reclaim().isEmpty() ? 0 : candidate.fullRemainder() ? 1 : 2)
                    .thenComparingInt(candidate -> candidate.reclaim().size())
                    .thenComparingInt(candidate -> candidate.missing().size())
                    .thenComparingDouble(candidate -> candidate.destination().workload())
                    .thenComparingLong(Choice::bytes).thenComparingDouble(candidate -> candidate.source().workload())
                    .thenComparing(candidate -> candidate.destination().lane()).thenComparing(candidate -> candidate.source().lane()))
                    .orElseThrow(() -> new IllegalStateException("no compatible unaffected preservation destination"));
            physical = Math.addExact(physical, choice.bytes());
            wire = Math.addExact(wire, choice.bytes());
            if (physical > limits.destinationBytes() || wire > limits.wireBytes()) {
                throw new IllegalStateException("valuable preservation exceeds byte budget");
            }
            var destination = choice.destination();
            String kind = choice.source().degree() == destination.degree() ? "intra" : "inter";
            if (!choice.missing().isEmpty()) {
                copies.add(new Proposal(kind, path, destination.degree(), choice.source().lane(), destination.lane(), choice.missing(),
                        choice.bytes(), choice.bytes(), 1, kind.equals("intra") ? 1 : 0));
            }
            required.add(new Retained(path, destination.lane()));
            installed.computeIfAbsent(destination.lane(), ignored -> new HashSet<>()).addAll(path);
            reserved.merge(destination.lane(), choice.bytes(), Math::addExact);
            reclaimed.computeIfAbsent(destination.lane(), ignored -> new HashSet<>()).addAll(choice.reclaim());
        }
        var reclaims = reclaimed.entrySet().stream().filter(entry -> !entry.getValue().isEmpty())
                .sorted(Map.Entry.comparingByKey()).map(entry -> new Reclaim(entry.getKey(), entry.getValue().stream().sorted().toList())).toList();
        return new Plan(required, copies, physical, wire, reclaims);
    }

    private static boolean available(Worker worker, List<Long> path) {
        return worker.blocks().containsAll(path) && path.stream().noneMatch(worker.unavailableBlocks()::contains);
    }
}
