package org.flexlb.cache.prefix;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class PrefixReplicaPlanner {
    public record Worker(String lane, String replicaDomain, int degree, Set<Long> blocks,
                         long physicalBlockBytes, long freeBytes, double workload, Set<Long> unavailableBlocks) {
        public Worker(String lane, String replicaDomain, int degree, Set<Long> blocks,
                      long physicalBlockBytes, long freeBytes, double workload) {
            this(lane, replicaDomain, degree, blocks, physicalBlockBytes, freeBytes, workload, Set.of());
        }
        public Worker {
            blocks = Set.copyOf(blocks);
            unavailableBlocks = Set.copyOf(unavailableBlocks);
        }
    }
    public record Demand(List<Long> path, int degree, double access, double missed) {
        public Demand { path = path instanceof PrefixPath ? path : List.copyOf(path); }
    }
    public record Limits(double inter, double intra, double delete, int perList,
                         long destinationBytes, long wireBytes, long safetyBytes) {
        public Limits {
            if (!Double.isFinite(inter) || inter < 0 || !Double.isFinite(intra) || intra <= 0
                    || !Double.isFinite(delete) || delete < 0 || delete >= intra
                    || perList < 0 || destinationBytes < 0 || wireBytes < 0 || safetyBytes < 0) {
                throw new IllegalArgumentException("invalid planner limits");
            }
        }
    }
    public record Proposal(String kind, List<Long> path, int degree, String source, String destination,
                            List<Long> missingBlocks, long destinationPhysicalBytes, long estimatedWireBytes,
                            double priority, int replicas, List<Long> reclaimKeys) {
        public Proposal(String kind, List<Long> path, int degree, String source, String destination,
                        List<Long> missingBlocks, long destinationPhysicalBytes, long estimatedWireBytes,
                        double priority, int replicas) {
            this(kind, path, degree, source, destination, missingBlocks, destinationPhysicalBytes, estimatedWireBytes,
                    priority, replicas, "reclaim".equals(kind) ? path : List.of());
        }
        public Proposal {
            path = List.copyOf(path);
            missingBlocks = List.copyOf(missingBlocks);
            reclaimKeys = List.copyOf(reclaimKeys);
        }
    }

    public List<Proposal> plan(List<Demand> demands, List<Worker> workers, Limits limits) {
        return plan(demands, workers, limits, PrefixCacheSettings.ACTION_KINDS);
    }

    public List<Proposal> plan(List<Demand> demands, List<Worker> workers, Limits limits, Set<String> enabledActions) {
        enabledActions = Set.copyOf(enabledActions);
        if (!PrefixCacheSettings.ACTION_KINDS.containsAll(enabledActions)) {
            throw new IllegalArgumentException("unknown prefix action kind");
        }
        if (enabledActions.isEmpty() || limits.perList() == 0) { return List.of(); }
        boolean includeReclaim = enabledActions.contains("reclaim");
        if (!includeReclaim) {
            boolean includeInter = enabledActions.contains("inter");
            boolean includeIntra = enabledActions.contains("intra");
            Set<Integer> allocatableDegrees = workers.stream()
                    .filter(worker -> worker.physicalBlockBytes() > 0
                            && worker.physicalBlockBytes() <= limits.destinationBytes()
                            && worker.physicalBlockBytes() <= limits.wireBytes()
                            && worker.physicalBlockBytes() <= Math.max(0, worker.freeBytes() - limits.safetyBytes()))
                    .map(Worker::degree).collect(java.util.stream.Collectors.toSet());
            if (allocatableDegrees.isEmpty()) { return List.of(); }
            demands = demands.stream().filter(demand -> allocatableDegrees.contains(demand.degree())
                    && (includeInter && demand.missed() > limits.inter()
                    || includeIntra && demand.access() > limits.intra())).toList();
            if (demands.isEmpty()) { return List.of(); }
        }
        var demandIndex = new DemandIndex(demands, workers, includeReclaim);
        Map<String, List<ClassifiedDemand>> classified = new HashMap<>();
        for (String kind : enabledActions) { classified.put(kind, new ArrayList<>()); }
        for (Demand demand : demands) {
            if (demand.path().isEmpty()) { continue; }
            var node = demandIndex.nodes.get(demand);
            int replicas = node.holders.isEmpty() ? 0 : (int) node.holders.stream()
                    .filter(worker -> worker.degree() == demand.degree()).map(Worker::replicaDomain).distinct().count();
            String action = replicas == 0
                    ? demand.missed() > limits.inter() ? "inter" : "none"
                    : demand.access() > replicas * limits.intra() ? "intra"
                    : demand.access() < replicas * limits.delete() ? "reclaim" : "none";
            if (enabledActions.contains(action)) {
                classified.get(action).add(new ClassifiedDemand(demand, node, replicas));
            }
        }
        Map<String, Set<Long>> plannedBlocks = new HashMap<>();
        Map<String, Long> reservedBytes = new HashMap<>();
        List<Proposal> selected = new ArrayList<>();
        long destinationBudget = limits.destinationBytes();
        long wireBudget = limits.wireBytes();
        for (String kind : List.of("inter", "intra", "reclaim")) {
            if (!enabledActions.contains(kind)) { continue; }
            List<Proposal> candidates = new ArrayList<>();
            for (ClassifiedDemand classification : classified.get(kind)) {
                Demand demand = classification.demand();
                var node = classification.node();
                List<Worker> holders = node.holders;
                int replicas = classification.replicas();
                if (kind.equals("reclaim")) {
                    var victim = holders.stream().filter(worker -> worker.degree() == demand.degree())
                            .filter(worker -> node.descendants(worker) == 0)
                            .filter(worker -> worker.unavailableBlocks().isEmpty()
                                    || demand.path().stream().noneMatch(worker.unavailableBlocks()::contains))
                            .min(Comparator.comparingLong(Worker::freeBytes).thenComparing(Worker::lane)).orElse(null);
                    if (victim != null) {
                        candidates.add(new Proposal(kind, demand.path(), demand.degree(), "", victim.lane(),
                                List.of(), 0, 0, demand.access() / replicas, replicas,
                                reclaimSuffix(victim, demand, node, limits.delete())));
                    }
                    continue;
                }
                var source = holders.stream().filter(worker -> kind.equals("inter") || worker.degree() == demand.degree())
                        .filter(worker -> worker.unavailableBlocks().isEmpty()
                                || demand.path().stream().noneMatch(worker.unavailableBlocks()::contains))
                        .min(Comparator.comparingDouble(Worker::workload).thenComparing(Worker::lane)).orElse(null);
                if (source == null) { continue; }
                List<Worker> destinations = workers.stream()
                        .filter(worker -> worker.degree() == demand.degree() && worker.physicalBlockBytes() > 0)
                        .filter(worker -> worker.unavailableBlocks().isEmpty()
                                || demand.path().stream().noneMatch(worker.unavailableBlocks()::contains))
                        .filter(worker -> holders.stream().noneMatch(holder -> holder.replicaDomain().equals(worker.replicaDomain())))
                        .filter(worker -> fitsPhysicalBudget(demand.path(), worker,
                                Math.max(0, worker.freeBytes() - limits.safetyBytes())))
                        .toList();
                long minimumBytes = destinations.stream().mapToLong(worker -> physicalBytes(demand.path(), worker, Set.of()))
                        .min().orElse(0);
                var destination = destinations.stream().min(Comparator
                        .<Worker>comparingInt(worker -> contiguous(demand.path(), worker.blocks())).reversed()
                        .thenComparingDouble(Worker::workload).thenComparing(Worker::lane)).orElse(null);
                if (destination == null || minimumBytes <= 0) { continue; }
                List<Long> missing = missing(demand.path(), destination, Set.of());
                long bytes = physicalBytes(demand.path(), destination, Set.of());
                candidates.add(new Proposal(kind, demand.path(), demand.degree(), source.lane(), destination.lane(),
                        missing, bytes, bytes, kind.equals("inter") ? demand.missed() / minimumBytes
                        : demand.access() / replicas, replicas));
            }
            Comparator<Proposal> order = Comparator.comparingDouble(Proposal::priority);
            if (!kind.equals("reclaim")) { order = order.reversed(); }
            candidates.sort(order.thenComparingLong(Proposal::destinationPhysicalBytes)
                    .thenComparing(proposal -> proposal.path().toString()).thenComparingInt(Proposal::degree)
                    .thenComparing(Proposal::destination));
            int count = 0;
            for (Proposal candidate : candidates) {
                if (kind.equals("reclaim")) {
                    if (count >= limits.perList()) { continue; }
                    if (selected.stream().anyMatch(proposal -> !proposal.kind().equals("reclaim")
                            && (proposal.destination().equals(candidate.destination()) || proposal.source().equals(candidate.destination()))
                            && overlaps(proposal.path(), candidate.path()))) { continue; }
                    selected.add(candidate);
                    count++;
                    continue;
                }
                int mergeIndex = -1;
                for (int index = 0; index < selected.size(); index++) {
                    Proposal previous = selected.get(index);
                    if (previous.kind().equals(candidate.kind()) && previous.degree() == candidate.degree()
                            && previous.source().equals(candidate.source())
                            && previous.destination().equals(candidate.destination())
                            && overlaps(previous.path(), candidate.path())) {
                        mergeIndex = index;
                        break;
                    }
                }
                if (mergeIndex < 0 && count >= limits.perList()) { continue; }
                Worker destination = workers.stream().filter(worker -> worker.lane().equals(candidate.destination())).findFirst().orElseThrow();
                Set<Long> alreadyPlanned = plannedBlocks.computeIfAbsent(candidate.destination(), ignored -> new HashSet<>());
                List<Long> missing = missing(candidate.path(), destination, alreadyPlanned);
                if (missing.isEmpty()) { continue; }
                long bytes = physicalBytes(candidate.path(), destination, alreadyPlanned);
                long reserved = reservedBytes.getOrDefault(candidate.destination(), 0L);
                if (bytes > destinationBudget || bytes > wireBudget
                        || bytes > Math.max(0, destination.freeBytes() - limits.safetyBytes() - reserved)) { continue; }
                alreadyPlanned.addAll(missing);
                reservedBytes.put(candidate.destination(), reserved + bytes);
                destinationBudget -= bytes;
                wireBudget -= bytes;
                if (mergeIndex >= 0) {
                    Proposal previous = selected.get(mergeIndex);
                    List<Long> mergedMissing = new ArrayList<>(previous.missingBlocks());
                    mergedMissing.addAll(missing);
                    selected.set(mergeIndex, new Proposal(candidate.kind(), candidate.path(), candidate.degree(),
                            candidate.source(), candidate.destination(), List.copyOf(mergedMissing),
                            previous.destinationPhysicalBytes() + bytes, previous.estimatedWireBytes() + bytes,
                            previous.priority(), candidate.replicas()));
                } else {
                    selected.add(new Proposal(candidate.kind(), candidate.path(), candidate.degree(), candidate.source(),
                            candidate.destination(), missing, bytes, bytes, candidate.priority(), candidate.replicas()));
                    count++;
                }
            }
        }
        return List.copyOf(selected);
    }

    private static List<Long> reclaimSuffix(Worker victim, Demand leaf, DemandNode leafNode, double threshold) {
        int start = leaf.path().size() - 1;
        var ancestor = leafNode.parent;
        while (start > 0) {
            var demand = ancestor.demands.get(leaf.degree());
            long replicas = ancestor.holders.stream().filter(worker -> worker.degree() == leaf.degree())
                    .map(Worker::replicaDomain).distinct().count();
            boolean otherBranch = ancestor.descendants(victim) > leafNode.pathTerminals - ancestor.pathTerminals;
            if (demand == null || demand.access() >= replicas * threshold || otherBranch) { break; }
            start--;
            ancestor = ancestor.parent;
        }
        return List.copyOf(leaf.path().subList(start, leaf.path().size()));
    }

    private static final class DemandNode {
        final DemandNode parent;
        final Map<Long, DemandNode> children = new HashMap<>();
        final Map<Integer, Demand> demands = new HashMap<>();
        final Map<String, Integer> subtreeTerminals = new HashMap<>();
        List<Worker> holders;
        int pathTerminals;

        DemandNode(DemandNode parent, List<Worker> holders) {
            this.parent = parent;
            this.holders = holders;
        }

        int descendants(Worker worker) {
            return subtreeTerminals.getOrDefault(worker.lane(), 0) - (demands.isEmpty() ? 0 : 1);
        }
    }

    private record ClassifiedDemand(Demand demand, DemandNode node, int replicas) { }

    private static final class DemandIndex {
        final Map<Demand, DemandNode> nodes = new java.util.IdentityHashMap<>();
        final Map<PrefixPath, DemandNode> prefixes = new java.util.IdentityHashMap<>();

        DemandIndex(List<Demand> demands, List<Worker> workers, boolean includeReclaim) {
            var root = new DemandNode(null, workers);
            var ordered = new ArrayList<DemandNode>();
            ordered.add(root);
            for (var demand : demands) {
                DemandNode node;
                if (demand.path() instanceof PrefixPath prefix) {
                    node = prefixNode(prefix, root, ordered);
                } else {
                    node = root;
                    for (long block : demand.path()) { node = childNode(node, block, ordered); }
                }
                node.demands.putIfAbsent(demand.degree(), demand);
                nodes.put(demand, node);
            }
            if (!includeReclaim) { return; }
            for (var node : ordered) {
                int terminal = node.demands.isEmpty() ? 0 : 1;
                node.pathTerminals = (node.parent == null ? 0 : node.parent.pathTerminals) + terminal;
                for (var worker : node.holders) { node.subtreeTerminals.put(worker.lane(), terminal); }
            }
            for (int position = ordered.size() - 1; position > 0; position--) {
                var node = ordered.get(position);
                node.subtreeTerminals.forEach((lane, count) -> node.parent.subtreeTerminals.merge(lane, count, Integer::sum));
            }
        }

        private DemandNode prefixNode(PrefixPath prefix, DemandNode root, List<DemandNode> ordered) {
            var found = prefixes.get(prefix);
            if (found != null) { return found; }
            var missing = new java.util.ArrayDeque<PrefixPath>();
            PrefixPath ancestor = prefix;
            while (ancestor != null && !prefixes.containsKey(ancestor)) {
                missing.push(ancestor);
                ancestor = ancestor.parent();
            }
            var node = ancestor == null ? root : prefixes.get(ancestor);
            while (!missing.isEmpty()) {
                var path = missing.pop();
                node = childNode(node, path.block(), ordered);
                prefixes.put(path, node);
            }
            return node;
        }

        private DemandNode childNode(DemandNode parent, long block, List<DemandNode> ordered) {
            var child = parent.children.get(block);
            if (child == null) {
                child = new DemandNode(parent, parent.holders.stream()
                        .filter(worker -> worker.blocks().contains(block)).toList());
                parent.children.put(block, child);
                ordered.add(child);
            }
            return child;
        }
    }

    private static boolean overlaps(List<Long> left, List<Long> right) {
        int shared = Math.min(left.size(), right.size());
        return left.subList(0, shared).equals(right.subList(0, shared));
    }

    private static int contiguous(List<Long> path, Set<Long> blocks) {
        int length = 0;
        while (length < path.size() && blocks.contains(path.get(length))) { length++; }
        return length;
    }

    private static List<Long> missing(List<Long> path, Worker destination, Set<Long> planned) {
        return path.stream().filter(block -> !destination.blocks().contains(block) && !planned.contains(block)).toList();
    }

    private static long physicalBytes(List<Long> path, Worker destination, Set<Long> planned) {
        int missingBlocks = 0;
        for (Long block : path) {
            if (!destination.blocks().contains(block) && !planned.contains(block)) { missingBlocks++; }
        }
        try {
            return Math.multiplyExact(missingBlocks, destination.physicalBlockBytes());
        } catch (ArithmeticException exception) {
            return Long.MAX_VALUE;
        }
    }

    static boolean fitsPhysicalBudget(List<Long> path, Worker destination, long budget) {
        if (budget == Long.MAX_VALUE) { return true; }
        long remainingBlocks = budget / destination.physicalBlockBytes();
        for (Long block : path) {
            if (!destination.blocks().contains(block) && --remainingBlocks < 0) { return false; }
        }
        return true;
    }
}
