package org.flexlb.cache.prefix;

import org.flexlb.dao.cache.CacheReportV2;
import java.time.Clock;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

public final class PrefixObserver {
    public record Node(long hash, int depth) { }
    public record DemandKey(Node node, int degree) { }
    public record Demand(double access, double missed) { }
    private static final Demand ZERO_DEMAND = new Demand(0, 0);
    public record Candidate(String kind, Node node, int degree, int replicas,
                            double value, List<String> sources) { }
    static final class CandidateCollector {
        private record Ranked(Candidate candidate, long ordinal) { }
        private final String kind;
        private final int limit;
        private long ordinal;
        private final Comparator<Ranked> order;
        private final PriorityQueue<Ranked> retained;

        CandidateCollector(String kind, int limit) {
            if (limit < 0 || !List.of("inter", "intra", "reclaim").contains(kind)) {
                throw new IllegalArgumentException("Invalid candidate collector");
            }
            this.kind = kind;
            this.limit = limit;
            order = (left, right) -> {
                Candidate candidate = left.candidate();
                int compared = compare(candidate.node(), candidate.degree(), candidate.value(), right.candidate());
                return compared != 0 ? compared : Long.compare(left.ordinal(), right.ordinal());
            };
            retained = new PriorityQueue<>(order.reversed());
        }

        private int compare(Node node, int degree, double value, Candidate other) {
            int compared = Double.compare(value, other.value());
            if (!kind.equals("reclaim")) { compared = -compared; }
            if (compared == 0) { compared = Integer.compare(node.depth(), other.node().depth()); }
            if (compared == 0) { compared = Long.compare(node.hash(), other.node().hash()); }
            if (compared == 0) { compared = Integer.compare(degree, other.degree()); }
            return compared;
        }

        void offer(Node node, int degree, int replicas, double value, List<String> sources) {
            if (limit == 0) { return; }
            if (retained.size() == limit) {
                if (compare(node, degree, value, retained.peek().candidate()) >= 0) { return; }
                retained.poll();
            }
            retained.add(new Ranked(new Candidate(kind, node, degree, replicas, value, sources), ordinal++));
        }

        List<Candidate> snapshot() {
            return retained.stream().sorted(order).map(Ranked::candidate).toList();
        }
    }
    public record Observation(String attemptId, long requestId, long inputTokens, int blockSize, List<Node> path,
                              Map<DemandKey, Double> missed, long createdMillis) { }
    public record ExecutionKey(String worker, int lane, String epoch) { }
    public record Scan(long sequence, long createdMillis, String namespace, List<PrefixReplicaPlanner.Proposal> proposals,
                       Map<String, ConfirmedCacheDirectory.Snapshot> inventory, Map<String, Double> workloads) {
        public Scan {
            proposals = List.copyOf(proposals);
            inventory = Map.copyOf(inventory);
            workloads = Map.copyOf(workloads);
        }
    }
    private record Pending(Observation observation, String route, String epoch, String layout, int degree) { }
    private record ObservationInput(String attemptId, long requestId, long inputTokens, int blockSize,
                                    List<Long> hashes, int preferredDegree, PrefillCostModel model, String route,
                                    Map<String, ConfirmedCacheDirectory.Snapshot> view, long createdMillis) { }
    private final int observationQueueCapacity;
    private final ArrayDeque<ObservationInput> observationQueue = new ArrayDeque<>();
    private long observationQueueGeneration;
    private long observationEnqueued;
    private long observationFallbacks;
    private int observationMaxQueueDepth;
    private long observationProcessed;
    private long observationQueueExpired;
    private long observationQueueCancelled;
    private long observationQueueFailures;
    private long observationMaxQueueAgeMillis;
    private String observationLastFailure = "";
    private final PrefixCacheSettings settings;
    private final Clock clock;
    private final ConfirmedCacheDirectory directory;
    private final boolean routingHintsEnabled;
    private final PrefixRoutingHistory routingHistory;
    private final PrefixLatencyMetrics reportLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics observationLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics acceptanceLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics scanCaptureWaitLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics scanCaptureHoldLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics scanFinalizeWaitLatency = new PrefixLatencyMetrics();
    private final PrefixLatencyMetrics scanFinalizeHoldLatency = new PrefixLatencyMetrics();
    private long hintedLookups;
    private long hintedBlocks;
    private final Map<Node, Node> parents = new HashMap<>();
    private final Map<Node, Long> lastSeenMillis = new HashMap<>();
    private long oldestSeenMillis = Long.MAX_VALUE;
    private boolean oldestSeenKnown;
    private final Map<Long, Node> identities = new HashMap<>();
    private final Set<Node> previouslyCached = new HashSet<>();
    private final Map<String, Double> workloads = new HashMap<>();
    private final Map<String, Double> queueSeconds = new HashMap<>();
    private final Map<DemandKey, Demand> demands = new HashMap<>();
    private final Map<String, Pending> pending = new LinkedHashMap<>();
    private final Map<ExecutionKey, Long> eventWatermarks = new HashMap<>();
    private long overflow;
    private long unmatchedEvents;
    private long negativeMarginals;
    private long expiredObservations;
    private long acceptedObservations;
    private long committedAttempts;
    private long lastScanMillis;
    private long lastScanNanos;
    private long scans;
    private long scanSequence;
    private long resets;
    private PrefixStructureMetrics lastStructureMetrics;
    private PrefixCopyMetrics lastCopyMetrics;
    private final Object scanLock = new Object();
    private Scan lastScan;
    private Map<String, Object> actionStatus = Map.of();
    private List<Candidate> lastPlan = List.of();
    private List<PrefixReplicaPlanner.Proposal> lastProposals = List.of();

    public PrefixObserver(PrefixCacheSettings settings, Clock clock) {
        this(settings, clock, false);
    }

    public PrefixObserver(PrefixCacheSettings settings, Clock clock, boolean routingHintsEnabled) {
        this(settings, clock, routingHintsEnabled, false);
    }

    public PrefixObserver(PrefixCacheSettings settings, Clock clock, boolean routingHintsEnabled,
                          boolean routingHistoryEnabled) {
        this(settings, clock, routingHintsEnabled, routingHistoryEnabled, 65536);
    }

    public PrefixObserver(PrefixCacheSettings settings, Clock clock, boolean routingHintsEnabled,
                          boolean routingHistoryEnabled, int routingHistoryCapacity) {
        this(settings, clock, routingHintsEnabled, routingHistoryEnabled, routingHistoryCapacity, 0);
    }

    public PrefixObserver(PrefixCacheSettings settings, Clock clock, boolean routingHintsEnabled,
                          boolean routingHistoryEnabled, int routingHistoryCapacity, int observationQueueCapacity) {
        if (routingHistoryCapacity < 1) { throw new IllegalArgumentException("history capacity must be positive"); }
        if (observationQueueCapacity < 0 || observationQueueCapacity > Math.min(4096, settings.maxObservations())) {
            throw new IllegalArgumentException("observation queue capacity must be between zero and min(4096, maxObservations)");
        }
        if (observationQueueCapacity > 0 && (routingHintsEnabled || routingHistoryEnabled)) {
            throw new IllegalArgumentException("observation handoff requires confirmed-only routing");
        }
        this.observationQueueCapacity = observationQueueCapacity;
        this.settings = settings;
        this.clock = clock;
        this.routingHintsEnabled = routingHintsEnabled;
        this.routingHistory = routingHistoryEnabled ? new PrefixRoutingHistory(routingHistoryCapacity) : null;
        this.directory = new ConfirmedCacheDirectory(settings.namespace(), clock, settings.reportTtlMillis());
        lastScanMillis = clock.millis();
        lastScan = new Scan(0, lastScanMillis, settings.namespace(), List.of(), Map.of(), Map.of());
    }

    public PrefixCacheSettings settings() { return settings; }
    public ConfirmedCacheDirectory directory() { return directory; }

    public void observeAndAccept(String attemptId, long requestId, long inputTokens, int blockSize,
                                 List<Long> hashes, int preferredDegree, PrefillCostModel model, String route,
                                 Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        if (observationQueueCapacity == 0) {
            accepted(observe(attemptId, requestId, inputTokens, blockSize, hashes, preferredDegree, model, view), route, view);
            return;
        }
        if (attemptId == null || attemptId.isBlank() || hashes == null) { return; }
        var input = new ObservationInput(attemptId, requestId, inputTokens, blockSize, List.copyOf(hashes),
                preferredDegree, model, route, Map.copyOf(view), clock.millis());
        long generation;
        synchronized (observationQueue) {
            if (observationQueue.size() < observationQueueCapacity) {
                observationQueue.addLast(input);
                observationEnqueued++;
                observationMaxQueueDepth = Math.max(observationMaxQueueDepth, observationQueue.size());
                return;
            }
            observationFallbacks++;
            generation = observationQueueGeneration;
        }
        synchronized (this) {
            synchronized (observationQueue) {
                if (generation != observationQueueGeneration) { return; }
            }
            drainObservations();
            processObservation(input);
        }
    }

    private void drainObservations() {
        if (observationQueueCapacity == 0) { return; }
        List<ObservationInput> batch;
        synchronized (observationQueue) {
            if (observationQueue.isEmpty()) { return; }
            batch = List.copyOf(observationQueue);
            observationQueue.clear();
        }
        for (var input : batch) { processObservation(input); }
    }

    private void processObservation(ObservationInput input) {
        long age = clock.millis() - input.createdMillis();
        observationMaxQueueAgeMillis = Math.max(observationMaxQueueAgeMillis, age);
        if (age < 0 || age > settings.observationTtlMillis()) {
            observationQueueExpired++;
            expiredObservations++;
            return;
        }
        try {
            var observation = observationLatency.measure(() -> observeLocked(input.attemptId(), input.requestId(),
                    input.inputTokens(), input.blockSize(), input.hashes(), input.preferredDegree(), input.model(),
                    input.view(), input.createdMillis()));
            accepted(observation, input.route(), input.view());
            observationProcessed++;
        } catch (RuntimeException exception) {
            observationQueueFailures++;
            String description = exception.getClass().getSimpleName() + ": " + exception.getMessage();
            observationLastFailure = description.substring(0, Math.min(512, description.length()));
        }
    }

    public Map<String, Integer> routingMatches(Map<String, ConfirmedCacheDirectory.Snapshot> view,
                                               List<Long> keys, List<String> candidates) {
        var confirmed = directory.measuredMatches(view, keys, candidates);
        if ((!routingHintsEnabled && routingHistory == null)
                || keys == null || candidates == null || keys.isEmpty()) { return confirmed; }
        List<Pending> executions;
        long capturedResets;
        synchronized (this) {
            expirePending();
            executions = routingHintsEnabled ? List.copyOf(pending.values()) : List.of();
            capturedResets = resets;
        }
        var predictions = new HashMap<>(confirmed);
        if (routingHistory != null) { routingHistory.merge(predictions, view, keys, candidates); }
        var eligible = Set.copyOf(candidates);
        long now = clock.millis();
        for (var execution : executions) {
            var snapshot = view.get(execution.route());
            long age = now - execution.observation().createdMillis();
            if (!eligible.contains(execution.route()) || snapshot == null || !snapshot.report().deviceReady()
                    || age < 0 || age >= settings.reportTtlMillis()
                    || !execution.epoch().equals(snapshot.report().epoch())
                    || !execution.layout().equals(snapshot.report().layoutFingerprint())
                    || execution.degree() != snapshot.report().cpDegree()) { continue; }
            var path = execution.observation().path();
            int matched = 0;
            while (matched < keys.size() && matched < path.size()
                    && path.get(matched).hash() == keys.get(matched)
                    && !snapshot.unavailableKeys().contains(keys.get(matched))) {
                matched++;
            }
            predictions.merge(execution.route(), matched, Math::max);
        }
        long additional = predictions.entrySet().stream()
                .mapToLong(entry -> entry.getValue() - confirmed.getOrDefault(entry.getKey(), 0)).sum();
        if (additional > 0) {
            synchronized (this) {
                if (resets == capturedResets) { hintedLookups++; hintedBlocks += additional; }
            }
        }
        return predictions;
    }
    public synchronized void workload(String route, double value) {
        workload(route, value, value);
    }
    public synchronized void workload(String route, double foregroundTasks, double prefillQueueSeconds) {
        workloads.put(route, Math.max(0, foregroundTasks));
        queueSeconds.put(route, Math.max(0, prefillQueueSeconds));
    }
    public synchronized Map<String, Double> workloadSnapshot() { return Map.copyOf(workloads); }

    public ConfirmedCacheDirectory.Result report(String route, CacheReportV2 report) {
        return reportLatency.measure(() -> reportLocked(route, report));
    }

    private synchronized ConfirmedCacheDirectory.Result reportLocked(String route, CacheReportV2 report) {
        drainObservations();
        expirePending();
        var result = directory.apply(route, report);
        if (settings.observe() && (result == ConfirmedCacheDirectory.Result.APPLIED
                || result == ConfirmedCacheDirectory.Result.NO_CHANGE)) {
            ExecutionKey key = new ExecutionKey(report.workerId(), report.dpLane(), report.epoch());
            long watermark = eventWatermarks.getOrDefault(key, 0L);
            for (var event : report.reuseEvents().stream()
                    .sorted(Comparator.comparingLong(CacheReportV2.PrefixReuseEvent::sequence)).toList()) {
                if (event.sequence() <= watermark) { continue; }
                if (event.sequence() > watermark + 1) { unmatchedEvents += event.sequence() - watermark - 1; }
                watermark = event.sequence();
                if (!event.prefill()) { continue; }
                Pending execution = pending.get(event.attemptId());
                if (execution == null || !execution.route().equals(route)
                        || !execution.epoch().equals(report.epoch())
                        || execution.observation().requestId() != event.requestId()
                        || execution.observation().inputTokens() != event.inputTokens()) {
                    unmatchedEvents++;
                    continue;
                }
                pending.remove(event.attemptId());
                commit(execution.observation(), report.cpDegree(), event.deviceReuseTokens());
            }
            eventWatermarks.put(key, watermark);
        }
        return result;
    }

    public synchronized Observation observe(String attemptId, long requestId, long inputTokens, int blockSize,
                                              List<Long> hashes, int preferredDegree, PrefillCostModel model) {
        return observe(attemptId, requestId, inputTokens, blockSize, hashes, preferredDegree, model, directory.view());
    }

    public Observation observe(String attemptId, long requestId, long inputTokens, int blockSize,
                               List<Long> hashes, int preferredDegree, PrefillCostModel model,
                               Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        return observationLatency.measure(() -> observeLocked(attemptId, requestId, inputTokens, blockSize,
                hashes, preferredDegree, model, view));
    }

    private synchronized Observation observeLocked(String attemptId, long requestId, long inputTokens, int blockSize,
                                                   List<Long> hashes, int preferredDegree, PrefillCostModel model,
                                                   Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        return observeLocked(attemptId, requestId, inputTokens, blockSize, hashes, preferredDegree, model, view, null);
    }

    private Observation observeLocked(String attemptId, long requestId, long inputTokens, int blockSize,
                                      List<Long> hashes, int preferredDegree, PrefillCostModel model,
                                      Map<String, ConfirmedCacheDirectory.Snapshot> view, Long submittedMillis) {
        if (attemptId == null || attemptId.isBlank() || hashes == null) { return null; }
        long maximum = PrefillCostModel.maxReusableTokens(inputTokens, blockSize) / blockSize;
        int count = (int) Math.min(maximum, hashes.size());
        int completeCount = (int) Math.min(Math.max(0, inputTokens) / blockSize, hashes.size());
        List<Node> path = new ArrayList<>();
        Set<Long> pathHashes = new HashSet<>();
        Node parent = null;
        for (int index = 0; index < completeCount; index++) {
            Node node = new Node(hashes.get(index), index + 1);
            if (!pathHashes.add(node.hash())
                    || (identities.containsKey(node.hash()) && !identities.get(node.hash()).equals(node))
                    || (parents.containsKey(node) && !java.util.Objects.equals(parents.get(node), parent))) {
                throw new IllegalArgumentException("prefix parent/depth conflict");
            }
            path.add(node);
            parent = node;
        }
        long missing = path.stream().filter(node -> !parents.containsKey(node)).count();
        if (parents.size() + missing > settings.maxNodes()) {
            overflow++;
            return null;
        }
        Map<DemandKey, Double> missed = new HashMap<>();
        int globallyCachedDepth = 0;
        int preferredCachedDepth = 0;
        for (var snapshot : view.values()) {
            int matchedDepth = 0;
            while (matchedDepth < completeCount && snapshot.keys().contains(hashes.get(matchedDepth))) {
                matchedDepth++;
            }
            globallyCachedDepth = Math.max(globallyCachedDepth, matchedDepth);
            if (snapshot.report().cpDegree() == preferredDegree) {
                preferredCachedDepth = Math.max(preferredCachedDepth, matchedDepth);
            }
        }
        for (Node node : path) {
            boolean globallyCached = node.depth() <= globallyCachedDepth;
            if (globallyCached) { previouslyCached.add(node); }
            if (node.depth() > count) { continue; }
            boolean preferredCached = node.depth() <= preferredCachedDepth;
            if (previouslyCached.contains(node) && !preferredCached) {
                long nodeTokens = (long) node.depth() * blockSize;
                long parentTokens = nodeTokens - blockSize;
                double marginal = model.seconds(preferredDegree, inputTokens - parentTokens, parentTokens)
                        - model.seconds(preferredDegree, inputTokens - nodeTokens, nodeTokens);
                if (marginal >= 0 && Double.isFinite(marginal)) {
                    missed.put(new DemandKey(node, preferredDegree), marginal);
                } else {
                    negativeMarginals++;
                }
            }
        }
        parent = null;
        for (Node node : path) {
            parents.put(node, parent);
            long seenMillis = submittedMillis == null ? clock.millis() : submittedMillis;
            if (submittedMillis == null) { lastSeenMillis.put(node, seenMillis); }
            else { lastSeenMillis.merge(node, seenMillis, Math::max); }
            oldestSeenMillis = Math.min(oldestSeenMillis, seenMillis);
            identities.put(node.hash(), node);
            parent = node;
        }
        return new Observation(attemptId, requestId, inputTokens, blockSize, List.copyOf(path.subList(0, count)),
                Map.copyOf(missed), submittedMillis == null ? clock.millis() : submittedMillis);
    }

    public synchronized void accepted(Observation observation, String route) {
        accepted(observation, route, directory.view());
    }

    public void accepted(Observation observation, String route,
                         Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        acceptanceLatency.measure(() -> {
            acceptedLocked(observation, route, view);
            return Boolean.TRUE;
        });
    }

    private synchronized void acceptedLocked(Observation observation, String route,
                                            Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        if (observation == null) { return; }
        expirePending();
        var snapshot = view.get(route);
        if (snapshot == null || pending.size() >= settings.maxObservations()) {
            overflow++;
            return;
        }
        if (pending.putIfAbsent(observation.attemptId(), new Pending(observation, route, snapshot.report().epoch(),
                snapshot.report().layoutFingerprint(), snapshot.report().cpDegree())) == null) {
            if (routingHistory != null) { routingHistory.remember(route, snapshot, observation.path()); }
            acceptedObservations++;
        }
    }

    public synchronized void cancel(long requestId) {
        synchronized (observationQueue) {
            int before = observationQueue.size();
            observationQueue.removeIf(input -> input.requestId() == requestId);
            observationQueueCancelled += before - observationQueue.size();
        }
        if (routingHistory != null) {
            pending.values().stream().filter(execution -> execution.observation().requestId() == requestId)
                    .forEach(execution -> routingHistory.invalidate(execution.route()));
        }
        pending.entrySet().removeIf(entry -> entry.getValue().observation().requestId() == requestId);
    }

    private void commit(Observation observation, int actualDegree, long deviceReuseTokens) {
        committedAttempts++;
        long reusable = Math.min(Math.max(0, deviceReuseTokens),
                PrefillCostModel.maxReusableTokens(observation.inputTokens(), observation.blockSize()));
        for (Node node : observation.path()) {
            if ((long) node.depth() * observation.blockSize() <= reusable) {
                DemandKey key = new DemandKey(node, actualDegree);
                Demand old = demands.getOrDefault(key, new Demand(0, 0));
                demands.put(key, new Demand(old.access() + 1, old.missed()));
            }
        }
        observation.missed().forEach((key, benefit) -> {
            Demand old = demands.getOrDefault(key, new Demand(0, 0));
            demands.put(key, new Demand(old.access(), old.missed() + benefit));
        });
    }

    public List<Candidate> scan() {
        synchronized (scanLock) { return scanOnce(); }
    }

    public Scan scanSnapshot() {
        synchronized (scanLock) {
            scanOnce();
            synchronized (this) { return lastScan; }
        }
    }

    public synchronized void actionStatus(Map<String, Object> status) { actionStatus = Map.copyOf(status); }

    static List<String> filterHolders(List<String> parents, long key,
                                      Map<String, ConfirmedCacheDirectory.Snapshot> view) {
        ArrayList<String> retained = null;
        for (int index = 0; index < parents.size(); ++index) {
            String route = parents.get(index);
            if (view.get(route).keys().contains(key)) {
                if (retained != null) { retained.add(route); }
            } else if (retained == null) {
                retained = new ArrayList<>(parents.subList(0, index));
            }
        }
        return retained == null ? parents : List.copyOf(retained);
    }

    static int replicaCount(List<String> holders, int degree,
                            Map<String, ConfirmedCacheDirectory.Snapshot> view,
                            Map<String, ConfirmedCacheDirectory.Lane> domains) {
        ConfirmedCacheDirectory.Lane first = null;
        Set<ConfirmedCacheDirectory.Lane> others = null;
        int count = 0;
        for (String route : holders) {
            if (view.get(route).report().cpDegree() != degree) { continue; }
            var domain = domains.get(route);
            if (first == null) {
                first = domain;
                count = 1;
            } else if (!first.equals(domain)) {
                if (others == null) { others = new HashSet<>(); }
                if (others.add(domain)) { count++; }
            }
        }
        return count;
    }

    private static PrefixPath plannerPath(Node node, Map<Node, Node> parents, Map<Node, PrefixPath> paths) {
        if (paths.containsKey(node)) { return paths.get(node); }
        var missing = new ArrayList<Node>();
        Node current = node;
        while (current != null && !paths.containsKey(current)) {
            if (!parents.containsKey(current)) { throw new IllegalStateException("missing prefix ancestor"); }
            missing.add(current);
            current = parents.get(current);
        }
        PrefixPath path = current == null ? null : paths.get(current);
        for (int index = missing.size() - 1; index >= 0; --index) {
            var ancestor = missing.get(index);
            path = new PrefixPath(path, ancestor.hash());
            paths.put(ancestor, path);
        }
        return java.util.Objects.requireNonNull(path);
    }

    private List<Candidate> scanOnce() {
        long startedNanos = System.nanoTime();
        Map<Node, Node> parents;
        Map<DemandKey, Demand> demands;
        Map<String, Double> workloads;
        Map<String, Double> queueSeconds;
        Map<String, ConfirmedCacheDirectory.Snapshot> view;
        long capturedResets;
        long capturedMillis;
        long captureStartedNanos = System.nanoTime();
        synchronized (this) {
            long acquiredNanos = System.nanoTime();
            boolean completed = false;
            try {
                drainObservations();
                expirePending();
                parents = new HashMap<>(this.parents);
                demands = new HashMap<>(this.demands);
                workloads = Map.copyOf(this.workloads);
                queueSeconds = Map.copyOf(this.queueSeconds);
                view = directory.view();
                capturedResets = resets;
                capturedMillis = clock.millis();
                completed = true;
            } finally {
                long finishedNanos = System.nanoTime();
                scanCaptureWaitLatency.recordDuration(acquiredNanos - captureStartedNanos, false);
                scanCaptureHoldLatency.recordDuration(finishedNanos - acquiredNanos, !completed);
            }
        }
        Set<Integer> degrees = new HashSet<>();
        view.values().forEach(snapshot -> degrees.add(snapshot.report().cpDegree()));
        var interCandidates = new CandidateCollector("inter", settings.operationsPerList());
        var intraCandidates = new CandidateCollector("intra", settings.operationsPerList());
        var reclaimCandidates = new CandidateCollector("reclaim", settings.operationsPerList());
        Set<Node> confirmedNodes = new HashSet<>();
        List<PrefixReplicaPlanner.Demand> plannerDemands = new ArrayList<>();
        Map<Node, PrefixPath> paths = new HashMap<>();
        Map<Node, List<String>> completeHolders = new HashMap<>();
        List<String> routes = view.keySet().stream().sorted().toList();
        Map<String, ConfirmedCacheDirectory.Lane> domains = new HashMap<>();
        view.forEach((route, snapshot) -> domains.put(route,
                new ConfirmedCacheDirectory.Lane(snapshot.report().workerId(), snapshot.report().dpLane())));
        boolean includeReclaim = settings.enabledActions().contains("reclaim");
        boolean includeInter = settings.enabledActions().contains("inter");
        boolean includeIntra = settings.enabledActions().contains("intra");
        for (Node node : depthOrderedNodes(parents.keySet())) {
            Node parent = parents.get(node);
            List<String> parentHolders = parent == null ? routes : completeHolders.get(parent);
            List<String> holders = filterHolders(parentHolders, node.hash(), view);
            completeHolders.put(node, holders);
            if (holders.isEmpty()) { continue; }
            confirmedNodes.add(node);
            for (int degree : degrees) {
                int replicas = replicaCount(holders, degree, view, domains);
                Demand demand = demands.getOrDefault(new DemandKey(node, degree), ZERO_DEMAND);
                if (includeReclaim || includeInter && demand.missed() > settings.interThreshold()
                        || includeIntra && demand.access() > settings.intraThreshold()) {
                    plannerDemands.add(new PrefixReplicaPlanner.Demand(plannerPath(node, parents, paths), degree,
                            demand.access(), demand.missed()));
                }
                if (replicas == 0 && demand.missed() > settings.interThreshold() && !holders.isEmpty()) {
                    interCandidates.offer(node, degree, replicas, demand.missed(), holders);
                } else if (replicas > 0 && demand.access() > replicas * settings.intraThreshold()) {
                    intraCandidates.offer(node, degree, replicas, demand.access() / replicas, holders);
                } else if (replicas > 0 && demand.access() < replicas * settings.deleteThreshold()) {
                    reclaimCandidates.offer(node, degree, replicas, demand.access() / replicas, holders);
                }
            }
        }
        List<PrefixReplicaPlanner.Worker> workers = view.entrySet().stream()
                .filter(entry -> entry.getValue().report().deviceReady())
                .map(entry -> new PrefixReplicaPlanner.Worker(entry.getKey(),
                        entry.getValue().report().workerId() + "/" + entry.getValue().report().dpLane(),
                        entry.getValue().report().cpDegree(), entry.getValue().keys(),
                        entry.getValue().report().physicalBlockBytes(), entry.getValue().report().physicalFreeBytes(),
                        queueSeconds.getOrDefault(entry.getKey(), 0.0), entry.getValue().unavailableKeys())).toList();
        var proposals = new PrefixReplicaPlanner().plan(plannerDemands, workers,
                new PrefixReplicaPlanner.Limits(settings.interThreshold(), settings.intraThreshold(),
                        settings.deleteThreshold(), settings.operationsPerList(), settings.destinationBudgetBytes(),
                        settings.wireBudgetBytes(), settings.safetyBytes()), settings.enabledActions());
        var structureMetrics = PrefixStructureMetrics.captureFromHolders(capturedMillis, parents, view, completeHolders);
        var copyMetrics = PrefixCopyMetrics.capture(capturedMillis, view);
        List<Candidate> bounded = new ArrayList<>();
        bounded.addAll(interCandidates.snapshot());
        bounded.addAll(intraCandidates.snapshot());
        bounded.addAll(reclaimCandidates.snapshot());
        long finalizeStartedNanos = System.nanoTime();
        synchronized (this) {
            long acquiredNanos = System.nanoTime();
            boolean completed = false;
            try {
                if (resets != capturedResets) { completed = true; return List.of(); }
                confirmedNodes.stream().filter(this.parents::containsKey).forEach(previouslyCached::add);
                this.demands.replaceAll((key, value) -> {
                    var observed = demands.getOrDefault(key, ZERO_DEMAND);
                    return new Demand(Math.max(0, value.access() - observed.access() * (1 - settings.decay())),
                            Math.max(0, value.missed() - observed.missed() * (1 - settings.decay())));
                });
                {
                    var removable = pruneCandidates(directory.view(), this.parents, this.demands,
                            lastSeenMillis, List.copyOf(pending.values()));
                    var removed = Set.copyOf(removable.stream().limit(1024).toList());
                    for (Node node : removed) {
                        this.parents.remove(node);
                        lastSeenMillis.remove(node);
                        identities.remove(node.hash());
                        previouslyCached.remove(node);
                    }
                    if (!removed.isEmpty()) {
                        this.demands.keySet().removeIf(key -> removed.contains(key.node()));
                    }
                }
                lastProposals = proposals;
                lastStructureMetrics = structureMetrics;
                lastCopyMetrics = copyMetrics;
                lastPlan = List.copyOf(bounded);
                lastScanMillis = clock.millis();
                lastScanNanos = System.nanoTime() - startedNanos;
                scans++;
                lastScan = new Scan(++scanSequence, capturedMillis, settings.namespace(), proposals, view, workloads);
                completed = true;
                return lastPlan;
            } finally {
                long finishedNanos = System.nanoTime();
                scanFinalizeWaitLatency.recordDuration(acquiredNanos - finalizeStartedNanos, false);
                scanFinalizeHoldLatency.recordDuration(finishedNanos - acquiredNanos, !completed);
            }
        }
    }

    public synchronized Map<DemandKey, Demand> demandSnapshot() { return Map.copyOf(demands); }

    public synchronized List<Long> reclamationOrder(Set<Long> candidates) {
        var depths = new HashMap<Long, Integer>();
        var ambiguous = new HashSet<Long>();
        for (var node : parents.keySet()) {
            if (!candidates.contains(node.hash())) { continue; }
            var previous = depths.putIfAbsent(node.hash(), node.depth());
            if (previous != null && previous != node.depth()) { ambiguous.add(node.hash()); }
        }
        // Removing descendants first forms suffix-closed victims. Unknown tree
        // nodes are excluded; native reclamation validates the complete tree.
        return depths.keySet().stream().filter(key -> !ambiguous.contains(key))
                .sorted(Comparator.<Long>comparingInt(depths::get).reversed().thenComparingLong(Long::longValue)).toList();
    }

    public record PreservationSnapshot(String namespace, long createdMillis,
                                       List<PrefixReplicaPlanner.Demand> demands, Map<String, Double> workloads) {
        public PreservationSnapshot {
            demands = List.copyOf(demands);
            workloads = Map.copyOf(workloads);
        }
    }

    public synchronized PreservationSnapshot preservationSnapshot() {
        drainObservations();
        var retained = new ArrayList<PrefixReplicaPlanner.Demand>();
        for (var entry : preservationDemands().entrySet()) {
            if (entry.getValue().access() <= 0 && entry.getValue().missed() <= 0) { continue; }
            var path = new ArrayList<Long>();
            for (var node = entry.getKey().node(); node != null; node = parents.get(node)) { path.add(node.hash()); }
            java.util.Collections.reverse(path);
            retained.add(new PrefixReplicaPlanner.Demand(path, entry.getKey().degree(), entry.getValue().access(), entry.getValue().missed()));
        }
        retained.sort(Comparator.comparing((PrefixReplicaPlanner.Demand demand) -> demand.path().toString())
                .thenComparingInt(PrefixReplicaPlanner.Demand::degree));
        return new PreservationSnapshot(settings.namespace(), clock.millis(), retained, queueSeconds);
    }

    public synchronized boolean preservationObservationsSettled() {
        synchronized (observationQueue) { return pending.isEmpty() && observationQueue.isEmpty(); }
    }

    public synchronized PreservationSnapshot preservationSnapshot(Set<String> affectedRoutes) {
        drainObservations();
        var preservationDemands = preservationDemands();
        var view = directory.preservationView(affectedRoutes);
        Map<Node, List<String>> holders = new HashMap<>();
        var routes = view.keySet().stream().sorted().toList();
        Set<Node> valuable = new HashSet<>();
        preservationDemands.forEach((key, value) -> {
            if (value.access() > 0 || value.missed() > 0) { valuable.add(key.node()); }
        });
        for (Node node : depthOrderedNodes(parents.keySet())) {
            Node parent = parents.get(node);
            var complete = filterHolders(parent == null ? routes : holders.get(parent), node.hash(), view);
            holders.put(node, complete);
            if (complete.isEmpty()) { valuable.remove(node); }
        }
        // The preservation planner needs maximal held paths, not a quadratic
        // expansion of every historical ancestor. A partial holder still keeps
        // its longest valuable prefix because completeness is checked first.
        Set<Node> covered = new HashSet<>();
        for (Node node : valuable) {
            for (Node parent = parents.get(node); parent != null && covered.add(parent); parent = parents.get(parent)) { }
        }
        valuable.removeAll(covered);
        Map<Node, PrefixPath> paths = new HashMap<>();
        var retained = new ArrayList<PrefixReplicaPlanner.Demand>();
        preservationDemands.entrySet().stream().filter(entry -> valuable.contains(entry.getKey().node())
                        && (entry.getValue().access() > 0 || entry.getValue().missed() > 0))
                .sorted(Comparator.comparingInt((Map.Entry<DemandKey, Demand> entry) -> entry.getKey().node().depth())
                        .thenComparingLong(entry -> entry.getKey().node().hash()).thenComparingInt(entry -> entry.getKey().degree()))
                .forEach(entry -> retained.add(new PrefixReplicaPlanner.Demand(plannerPath(entry.getKey().node(), parents, paths),
                        entry.getKey().degree(), entry.getValue().access(), entry.getValue().missed())));
        return new PreservationSnapshot(settings.namespace(), clock.millis(), retained, queueSeconds);
    }

    private Map<DemandKey, Demand> preservationDemands() {
        var retained = new HashMap<>(demands);
        // Drained requests can outlive their bounded reuse-event history. Protect
        // their held prefixes without inventing measured reuse or miss statistics.
        for (var execution : pending.values()) {
            for (var node : execution.observation().path()) {
                var key = new DemandKey(node, execution.degree());
                var demand = retained.getOrDefault(key, ZERO_DEMAND);
                retained.put(key, new Demand(Math.max(1.0, demand.access()), demand.missed()));
            }
        }
        return retained;
    }

    private void expirePending() {
        int beforeExpiry = pending.size();
        pending.entrySet().removeIf(entry -> clock.millis() - entry.getValue().observation().createdMillis()
                > settings.observationTtlMillis());
        expiredObservations += beforeExpiry - pending.size();
    }
    private List<Node> pruneCandidates(Map<String, ConfirmedCacheDirectory.Snapshot> view, Map<Node, Node> parents,
                                       Map<DemandKey, Demand> demands, Map<Node, Long> seen, List<Pending> observations) {
        long startedMillis = clock.millis();
        if (!mayHaveExpiredNodes(startedMillis, seen)) {
            return List.of();
        }
        List<Node> expired = parents.keySet().stream()
                .filter(node -> clock.millis() - seen.get(node) > settings.nodeIdleTtlMillis()).toList();
        if (expired.isEmpty()) { return List.of(); }
        boolean allExpired = expired.size() == parents.size();
        Set<Node> eligible = allExpired ? Set.of() : new HashSet<>(expired);
        Set<Node> protectedNodes = new HashSet<>();
        observations.forEach(execution -> protectedNodes.addAll(execution.observation().path()));
        demands.forEach((key, demand) -> {
            if (demand.access() > 1e-9 || demand.missed() > 1e-9) { protectedNodes.add(key.node()); }
        });
        Set<Long> present = new HashSet<>();
        view.values().forEach(snapshot -> present.addAll(snapshot.keys()));
        Map<Node, Integer> children = new HashMap<>();
        List<Node> removable = new ArrayList<>();
        parents.values().forEach(parent -> {
            if (parent != null && (allExpired || eligible.contains(parent))) { children.merge(parent, 1, Integer::sum); }
        });
        for (Node node : expired.stream().sorted(Comparator.comparingInt(Node::depth).reversed()).toList()) {
            if (children.getOrDefault(node, 0) != 0 || protectedNodes.contains(node) || present.contains(node.hash())
                    || clock.millis() - seen.get(node) <= settings.nodeIdleTtlMillis()) { continue; }
            Node parent = parents.get(node);
            if (parent != null) { children.computeIfPresent(parent, (ignored, count) -> count - 1); }
            removable.add(node);
        }
        return removable;
    }
    private boolean mayHaveExpiredNodes(long startedMillis, Map<Node, Long> seen) {
        if (seen != lastSeenMillis) {
            return seen.values().stream().anyMatch(lastSeen -> startedMillis - lastSeen > settings.nodeIdleTtlMillis());
        }
        if (oldestSeenKnown && startedMillis >= 0 && oldestSeenMillis >= 0
                && startedMillis - oldestSeenMillis <= settings.nodeIdleTtlMillis()) {
            return false;
        }
        long oldest = Long.MAX_VALUE;
        for (long lastSeen : seen.values()) {
            if (startedMillis - lastSeen > settings.nodeIdleTtlMillis()) { return true; }
            oldest = Math.min(oldest, lastSeen);
        }
        oldestSeenMillis = oldest;
        oldestSeenKnown = true;
        return false;
    }
    public synchronized void clear() {
        resets++;
        synchronized (observationQueue) {
            observationQueueGeneration++;
            observationQueue.clear();
            observationEnqueued = observationFallbacks = observationMaxQueueDepth = 0;
        }
        observationProcessed = observationQueueExpired = observationQueueCancelled = observationQueueFailures = 0;
        observationMaxQueueAgeMillis = 0;
        observationLastFailure = "";
        directory.clear();
        parents.clear();
        lastSeenMillis.clear();
        oldestSeenMillis = Long.MAX_VALUE;
        oldestSeenKnown = false;
        identities.clear();
        previouslyCached.clear();
        demands.clear();
        pending.clear();
        if (routingHistory != null) { routingHistory.clear(); }
        workloads.clear();
        queueSeconds.clear();
        eventWatermarks.clear();
        lastPlan = List.of();
        lastProposals = List.of();
        lastScan = new Scan(++scanSequence, clock.millis(), settings.namespace(), List.of(), Map.of(), Map.of());
        actionStatus = Map.of();
        lastStructureMetrics = null;
        lastCopyMetrics = null;
        overflow = unmatchedEvents = negativeMarginals = expiredObservations = acceptedObservations = committedAttempts = 0;
        lastScanNanos = scans = 0;
        hintedLookups = hintedBlocks = 0;
        lastScanMillis = clock.millis();
    }
    public synchronized Map<String, Object> status() {
        Map<String, Object> result = new LinkedHashMap<>();
        synchronized (observationQueue) {
            result.put("observation_queue_capacity", observationQueueCapacity);
            result.put("observation_queue_depth", observationQueue.size());
            result.put("observation_queue_max_depth", observationMaxQueueDepth);
            result.put("observation_queue_enqueued", observationEnqueued);
            result.put("observation_queue_fallbacks", observationFallbacks);
        }
        result.put("observation_queue_processed", observationProcessed);
        result.put("observation_queue_expired", observationQueueExpired);
        result.put("observation_queue_cancelled", observationQueueCancelled);
        result.put("observation_queue_failures", observationQueueFailures);
        result.put("observation_queue_max_age_ms", observationMaxQueueAgeMillis);
        result.put("observation_queue_last_failure", observationLastFailure);
        result.put("nodes", parents.size());
        result.put("routing_hints_enabled", routingHintsEnabled);
        result.put("routing_history_enabled", routingHistory != null);
        result.put("routing_history_capacity_per_route", routingHistory != null ? routingHistory.capacity() : 0);
        result.put("routing_hint_ttl_ms", settings.reportTtlMillis());
        result.put("routing_hinted_lookups", hintedLookups);
        result.put("routing_hinted_candidate_blocks", hintedBlocks);
        result.put("metadata_lookup_latency", directory.lookupLatency());
        result.put("directory_lookup_latency", directory.directoryLookupLatency());
        result.put("report_apply_latency", reportLatency.snapshot());
        result.put("observation_update_latency", observationLatency.snapshot());
        result.put("observation_accept_latency", acceptanceLatency.snapshot());
        result.put("scan_capture_wait_latency", scanCaptureWaitLatency.snapshot());
        result.put("scan_capture_hold_latency", scanCaptureHoldLatency.snapshot());
        result.put("scan_finalize_wait_latency", scanFinalizeWaitLatency.snapshot());
        result.put("scan_finalize_hold_latency", scanFinalizeHoldLatency.snapshot());
        if (lastStructureMetrics != null) { result.put("cache_structure", lastStructureMetrics); }
        if (lastCopyMetrics != null) { result.put("copy_lifetime", lastCopyMetrics); }
        result.put("pending", pending.size());
        result.put("overflow", overflow);
        result.put("unmatched_events", unmatchedEvents);
        result.put("negative_marginals", negativeMarginals);
        result.put("last_scan_ms", lastScanMillis);
        result.put("last_scan_duration_ns", lastScanNanos);
        result.put("scans", scans);
        result.put("candidates", lastPlan);
        result.put("proposals", lastProposals);
        result.put("expired_observations", expiredObservations);
        result.put("accepted_observations", acceptedObservations);
        result.put("committed_attempts", committedAttempts);
        result.put("physical_actions", actionStatus.getOrDefault("submitted", 0));
        result.put("actions", actionStatus);
        return Map.copyOf(result);
    }

    static Node[] depthOrderedNodes(Set<Node> nodes) {
        Node[] captured = nodes.toArray(Node[]::new);
        int maximumDepth = 0;
        for (Node node : captured) {
            if (node.depth() < 0 || node.depth() > captured.length) {
                java.util.Arrays.sort(captured, Comparator.comparingInt(Node::depth));
                return captured;
            }
            maximumDepth = Math.max(maximumDepth, node.depth());
        }
        int[] positions = new int[maximumDepth + 1];
        for (Node node : captured) { positions[node.depth()]++; }
        int offset = 0;
        for (int depth = 0; depth < positions.length; depth++) {
            int count = positions[depth];
            positions[depth] = offset;
            offset += count;
        }
        Node[] ordered = new Node[captured.length];
        for (Node node : captured) { ordered[positions[node.depth()]++] = node; }
        return ordered;
    }
}
