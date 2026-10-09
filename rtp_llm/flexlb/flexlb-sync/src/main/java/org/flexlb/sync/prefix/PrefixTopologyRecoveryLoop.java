package org.flexlb.sync.prefix;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Clock;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class PrefixTopologyRecoveryLoop implements AutoCloseable {
    public record Limits(long intervalMillis, long maxBackoffMillis, int actionsPerTick) {
        public Limits {
            if (intervalMillis < 1 || maxBackoffMillis < intervalMillis || maxBackoffMillis > 3600000
                    || actionsPerTick < 1 || actionsPerTick > 4096) {
                throw new IllegalArgumentException("invalid topology recovery limits");
            }
        }
    }
    public record Status(long ticks, long confirmed, long cancelled, long unresolved, long failures, int pending) { }
    private record Retry(long nextMillis, long delayMillis) { }
    private static final Logger LOG = LoggerFactory.getLogger(PrefixTopologyRecoveryLoop.class);
    private final ZookeeperPrefixTopologyJournal journal;
    private final PrefixTopologyCoordinator coordinator;
    private final Clock clock;
    private final Limits limits;
    private final PrefixTopologyPlacement placement;
    private final Map<String, Retry> retries = new HashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final Object lifecycleLock = new Object();
    private volatile Status status = new Status(0, 0, 0, 0, 0, 0);
    private ScheduledExecutorService executor;
    private int cursor;

    public PrefixTopologyRecoveryLoop(ZookeeperPrefixTopologyJournal journal, PrefixTopologyCoordinator coordinator,
                                      Clock clock, Limits limits) {
        this(journal, coordinator, clock, limits, null);
    }

    public PrefixTopologyRecoveryLoop(ZookeeperPrefixTopologyJournal journal, PrefixTopologyCoordinator coordinator,
                                      Clock clock, Limits limits, PrefixTopologyPlacement placement) {
        this.journal = Objects.requireNonNull(journal);
        this.coordinator = Objects.requireNonNull(coordinator);
        this.clock = Objects.requireNonNull(clock);
        this.limits = Objects.requireNonNull(limits);
        this.placement = placement;
    }

    public void start() {
        synchronized (lifecycleLock) {
            if (closed.get()) { throw new IllegalStateException("topology recovery is closed"); }
            if (executor != null) { return; }
            executor = Executors.newSingleThreadScheduledExecutor(task -> {
                var thread = new Thread(task, "prefix-topology-recovery");
                thread.setDaemon(true);
                return thread;
            });
            executor.scheduleWithFixedDelay(this::tick, limits.intervalMillis(), limits.intervalMillis(), TimeUnit.MILLISECONDS);
        }
    }

    public synchronized void tick() {
        if (closed.get()) { return; }
        long confirmed = status.confirmed();
        long cancelled = status.cancelled();
        long unresolved = status.unresolved();
        long failures = status.failures();
        int pending = status.pending();
        try {
            if (placement != null) { placement.refresh(); }
            var entries = journal.entries().stream().filter(entry -> entry.state() == ZookeeperPrefixTopologyJournal.State.PLANNED
                    || entry.state() == ZookeeperPrefixTopologyJournal.State.SUBMITTED).toList();
            pending = entries.size();
            var ids = entries.stream().map(entry -> entry.plan().switchId()).collect(java.util.stream.Collectors.toSet());
            retries.keySet().retainAll(ids);
            int actions = 0;
            int visited = 0;
            int start = entries.isEmpty() ? 0 : Math.floorMod(cursor, entries.size());
            while (visited < entries.size() && actions < limits.actionsPerTick() && !closed.get()) {
                var entry = entries.get((start + visited++) % entries.size());
                var id = entry.plan().switchId();
                var retry = retries.get(id);
                long now = clock.millis();
                if ((retry != null && retry.nextMillis() > now)
                        || (entry.state() == ZookeeperPrefixTopologyJournal.State.PLANNED && entry.plan().deadlineUnixMs() > now)) { continue; }
                actions++;
                boolean resolved = false;
                try {
                    var result = coordinator.recover(id);
                    if (result.state() == ZookeeperPrefixTopologyJournal.State.CONFIRMED) { confirmed++; resolved = true; }
                    else if (result.state() == ZookeeperPrefixTopologyJournal.State.CANCELLED) { cancelled++; resolved = true; }
                    else { unresolved++; }
                } catch (RuntimeException failure) {
                    failures++;
                    LOG.warn("Topology recovery remains unconfirmed for {}", id, failure);
                }
                if (resolved) { retries.remove(id); pending--; }
                else {
                    long delay = retry == null ? limits.intervalMillis() : Math.min(limits.maxBackoffMillis(), retry.delayMillis() * 2);
                    retries.put(id, new Retry(Math.addExact(now, delay), delay));
                }
            }
            cursor = entries.isEmpty() ? 0 : (start + visited) % entries.size();
        } catch (RuntimeException failure) {
            failures++;
            LOG.warn("Cannot scan durable topology recovery journal", failure);
        }
        status = new Status(status.ticks() + 1, confirmed, cancelled, unresolved, failures, pending);
    }

    public Status status() { return status; }

    @Override
    public void close() {
        closed.set(true);
        ScheduledExecutorService current;
        synchronized (lifecycleLock) { current = executor; }
        if (current != null) {
            current.shutdownNow();
            try {
                if (!current.awaitTermination(5, TimeUnit.SECONDS)) { throw new IllegalStateException("topology recovery did not stop"); }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted stopping topology recovery", interrupted);
            }
        }
    }
}
