package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Clock;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class PrefixTransferRecoveryLoop implements AutoCloseable {
    public record Limits(long intervalMillis, long maxBackoffMillis, int maxActionsPerTick, long retentionMillis) {
        public Limits {
            if (intervalMillis <= 0 || maxBackoffMillis < intervalMillis || maxBackoffMillis > 3600000
                    || maxActionsPerTick < 1 || maxActionsPerTick > 4096 || retentionMillis < 0
                    || retentionMillis > 86400000) {
                throw new IllegalArgumentException("invalid prefix recovery limits");
            }
        }
    }

    public record Status(long ticks, long recovered, long reportedReady, long noLongerRetained,
                         long terminalRetired, long failures, int trackedEntries) { }

    private record Retry(long nextMillis, long delayMillis) { }

    private static final Logger LOG = LoggerFactory.getLogger(PrefixTransferRecoveryLoop.class);
    private final PrefixTransferJournal journal;
    private final PrefixTransferCoordinator coordinator;
    private final PrefixControllerAuthority authority;
    private final ConfirmedCacheDirectory directory;
    private final String namespace;
    private final Clock clock;
    private final Limits limits;
    private final Map<String, Retry> retries = new HashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean();
    private ScheduledExecutorService scheduler;
    private int cursor;
    private long ticks;
    private long recovered;
    private long reportedReady;
    private long noLongerRetained;
    private long terminalRetired;
    private long failures;

    public PrefixTransferRecoveryLoop(PrefixTransferJournal journal, PrefixTransferCoordinator coordinator,
                                      PrefixControllerAuthority authority, ConfirmedCacheDirectory directory,
                                      String namespace, Clock clock, Limits limits) {
        this.journal = Objects.requireNonNull(journal);
        this.coordinator = Objects.requireNonNull(coordinator);
        this.authority = Objects.requireNonNull(authority);
        this.directory = Objects.requireNonNull(directory);
        this.namespace = Objects.requireNonNull(namespace);
        this.clock = Objects.requireNonNull(clock);
        this.limits = Objects.requireNonNull(limits);
        if (namespace.isBlank()) { throw new IllegalArgumentException("missing recovery namespace"); }
    }

    public synchronized void start() {
        if (closed.get()) { throw new IllegalStateException("prefix recovery loop is closed"); }
        if (scheduler != null) { return; }
        scheduler = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread thread = new Thread(task, "prefix-transfer-recovery");
            thread.setDaemon(true);
            return thread;
        });
        scheduler.scheduleWithFixedDelay(() -> {
            try {
                tick();
            } catch (RuntimeException failure) {
                LOG.warn("Prefix recovery tick failed; unresolved durable records remain retained", failure);
            }
        }, limits.intervalMillis(), limits.intervalMillis(), TimeUnit.MILLISECONDS);
    }

    public synchronized void tick() {
        if (closed.get()) { return; }
        ticks++;
        org.flexlb.engine.grpc.EngineRpcService.PrefixControllerPB controller;
        try {
            controller = authority.snapshot(namespace);
        } catch (IllegalStateException standby) {
            return;
        }
        if (!authority.isCurrent(controller)) { return; }
        var entries = journal.entries();
        var activeIds = new HashSet<String>();
        entries.forEach(entry -> activeIds.add(entry.intent().requestId()));
        retries.keySet().retainAll(activeIds);
        if (entries.isEmpty()) { cursor = 0; return; }
        int actions = 0;
        int visited = 0;
        int start = Math.floorMod(cursor, entries.size());
        while (visited < entries.size() && actions < limits.maxActionsPerTick()) {
            int index = (start + visited++) % entries.size();
            cursor = (index + 1) % entries.size();
            if (closed.get() || !authority.isCurrent(controller)) { return; }
            var entry = entries.get(index);
            String id = entry.intent().requestId();
            Retry retry = retries.get(id);
            long now = clock.millis();
            if (now < entry.intent().deadlineUnixMs() || (retry != null && now < retry.nextMillis())) { continue; }
            actions++;
            boolean completed = false;
            try (var permit = directory.acquireRecovery()) {
                if (!entry.intent().namespace().equals(namespace)) {
                    throw new IllegalStateException("recovery journal namespace mismatch");
                }
                if (entry.result() == null || entry.result().cleanupPending()
                        || entry.result().outcome() == PrefixTransferCoordinator.Outcome.UNKNOWN) {
                    var result = coordinator.recover(entry);
                    if (!result.cleanupPending() && result.outcome() != PrefixTransferCoordinator.Outcome.UNKNOWN) {
                        recovered++;
                        completed = true;
                    }
                } else {
                    var retirement = journal.retire(entry, directory, limits.retentionMillis());
                    switch (retirement) {
                        case REPORTED_READY -> reportedReady++;
                        case NO_LONGER_RETAINED -> noLongerRetained++;
                        case TERMINAL -> terminalRetired++;
                        case WAITING -> { }
                    }
                    completed = retirement != PrefixTransferJournal.Retirement.WAITING;
                }
            } catch (RuntimeException failure) {
                failures++;
                LOG.warn("Prefix transfer recovery remains unconfirmed; retaining its durable record", failure);
            }
            if (completed) {
                retries.remove(id);
            } else {
                long delay = retry == null ? limits.intervalMillis()
                        : Math.min(limits.maxBackoffMillis(), retry.delayMillis() * 2);
                retries.put(id, new Retry(Math.addExact(now, delay), delay));
            }
        }
    }

    public synchronized Status status() {
        return new Status(ticks, recovered, reportedReady, noLongerRetained, terminalRetired, failures, retries.size());
    }

    @Override
    public void close() {
        closed.set(true);
        ScheduledExecutorService executor;
        synchronized (this) {
            executor = scheduler;
        }
        if (executor != null) { executor.shutdownNow(); }
    }
}
