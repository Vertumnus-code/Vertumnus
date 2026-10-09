package org.flexlb.cache.prefix;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

public final class PrefixLatencyMetrics {
    public record Snapshot(long completedCalls, long exceptionalCalls, long nullResponses, long invalidClockSamples,
                           long totalNanos, long maxNanos, boolean saturated, List<Long> powerOfTwoBuckets,
                           Long p50UpperBoundNanos, Long p95UpperBoundNanos, Long p99UpperBoundNanos) { }

    private final LongSupplier nanoClock;
    private final long[] buckets = new long[64];
    private long completed;
    private long exceptional;
    private long nullResponses;
    private long invalidClockSamples;
    private long totalNanos;
    private long maxNanos;
    private boolean saturated;

    public PrefixLatencyMetrics() { this(System::nanoTime); }

    PrefixLatencyMetrics(LongSupplier nanoClock) { this.nanoClock = Objects.requireNonNull(nanoClock); }

    public <Value> Value measure(Supplier<Value> operation) {
        long started = nanoClock.getAsLong();
        boolean returned = false;
        boolean nullResponse = false;
        try {
            Value result = operation.get();
            returned = true;
            nullResponse = result == null;
            return result;
        } finally {
            record(nanoClock.getAsLong() - started, !returned, nullResponse);
        }
    }

    void recordDuration(long nanos, boolean failed) { record(nanos, failed, false); }

    private synchronized void record(long nanos, boolean failed, boolean nullResponse) {
        completed = add(completed, 1);
        if (failed) { exceptional = add(exceptional, 1); }
        if (nullResponse) { nullResponses = add(nullResponses, 1); }
        if (nanos < 0) { invalidClockSamples = add(invalidClockSamples, 1); return; }
        totalNanos = add(totalNanos, nanos);
        maxNanos = Math.max(maxNanos, nanos);
        int bucket = nanos == 0 ? 0 : Long.SIZE - Long.numberOfLeadingZeros(nanos);
        buckets[bucket] = add(buckets[bucket], 1);
    }

    private long add(long value, long increment) {
        if (Long.MAX_VALUE - value < increment) { saturated = true; return Long.MAX_VALUE; }
        return value + increment;
    }

    private Long percentile(int percent) {
        if (saturated) { return null; }
        long samples = completed - invalidClockSamples;
        if (samples == 0) { return null; }
        long rank = samples / 100 * percent + (samples % 100 * percent + 99) / 100;
        long count = 0;
        for (int bucket = 0; bucket < buckets.length; bucket++) {
            count += buckets[bucket];
            if (count >= rank) { return bucket == 63 ? Long.MAX_VALUE : (1L << bucket) - 1; }
        }
        return null;
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(completed, exceptional, nullResponses, invalidClockSamples, totalNanos, maxNanos, saturated,
                Arrays.stream(buckets).boxed().toList(), percentile(50), percentile(95), percentile(99));
    }
}
