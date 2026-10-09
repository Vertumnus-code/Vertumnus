package org.flexlb.dao.cache;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;

public record PrefixCopyLifetime(@JsonProperty("valid") boolean valid,
                                 @JsonProperty("sampled_ns") long sampledNanos,
                                 @JsonProperty("published_blocks") long publishedBlocks,
                                 @JsonProperty("reused_blocks") long reusedBlocks,
                                 @JsonProperty("wasted_blocks") long wastedBlocks,
                                 @JsonProperty("unknown_retired_blocks") long unknownRetiredBlocks,
                                 @JsonProperty("dropped_terminal_records") long droppedTerminalRecords,
                                 @JsonProperty("counters_saturated") boolean countersSaturated,
                                 @JsonProperty("records") List<Copy> records) {
    public PrefixCopyLifetime {
        records = records == null ? List.of() : List.copyOf(records);
    }

    public static final class UnsignedIdentityDeserializer extends JsonDeserializer<Long> {
        @Override
        public Long deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            String value = parser.getText();
            try {
                return value.startsWith("-") ? Long.parseLong(value) : Long.parseUnsignedLong(value);
            } catch (NumberFormatException invalid) {
                throw context.weirdStringException(value, Long.class, "expected a signed or unsigned 64-bit identity");
            }
        }
    }

    public record Copy(@JsonProperty("generation") @JsonDeserialize(using = UnsignedIdentityDeserializer.class) long generation,
                       @JsonProperty("operation_id") @JsonDeserialize(using = UnsignedIdentityDeserializer.class) long operationId,
                       @JsonProperty("import_epoch") @JsonDeserialize(using = UnsignedIdentityDeserializer.class) long importEpoch,
                       @JsonProperty("cache_key") @JsonDeserialize(using = UnsignedIdentityDeserializer.class) long cacheKey,
                       @JsonProperty("block_index") int blockIndex,
                       @JsonProperty("published_ns") long publishedNanos,
                       @JsonProperty("first_reuse_ns") long firstReuseNanos,
                       @JsonProperty("evicted_ns") long evictedNanos,
                       @JsonProperty("physical_free_ns") long physicalFreeNanos,
                       @JsonProperty("first_request_id") long firstRequestId,
                       @JsonProperty("first_attempt_id") String firstAttemptId,
                       @JsonProperty("reused") boolean reused,
                       @JsonProperty("evicted") boolean evicted,
                       @JsonProperty("physically_freed") boolean physicallyFreed,
                       @JsonProperty("retired") boolean retired,
                       @JsonProperty("tracking_valid") boolean trackingValid,
                       @JsonProperty("pending_executions") long pendingExecutions) {
        public String outcome() {
            if (reused) { return "REUSED"; }
            if (!trackingValid) { return "UNKNOWN"; }
            return retired ? "UNUSED_RETIRED" : "RIGHT_CENSORED";
        }

        public Long firstSuccessfulReuseNanos() {
            return reused && trackingValid && firstReuseNanos >= publishedNanos && publishedNanos >= 0
                    ? firstReuseNanos - publishedNanos : null;
        }

        private boolean consistent(long sampled) {
            if (generation == 0 || operationId == 0 || importEpoch == 0 || blockIndex <= 0
                    || pendingExecutions < 0 || retired != (physicallyFreed && pendingExecutions == 0)) { return false; }
            if ((!reused && firstReuseNanos != -1) || (!evicted && evictedNanos != -1)
                    || (!physicallyFreed && physicalFreeNanos != -1)) { return false; }
            if (!trackingValid) { return true; }
            return publishedNanos >= 0 && sampled >= publishedNanos
                    && (!reused || firstReuseNanos >= publishedNanos && firstReuseNanos <= sampled)
                    && (!evicted || evictedNanos >= publishedNanos && evictedNanos <= sampled)
                    && (!physicallyFreed || evicted && physicalFreeNanos >= evictedNanos && physicalFreeNanos <= sampled);
        }
    }

    public boolean consistent() {
        if (!valid || countersSaturated || sampledNanos < 0 || publishedBlocks < 0 || reusedBlocks < 0
                || wastedBlocks < 0 || unknownRetiredBlocks < 0 || droppedTerminalRecords < 0) { return false; }
        var generations = new HashSet<Long>();
        long liveUnused = 0;
        long live = 0;
        long terminal = 0;
        long observedReused = 0;
        long observedWasted = 0;
        long observedUnknown = 0;
        for (Copy copy : records) {
            if (!generations.add(copy.generation()) || !copy.consistent(sampledNanos)) { return false; }
            if (copy.retired()) { terminal++; } else { live++; }
            if (!copy.retired() && !copy.reused()) { liveUnused++; }
            if (copy.reused()) { observedReused++; }
            if (copy.retired() && !copy.reused()) {
                if (copy.trackingValid()) { observedWasted++; } else { observedUnknown++; }
            }
        }
        try {
            return Math.addExact(Math.addExact(Math.addExact(reusedBlocks, wastedBlocks), unknownRetiredBlocks),
                    liveUnused) == publishedBlocks
                    && Math.addExact(Math.addExact(terminal, droppedTerminalRecords), live) == publishedBlocks
                    && observedReused <= reusedBlocks && observedWasted <= wastedBlocks
                    && observedUnknown <= unknownRetiredBlocks;
        } catch (ArithmeticException overflow) { return false; }
    }
}
