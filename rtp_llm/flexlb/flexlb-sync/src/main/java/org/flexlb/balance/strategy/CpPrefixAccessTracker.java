package org.flexlb.balance.strategy;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Recent access frequency of prefix blocks, the weight the reconciliation-traffic
 * estimate applies to a retained prefix.
 *
 * <p>Every routed request contributes one access to each of its prefix blocks.
 * At the end of an observation window the counts are decayed, so a block keeps
 * weight only while it is being reused; a block that stops being requested fades
 * out of the estimate instead of holding its historical weight forever. The map
 * is bounded and evicts the least recently accessed block.
 */
final class CpPrefixAccessTracker {

    private static final double MINIMUM_WEIGHT = 1e-3;

    private final double decayPerWindow;
    private final LinkedHashMap<Long, Double> accessCount;

    CpPrefixAccessTracker(int maxKeys, double decayPerWindow) {
        int capacity = Math.max(1, maxKeys);
        this.decayPerWindow = decayPerWindow > 0.0 && decayPerWindow < 1.0 ? decayPerWindow : 0.5;
        this.accessCount = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, Double> eldest) {
                return size() > capacity;
            }
        };
    }

    synchronized void record(Collection<Long> blockKeys) {
        if (blockKeys == null) {
            return;
        }
        for (Long key : blockKeys) {
            if (key != null) {
                accessCount.merge(key, 1.0, Double::sum);
            }
        }
    }

    synchronized double frequency(long blockKey) {
        return accessCount.getOrDefault(blockKey, 0.0);
    }

    /** Total recent access frequency of the given blocks. */
    synchronized double weight(Collection<Long> blockKeys) {
        if (blockKeys == null || blockKeys.isEmpty()) {
            return 0.0;
        }
        double total = 0.0;
        for (Long key : blockKeys) {
            if (key != null) {
                total += accessCount.getOrDefault(key, 0.0);
            }
        }
        return total;
    }

    /** Total frequency of the blocks whose recent frequency reaches a threshold. */
    synchronized double weightAbove(Collection<Long> blockKeys, double minimumFrequency) {
        if (blockKeys == null || blockKeys.isEmpty()) {
            return 0.0;
        }
        double total = 0.0;
        for (Long key : blockKeys) {
            if (key == null) {
                continue;
            }
            double frequency = accessCount.getOrDefault(key, 0.0);
            if (frequency >= minimumFrequency) {
                total += frequency;
            }
        }
        return total;
    }

    synchronized void decay() {
        accessCount.entrySet().removeIf(entry -> {
            double decayed = entry.getValue() * decayPerWindow;
            if (decayed < MINIMUM_WEIGHT) {
                return true;
            }
            entry.setValue(decayed);
            return false;
        });
    }

    synchronized int size() {
        return accessCount.size();
    }

    synchronized Set<Long> keysForTest() {
        return Set.copyOf(accessCount.keySet());
    }
}
