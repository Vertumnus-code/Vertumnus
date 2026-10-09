package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Per-engine cache-block view (small hash table).
 *
 * <p>Storage is a bounded LRU per engine. Adding an already-present key refreshes its
 * recency; adding a new key past the capacity evicts the least-recently-touched key.
 * Capacity is controlled by {@code FLEXLB_ENGINE_LOCAL_CAPACITY} (default 65536).
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class EngineLocalView {

    private static final String ENV_CAPACITY = "FLEXLB_ENGINE_LOCAL_CAPACITY";
    private static final int DEFAULT_CAPACITY = 65536;

    private final int perEngineCapacity;

    private final ConcurrentMap<String, BoundedKeySet> engineViews = new ConcurrentHashMap<>();

    public EngineLocalView() {
        this.perEngineCapacity = readIntEnv(ENV_CAPACITY, DEFAULT_CAPACITY);
        log.info("EngineLocalView initialized perEngineCapacity={}", perEngineCapacity);
    }

    /**
     * Add a cache block for the engine. Adding an already-present key refreshes recency.
     * Returns evicted key if the LRU had to make room (or null otherwise).
     */
    public Long addOrUpdateCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            log.warn("Invalid parameters: engineIPort={}, blockCacheKey={}", engineIPort, blockCacheKey);
            return null;
        }
        BoundedKeySet keys = engineViews.computeIfAbsent(engineIPort, k -> new BoundedKeySet(perEngineCapacity));
        return keys.addAndReturnEvicted(blockCacheKey);
    }

    /**
     * Batch-add cache blocks for the engine. Returns a list parallel to {@code newBlocks}
     * of evicted keys (nulls for entries where no eviction occurred). Callers use this to
     * mirror evictions into {@link GlobalCacheIndex}.
     */
    public List<Long> addOrUpdateCacheBlocks(String engineIPort, List<Long> newBlocks) {
        if (engineIPort == null || newBlocks == null || newBlocks.isEmpty()) {
            return Collections.emptyList();
        }
        BoundedKeySet keys = engineViews.computeIfAbsent(engineIPort, k -> new BoundedKeySet(perEngineCapacity));
        List<Long> evicted = new ArrayList<>(newBlocks.size());
        for (Long k : newBlocks) {
            if (k == null) {
                evicted.add(null);
                continue;
            }
            evicted.add(keys.addAndReturnEvicted(k));
        }
        return evicted;
    }

    public void removeCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            return;
        }
        BoundedKeySet keys = engineViews.get(engineIPort);
        if (keys == null) {
            return;
        }
        if (keys.remove(blockCacheKey) && keys.isEmpty()) {
            engineViews.remove(engineIPort);
        }
    }

    public void removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return;
        }
        BoundedKeySet removed = engineViews.remove(engineIPort);
        if (removed == null) {
            log.warn("Remove failed, the engine: {} not exist.", engineIPort);
        }
    }

    public Set<Long> getEngineCacheBlocks(String engineIPort) {
        if (engineIPort == null) {
            return Collections.emptySet();
        }
        BoundedKeySet keys = engineViews.get(engineIPort);
        return keys == null ? Collections.emptySet() : keys.snapshot();
    }

    public void clear() {
        engineViews.clear();
        log.info("Cleared engine local view");
    }

    public int size(String engineIpPort) {
        BoundedKeySet keys = engineViews.get(engineIpPort);
        return keys == null ? 0 : keys.size();
    }

    public int getEngineViewsMapSize() {
        return engineViews.size();
    }

    public Set<String> getAllEngineIpPorts() {
        return engineViews.keySet();
    }

    private static int readIntEnv(String name, int defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Math.max(1, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            log.warn("Invalid {}={}, falling back to {}", name, raw, defaultValue);
            return defaultValue;
        }
    }

    private static final class BoundedKeySet {
        private final int capacity;
        private final LinkedHashMap<Long, Boolean> keys;
        private Long lastEvicted;

        BoundedKeySet(int capacity) {
            this.capacity = capacity;
            this.keys = new LinkedHashMap<>(Math.min(capacity, 1024), 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, Boolean> eldest) {
                    if (size() > BoundedKeySet.this.capacity) {
                        BoundedKeySet.this.lastEvicted = eldest.getKey();
                        return true;
                    }
                    return false;
                }
            };
        }

        synchronized Long addAndReturnEvicted(Long key) {
            lastEvicted = null;
            keys.put(key, Boolean.TRUE);
            Long evicted = lastEvicted;
            lastEvicted = null;
            return evicted;
        }

        synchronized boolean remove(Long key) {
            return keys.remove(key) != null;
        }

        synchronized boolean isEmpty() {
            return keys.isEmpty();
        }

        synchronized int size() {
            return keys.size();
        }

        synchronized Set<Long> snapshot() {
            return new java.util.HashSet<>(keys.keySet());
        }
    }
}
