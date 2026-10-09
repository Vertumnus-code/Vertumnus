package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * Cache-aware service interface
 * Provides unified cache management interface for external modules
 *
 * @author FlexLB
 */
public interface CacheAwareService {
    default org.flexlb.cache.prefix.PrefixObserver prefixObserver() { return null; }
    
    /**
     * Find matching engines
     *
     * @param blockCacheKeys List of cache block IDs to query
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys, RoleType roleType, String group);

    /**
     * Find matching engines from an explicit candidate set. This is used by
     * cross-profile schedulers whose candidate pool should not be limited by
     * the globally active profile.
     *
     * @param blockCacheKeys        List of cache block IDs to query
     * @param roleType              Engine role to query
     * @param candidateEngineIpPorts Candidate engine IP:Port list
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys, RoleType roleType, List<String> candidateEngineIpPorts);
    
    /**
     * Update engine block KV cache status
     *
     * @param workerStatus Worker status information
     * @return Update result
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);

    /** Update cache metadata for one logical DP replica. */
    WorkerCacheUpdateResult updateEngineBlockCache(
            String engineIpPort, String role, org.flexlb.dao.master.CacheStatus cacheStatus);

    /**
     * Record that FlexLB routed a request with {@code cacheKeys} to {@code engineIpPort}.
     * This makes the routing decision authoritative for the fleet cache index, so the
     * next request sharing the same prefix hits without waiting for the engine to admit
     * it holds the blocks.
     *
     * @param engineIpPort Engine IP:Port
     * @param role         Engine role
     * @param cacheKeys    Ordered block cache keys of the routed request
     */
    void addCacheBlocks(String engineIpPort, String role, List<Long> cacheKeys);
}
