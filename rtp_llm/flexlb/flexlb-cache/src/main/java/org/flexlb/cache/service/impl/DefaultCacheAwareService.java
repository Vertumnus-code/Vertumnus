package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Default implementation of cache-aware service
 * Provides unified cache management service, encapsulating underlying KvCacheManager
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {
    @Override
    public org.flexlb.cache.prefix.PrefixObserver prefixObserver() { return kvCacheManager.getPrefixObserver(); }
    
    @Autowired
    private KvCacheManager kvCacheManager;
    
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;
    
    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> resultMap
                = kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group);

            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "0");

            return resultMap;
        } catch (Exception e) {
            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "1");
            log.error("Error finding matching engines for role: {}", roleType, e);
            return Collections.emptyMap();
        }
    }
    
    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, List<String> candidateEngineIpPorts) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()
                    || candidateEngineIpPorts == null || candidateEngineIpPorts.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> resultMap
                = kvCacheManager.findMatchingEngines(blockCacheKeys, candidateEngineIpPorts);

            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "0");

            return resultMap;
        } catch (Exception e) {
            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "1");
            log.error("Error finding matching engines for explicit candidates, role: {}", roleType, e);
            return Collections.emptyMap();
        }
    }
    
    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        long startTime = System.nanoTime() / 1000;
        String engineIpPort = workerStatus.getIpPort();
        String role = workerStatus.getRole();

        try {
            if (workerStatus.getCacheStatus() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            String ipPort = workerStatus.getIpPort();
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus.getCachedKeys() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            
            // Update cache
            kvCacheManager.updateEngineCache(ipPort, role, cachedKeys);
            
            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(ipPort, role, startTime, "1");
            
            return result;
                
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);
            
            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, e.getMessage());

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
            
            return result;
        }
    }

    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(
            String engineIpPort, String role, CacheStatus cacheStatus) {
        long startTime = System.nanoTime() / 1000;
        try {
            if (engineIpPort == null || cacheStatus == null || cacheStatus.getCachedKeys() == null) {
                return buildFailureResult(engineIpPort, "DP-rank cache status is incomplete");
            }
            kvCacheManager.updateEngineCache(engineIpPort, role, cacheStatus.getCachedKeys());
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "1");
            return WorkerCacheUpdateResult.builder()
                    .success(true)
                    .engineIpPort(engineIpPort)
                    .cacheBlockCount(cacheStatus.getCachedKeys().size())
                    .availableKvCache(cacheStatus.getAvailableKvCache())
                    .totalKvCache(cacheStatus.getTotalKvCache())
                    .cacheVersion(cacheStatus.getVersion())
                    .build();
        } catch (Throwable e) {
            log.error("Error updating DP-rank cache for: {}", engineIpPort, e);
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
            return buildFailureResult(engineIpPort, e.getMessage());
        }
    }

    /**
     * Build success result
     */
    private WorkerCacheUpdateResult buildSuccessResult(WorkerStatus workerStatus, CacheStatus cacheStatus) {
        return WorkerCacheUpdateResult.builder()
            .success(true)
            .engineIpPort(workerStatus.getIpPort())
            .cacheBlockCount(cacheStatus.getCachedKeys() != null ? cacheStatus.getCachedKeys().size() : 0)
            .availableKvCache(cacheStatus.getAvailableKvCache())
            .totalKvCache(cacheStatus.getTotalKvCache())
            .cacheVersion(cacheStatus.getVersion())
            .build();
    }
    
    @Override
    public void addCacheBlocks(String engineIpPort, String role, List<Long> cacheKeys) {
        if (engineIpPort == null || cacheKeys == null || cacheKeys.isEmpty()) {
            return;
        }
        try {
            kvCacheManager.addCacheBlocks(engineIpPort, role, cacheKeys);
        } catch (Throwable e) {
            log.error("Error recording routed cache blocks for: {}", engineIpPort, e);
        }
    }

    /**
     * Build failure result
     */
    private WorkerCacheUpdateResult buildFailureResult(String engineIpPort, String errorMessage) {
        return WorkerCacheUpdateResult.builder()
            .success(false)
            .engineIpPort(engineIpPort)
            .errorMessage(errorMessage)
            .build();
    }
}
