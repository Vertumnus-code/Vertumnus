package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converter between gRPC protobuf messages and Java objects for engine status
 */
public class EngineStatusConverter {

    /**
     * Convert WorkerStatusPB to WorkerStatusResponse
     */
    public static WorkerStatusResponse convertToWorkerStatusResponse(EngineRpcService.WorkerStatusPB workerStatusPB) {
        WorkerStatusResponse response = new WorkerStatusResponse();

        // Set role directly as string
        response.setRole(workerStatusPB.getRole());
        response.setAvailableConcurrency(workerStatusPB.getAvailableConcurrency());
        response.setRunningQueryLen(workerStatusPB.getRunningQueryLen());
        response.setWaitingQueryLen(workerStatusPB.getWaitingQueryLen());
        response.setStepLatencyMs(workerStatusPB.getStepLatencyMs());
        response.setIterateCount(workerStatusPB.getIterateCount());
        response.setDpSize(workerStatusPB.getDpSize());
        response.setTpSize(workerStatusPB.getTpSize());
        response.setStatusVersion(workerStatusPB.getStatusVersion());
        response.setLatestFinishedVersion(workerStatusPB.getLatestFinishedVersion());
        response.setAlive(workerStatusPB.getAlive());

        List<EngineRpcService.TaskInfoPB> srcRunningTaskInfoList = workerStatusPB.getRunningTaskInfoList();
        List<EngineRpcService.TaskInfoPB> waitingTaskInfoList = srcRunningTaskInfoList.stream()
                .filter(taskInfoPB -> !isRunning(taskInfoPB))
                .toList();
        List<EngineRpcService.TaskInfoPB> runningTaskInfoList = srcRunningTaskInfoList.stream()
                .filter(EngineStatusConverter::isRunning)
                .toList();

        // Convert waiting task info
        response.setWaitingTaskInfo(convertToTaskInfoList(waitingTaskInfoList));

        // Convert running task info
        response.setRunningTaskInfo(convertToTaskInfoList(runningTaskInfoList));

        // Convert finished task list
        response.setFinishedTaskInfo(convertToTaskInfoList(workerStatusPB.getFinishedTaskListList()));

        return response;
    }

    private static boolean isRunning(EngineRpcService.TaskInfoPB task) {
        return task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RUNNING
                || (task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_PENDING && !task.getIsWaiting());
    }

    /**
     * Convert CacheStatusPB to CacheStatus
     */
    public static CacheStatus convertToCacheStatus(EngineRpcService.CacheStatusPB cacheStatusPB) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(cacheStatusPB.getAvailableKvCache());
        cacheStatus.setTotalKvCache(cacheStatusPB.getTotalKvCache());
        cacheStatus.setBlockSize(cacheStatusPB.getBlockSize());
        cacheStatus.setVersion(cacheStatusPB.getVersion());
        if (cacheStatusPB.hasInventory()) {
            var report = cacheStatusPB.getInventory();
            cacheStatus.setInventory(new org.flexlb.dao.cache.CacheReportV2(
                    report.getNamespace(), report.getWorkerId(), report.getDpLane(), report.getEpoch(),
                    report.getLayoutFingerprint(), report.getSnapshotKind(), report.getVersion(),
                    report.getBaseVersion(), report.getRankReadinessGeneration(), report.getDeviceReady(),
                    report.getCpDegree(), report.getBlockSize(), report.getPhysicalFreeBytes(),
                    report.getDeviceKeysList(), report.getReuseEventsList().stream().map(event ->
                            new org.flexlb.dao.cache.CacheReportV2.PrefixReuseEvent(event.getSequence(),
                                    event.getRequestId(), event.getDeviceReuseTokens(), event.getInputTokens(),
                                    event.getPrefill(), event.getAttemptId())).toList(), report.getDroppedReuseEvents(),
                    report.getPhysicalBlockBytes(), report.getEvictableBytes(), report.getActiveBytes(),
                    report.getResidentBytes(), report.getOperationReservedBytes(), report.getPhysicalMemoryStatsValid(),
                    report.getPhysicalCapacityBytes(), report.hasCopyLifetime() ? convertCopyLifetime(report.getCopyLifetime()) : null));
        }

        Map<Long, Boolean> cacheKeysMap = cacheStatusPB.getCacheKeysMap();
        Set<Long> cachedKeysSet = cacheKeysMap.keySet();
        cacheStatus.setCachedKeys(cachedKeysSet);
        cacheStatus.setCacheKeySize(cacheKeysMap.size());
        return cacheStatus;
    }

    public static org.flexlb.dao.cache.PrefixCopyLifetime convertCopyLifetime(EngineRpcService.PrefixCopySnapshotPB report) {
        return new org.flexlb.dao.cache.PrefixCopyLifetime(report.getValid(), report.getSampledNs(),
                report.getPublishedBlocks(), report.getReusedBlocks(), report.getWastedBlocks(),
                report.getUnknownRetiredBlocks(), report.getDroppedTerminalRecords(), report.getCountersSaturated(),
                report.getRecordsList().stream().map(copy -> new org.flexlb.dao.cache.PrefixCopyLifetime.Copy(
                        copy.getGeneration(), copy.getOperationId(), copy.getImportEpoch(), copy.getCacheKey(),
                        copy.getBlockIndex(), copy.getPublishedNs(), copy.getFirstReuseNs(), copy.getEvictedNs(),
                        copy.getPhysicalFreeNs(), copy.getFirstRequestId(), copy.getFirstAttemptId(), copy.getReused(),
                        copy.getEvicted(), copy.getPhysicallyFreed(), copy.getRetired(), copy.getTrackingValid(),
                        copy.getPendingExecutions())).toList());
    }

    /**
     * Convert list of TaskInfoPB to list of TaskInfo
     */
    private static Map<String, TaskInfo> convertToTaskInfoList(List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null) {
            return null;
        }
        Map<String, TaskInfo> taskInfoMap = new HashMap<>(taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB taskInfoPB : taskInfoPBList) {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setRequestId(taskInfoPB.getRequestId());
            taskInfo.setPrefixLength(taskInfoPB.getPrefixLength());
            taskInfo.setInputLength(taskInfoPB.getInputLength());
            taskInfo.setWaitingTime(taskInfoPB.getWaitingTimeMs());
            taskInfo.setIterateCount(taskInfoPB.getIterateCount());
            taskInfo.setEndTimeMs(taskInfoPB.getEndTimeMs());
            taskInfo.setDpRank(taskInfoPB.getDpRank());
            taskInfo.setPhase(taskInfoPB.getPhase().name());
            taskInfo.setBatchId(taskInfoPB.getBatchId());
            taskInfo.setExecutionTimeMs(taskInfoPB.getExecutionTimeMs());

            taskInfoMap.put(String.valueOf(taskInfoPB.getRequestId()), taskInfo);
        }

        return taskInfoMap;
    }
}
