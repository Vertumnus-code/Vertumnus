package org.flexlb.sync.runner;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private static final Duration WORKER_STATUS_HTTP_TIMEOUT = Duration.ofMillis(1000);
    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(WORKER_STATUS_HTTP_TIMEOUT)
            .build();

    private final int httpPort;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.httpPort = Integer.parseInt(split[1]);
        this.grpcPort = CommonUtils.toGrpcPort(httpPort);
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.site = site;
        this.roleType = roleType;
        this.group = group;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
    }

    @Override
    public void run() {
        try {
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            var topology = workerStatus.getPrefixTopology();
            WorkerStatusResponse response = launchGrpcStatusCheck(ip, grpcPort, latestFinishedTaskVersion);
            workerStatus.lock.lock();
            try {
                if (topology == workerStatus.getPrefixTopology()) { handleStatusResponse(response, startTime); }
            } finally { workerStatus.lock.unlock(); }
        } finally {
            workerStatus.getStatusCheckInProgress().set(false);
        }
    }

    private WorkerStatusResponse launchGrpcStatusCheck(String ip, int grpcPort, long latestFinishedTaskVersion) {
        try {
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs, roleType);
            WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(workerStatusPB);
            mergeHttpWorkerStatusMetadata(response);
            return response;
        } catch (Throwable throwable) {
            handleException(throwable);
            WorkerStatusResponse errorResponse = new WorkerStatusResponse();
            errorResponse.setMessage("Worker status gRPC call failed: " + throwable.getMessage());
            return errorResponse;
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            var topology = workerStatus.getPrefixTopology();
            if (topology != null && (newWorkerStatus == null || newWorkerStatus.getTpSize() != topology.degree()
                    || newWorkerStatus.getDpSize() != topology.lanes()
                    || (newWorkerStatus.getCpSize() > 0 && newWorkerStatus.getCpSize() != topology.degree()))) { return; }
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, ip, roleType);
                return;
            }

            if (newWorkerStatus.getMessage() != null) {
                workerStatus.setAlive(false);
                logger.error("query engine worker status via gRPC, msg={}", newWorkerStatus.getMessage());
                return;
            }

            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(modelName, ipPort, newWorkerStatus.getRole(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
            }

            workerStatus.setSite(site);
            workerStatus.setGroup(group);
            workerStatus.setRole(newWorkerStatus.getRole());

            long currentVersion = workerStatus.getStatusVersion().get();
            if (currentVersion >= responseVersion) {
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);
                // Update basic worker status even when version is not updated
                workerStatus.setAlive(newWorkerStatus.isAlive());
                workerStatus.setDpSize(newWorkerStatus.getDpSize());
                workerStatus.setTpSize(newWorkerStatus.getTpSize());
                updateWorkerProfile(workerStatus, newWorkerStatus);
                updateDpRankStatuses(workerStatus, newWorkerStatus);

                // Update status timestamp and record actual sync interval
                long nowUs = System.nanoTime() / 1000;
                long prevUpdateTime = workerStatus.getStatusLastUpdateTime().get();
                if (prevUpdateTime > 0) {
                    workerStatus.getStatusUpdateIntervalUs().set(nowUs - prevUpdateTime);
                }
                workerStatus.getStatusLastUpdateTime().set(nowUs);

                // Update task state
                Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
                Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
                Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
                workerStatus.setWaitingTaskList(waitingTaskInfo);
                workerStatus.setRunningTaskList(runningTaskInfo);
                workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);

                // Report success even when version is not updated
                engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

                logWorkerStatusUpdate(startTime, workerStatus);
                return;
            }

            // Update worker status from gRPC response
            workerStatus.setAvailableConcurrency(newWorkerStatus.getAvailableConcurrency());
            workerStatus.setStepLatencyMs(newWorkerStatus.getStepLatencyMs());
            workerStatus.setIterateCount(newWorkerStatus.getIterateCount());
            workerStatus.setDpSize(newWorkerStatus.getDpSize());
            workerStatus.setTpSize(newWorkerStatus.getTpSize());
            updateWorkerProfile(workerStatus, newWorkerStatus);
            updateDpRankStatuses(workerStatus, newWorkerStatus);
            workerStatus.setAlive(newWorkerStatus.isAlive());
            workerStatus.getStatusVersion().set(responseVersion != null ? responseVersion : -1L);
            workerStatus.getLatestFinishedTaskVersion().set(newWorkerStatus.getLatestFinishedVersion() != null ? newWorkerStatus.getLatestFinishedVersion() : -1L);

            Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
            workerStatus.setWaitingTaskList(waitingTaskInfo);
            workerStatus.setRunningTaskList(runningTaskInfo);

            // Update local task state (including checking lost, updating running, and cleaning completed)
            workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);

            // Correct running queue total wait time
            workerStatus.updateRunningQueueTime();

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

            // Update status timestamp and record actual sync interval
            long nowUs = System.nanoTime() / 1000;
            long prevUpdateTime = workerStatus.getStatusLastUpdateTime().get();
            if (prevUpdateTime > 0) {
                workerStatus.getStatusUpdateIntervalUs().set(nowUs - prevUpdateTime);
            }
            workerStatus.getStatusLastUpdateTime().set(nowUs);
            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip, roleType);
        }
    }

    private void mergeHttpWorkerStatusMetadata(WorkerStatusResponse response) {
        try {
            URI uri = URI.create("http://" + ip + ":" + httpPort + "/worker_status");
            HttpRequest request = HttpRequest.newBuilder(uri)
                    .timeout(WORKER_STATUS_HTTP_TIMEOUT)
                    .GET()
                    .build();
            HttpResponse<String> httpResponse = HTTP_CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
            if (httpResponse.statusCode() < 200 || httpResponse.statusCode() >= 300) {
                logger.debug("query worker HTTP status failed, ipPort: {}, status: {}", ipPort, httpResponse.statusCode());
                return;
            }
            WorkerStatusResponse httpStatus = JsonUtils.toObjectOrNull(httpResponse.body(), WorkerStatusResponse.class);
            if (httpStatus == null) {
                return;
            }
            mergeWorkerProfile(response, httpStatus);
            mergeDpRankResults(response, httpStatus);
        } catch (Throwable throwable) {
            logger.debug("merge worker HTTP status metadata failed, ipPort: {}, msg: {}", ipPort, throwable.getMessage());
        }
    }

    private void mergeDpRankResults(WorkerStatusResponse target, WorkerStatusResponse source) {
        List<WorkerStatusResponse> results = source.getResults();
        if (results == null || results.isEmpty()) {
            return;
        }
        target.setResults(results);
        if (target.getDpSize() <= 0) {
            target.setDpSize(results.size());
        }
        target.setAlive(results.stream().anyMatch(WorkerStatusResponse::isAlive));
        target.setAvailableConcurrency(results.stream()
                .mapToLong(WorkerStatusResponse::getAvailableConcurrency).sum());
        target.setWaitingTaskInfo(mergeTaskMaps(results, TaskMapKind.WAITING));
        target.setRunningTaskInfo(mergeTaskMaps(results, TaskMapKind.RUNNING));
        target.setFinishedTaskInfo(mergeTaskMaps(results, TaskMapKind.FINISHED));
        target.setStepLatencyMs(results.stream()
                .mapToDouble(WorkerStatusResponse::getStepLatencyMs).max().orElse(0.0));
        target.setIterateCount(results.stream()
                .mapToLong(WorkerStatusResponse::getIterateCount).sum());
        target.setStatusVersion(results.stream()
                .map(WorkerStatusResponse::getStatusVersion)
                .filter(value -> value != null)
                .mapToLong(Long::longValue).max().orElse(0L));
        target.setLatestFinishedVersion(results.stream()
                .map(WorkerStatusResponse::getLatestFinishedVersion)
                .filter(value -> value != null)
                .mapToLong(Long::longValue).max().orElse(0L));
    }

    private Map<String, TaskInfo> mergeTaskMaps(
            List<WorkerStatusResponse> results, TaskMapKind kind) {
        Map<String, TaskInfo> merged = new HashMap<>();
        for (WorkerStatusResponse result : results) {
            Map<String, TaskInfo> tasks = switch (kind) {
                case WAITING -> result.getWaitingTaskInfo();
                case RUNNING -> result.getRunningTaskInfo();
                case FINISHED -> result.getFinishedTaskInfo();
            };
            if (tasks != null) {
                merged.putAll(tasks);
            }
        }
        return merged;
    }

    private void updateDpRankStatuses(
            WorkerStatus target, WorkerStatusResponse response) {
        List<WorkerStatusResponse> results = response.getResults();
        if (results == null || results.isEmpty()) {
            return;
        }
        Map<Integer, DpRankStatus> statuses = new HashMap<>();
        for (int dpRank = 0; dpRank < results.size(); dpRank++) {
            WorkerStatusResponse source = results.get(dpRank);
            DpRankStatus status = new DpRankStatus();
            status.setDpRank(dpRank);
            status.setAddress(source.getAddress());
            status.setAlive(source.isAlive());
            status.setAvailableConcurrency(source.getAvailableConcurrency());
            status.setStepLatencyMs(source.getStepLatencyMs());
            status.setIterateCount(source.getIterateCount());
            status.setWaitingTaskInfo(source.getWaitingTaskInfo());
            status.setRunningTaskInfo(source.getRunningTaskInfo());
            status.setFinishedTaskInfo(source.getFinishedTaskInfo());
            statuses.put(dpRank, status);
        }
        target.replaceDpRankStatuses(statuses);
    }

    private enum TaskMapKind {
        WAITING,
        RUNNING,
        FINISHED
    }

    private void mergeWorkerProfile(WorkerStatusResponse target, WorkerStatusResponse source) {
        if (source.getProfileName() != null && !source.getProfileName().isBlank()) {
            target.setProfileName(source.getProfileName());
        }
        if (source.getDpSize() > 0) {
            target.setDpSize(source.getDpSize());
        }
        if (source.getTpSize() > 0) {
            target.setTpSize(source.getTpSize());
        }
        if (source.getCpSize() > 0) {
            target.setCpSize(source.getCpSize());
        }
        if (source.getWorldSize() > 0) {
            target.setWorldSize(source.getWorldSize());
        }
        if (source.getLocalWorldSize() > 0) {
            target.setLocalWorldSize(source.getLocalWorldSize());
        }
        if (source.getCpRotateMethod() != null && !source.getCpRotateMethod().isBlank()) {
            target.setCpRotateMethod(source.getCpRotateMethod());
        }
        if (source.getCpEpoch() >= 0) {
            target.setCpEpoch(source.getCpEpoch());
        }
        if (source.getCpPhase() != null && !source.getCpPhase().isBlank()) {
            target.setCpPhase(source.getCpPhase());
            target.setCpPrepareEpoch(source.getCpPrepareEpoch());
        }
        if (source.getCpTargetSize() > 0) {
            target.setCpTargetSize(source.getCpTargetSize());
        }
        if (source.getLogicalLaneCount() > 0) {
            target.setLogicalLaneCount(source.getLogicalLaneCount());
        }
        if (source.getCpRanksPerEndpoint() > 0) {
            target.setCpRanksPerEndpoint(source.getCpRanksPerEndpoint());
        }
        // Non-null, not ">0": false is a meaningful value here and must survive the
        // merge, so the guard is presence, not truth.
        if (source.getCpAcceptsExternal() != null) {
            target.setCpAcceptsExternal(source.getCpAcceptsExternal());
        }
        if (source.getRouteFanout() > 0) {
            target.setRouteFanout(source.getRouteFanout());
        }
    }

    private void updateWorkerProfile(WorkerStatus workerStatus, WorkerStatusResponse newWorkerStatus) {
        boolean hasProfileName = newWorkerStatus.getProfileName() != null && !newWorkerStatus.getProfileName().isBlank();
        boolean hasCpSize = newWorkerStatus.getCpSize() > 0;
        long inferredCpSize = shouldInferCpProfile(newWorkerStatus) ? newWorkerStatus.getTpSize() : -1L;

        if (hasCpSize && newWorkerStatus.getCpEpoch() >= 0) {
            // A resident pool changes CP geometry in place and keeps the profile name it
            // was launched with, so the name has to be rebuilt from the reported degree:
            // otherwise the Global Manager publishes CP2 for a process that has already
            // committed CP4/CP8.
            workerStatus.setProfileName(buildCpProfileName(
                    newWorkerStatus.getCpSize(), newWorkerStatus.getTpSize()));
        } else if (hasProfileName) {
            workerStatus.setProfileName(newWorkerStatus.getProfileName());
        } else if (inferredCpSize > 0) {
            workerStatus.setProfileName(buildCpProfileName(inferredCpSize, newWorkerStatus.getTpSize()));
        }
        if (hasCpSize) {
            workerStatus.setCpSize(newWorkerStatus.getCpSize());
        } else if (inferredCpSize > 0) {
            workerStatus.setCpSize(inferredCpSize);
        }
        if (newWorkerStatus.getWorldSize() > 0) {
            workerStatus.setWorldSize(newWorkerStatus.getWorldSize());
        }
        if (newWorkerStatus.getLocalWorldSize() > 0) {
            workerStatus.setLocalWorldSize(newWorkerStatus.getLocalWorldSize());
        }
        if (newWorkerStatus.getCpRotateMethod() != null && !newWorkerStatus.getCpRotateMethod().isBlank()) {
            workerStatus.setCpRotateMethod(newWorkerStatus.getCpRotateMethod());
        }
        if (newWorkerStatus.getCpEpoch() >= 0) {
            workerStatus.setCpEpoch(newWorkerStatus.getCpEpoch());
        }
        if (newWorkerStatus.getCpPhase() != null && !newWorkerStatus.getCpPhase().isBlank()) {
            workerStatus.setCpPhase(newWorkerStatus.getCpPhase());
            workerStatus.setCpPrepareEpoch(newWorkerStatus.getCpPrepareEpoch());
        }
        if (newWorkerStatus.getCpTargetSize() > 0) {
            workerStatus.setCpTargetSize(newWorkerStatus.getCpTargetSize());
        }
        if (newWorkerStatus.getLogicalLaneCount() > 0) {
            workerStatus.setLogicalLaneCount(newWorkerStatus.getLogicalLaneCount());
        }
        if (newWorkerStatus.getCpRanksPerEndpoint() > 0) {
            workerStatus.setCpRanksPerEndpoint(newWorkerStatus.getCpRanksPerEndpoint());
        }
        // Presence-guarded like the rest of the CP block, so the last published claim
        // survives a poll that carried no metadata. That matters across a switch: the
        // lane that stops being the ingress flips true -> false on its next successful
        // poll, and until then admission is paused anyway (isCpEpochVisible requires
        // every front to report the new degree before finishCpSwitch reopens it).
        if (newWorkerStatus.getCpAcceptsExternal() != null) {
            workerStatus.setCpAcceptsExternal(newWorkerStatus.getCpAcceptsExternal());
        }
        if (newWorkerStatus.getRouteFanout() > 0) {
            workerStatus.setRouteFanout(newWorkerStatus.getRouteFanout());
        }
    }

    private boolean shouldInferCpProfile(WorkerStatusResponse newWorkerStatus) {
        if (newWorkerStatus == null || newWorkerStatus.getTpSize() <= 0 || newWorkerStatus.getCpSize() > 0) {
            return false;
        }
        boolean hasProfileName = newWorkerStatus.getProfileName() != null && !newWorkerStatus.getProfileName().isBlank();
        if (hasProfileName) {
            return false;
        }
        String model = modelName == null ? "" : modelName.toLowerCase();
        String workerGroup = group == null ? "" : group.toLowerCase();
        return "native".equalsIgnoreCase(System.getenv("FLEXLB_CP_SWITCH_MODE"))
                || model.contains("_cp") || model.contains("cp_")
                || workerGroup.contains("_cp") || workerGroup.contains("cp_");
    }

    private String buildCpProfileName(long cpSize, long tpSize) {
        if (tpSize > 0) {
            return "cp" + cpSize + "-tp" + tpSize;
        }
        return "cp" + cpSize;
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, ip, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, ip, roleType);
        }
    }

    private void log(String msg) {
        logger.info("[gRPC][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.nanoTime() / 1000 - createTimeUs,
                msg);
    }
}
