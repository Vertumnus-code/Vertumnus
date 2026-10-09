package org.flexlb.domain.worker;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import lombok.Data;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class WorkerStatusResponse {

    @JsonProperty("role")
    private String role;

    @JsonProperty("available_concurrency")
    private long availableConcurrency;

    @JsonProperty("running_query_len")
    private long runningQueryLen;

    @JsonProperty("waiting_query_len")
    private long waitingQueryLen;

    @JsonProperty("running_task_info")
    @JsonDeserialize(using = TaskInfoMapDeserializer.class)
    private Map<String, TaskInfo> runningTaskInfo;

    @JsonProperty("waiting_task_info")
    @JsonDeserialize(using = TaskInfoMapDeserializer.class)
    private Map<String, TaskInfo> waitingTaskInfo;

    @JsonProperty("finished_task_info")
    @JsonAlias("finished_task_list")
    @JsonDeserialize(using = TaskInfoMapDeserializer.class)
    private Map<String, TaskInfo> finishedTaskInfo;

    @JsonProperty("status_version")
    private Long statusVersion = 0L;

    @JsonProperty("latest_finished_version")
    private Long latestFinishedVersion = 0L;

    @JsonProperty("cache_status")
    private CacheStatus cacheStatus;

    @JsonProperty("step_latency_ms")
    private double stepLatencyMs;

    @JsonProperty("iterate_count")
    private long iterateCount;

    @JsonProperty("dpSize")
    @JsonAlias("dp_size")
    private long dpSize;

    @JsonProperty("tpSize")
    @JsonAlias("tp_size")
    private long tpSize;

    @JsonProperty("profile_name")
    @JsonAlias("profileName")
    private String profileName;

    @JsonProperty("cp_size")
    @JsonAlias("cpSize")
    private long cpSize = -1L;

    @JsonProperty("world_size")
    @JsonAlias("worldSize")
    private long worldSize = -1L;

    @JsonProperty("local_world_size")
    @JsonAlias("localWorldSize")
    private long localWorldSize = -1L;

    @JsonProperty("cp_rotate_method")
    @JsonAlias("cpRotateMethod")
    private String cpRotateMethod;

    @JsonProperty("cp_epoch")
    @JsonAlias("cpEpoch")
    private long cpEpoch = -1L;

    @JsonProperty("cp_prepare_epoch")
    @JsonAlias("cpPrepareEpoch")
    private long cpPrepareEpoch = -1L;

    @JsonProperty("cp_phase")
    @JsonAlias("cpPhase")
    private String cpPhase;

    @JsonProperty("cp_target_size")
    @JsonAlias("cpTargetSize")
    private long cpTargetSize = -1L;

    @JsonProperty("logical_lane_count")
    @JsonAlias("logicalLaneCount")
    private long logicalLaneCount = -1L;

    @JsonProperty("cp_ranks_per_endpoint")
    @JsonAlias("cpRanksPerEndpoint")
    private long cpRanksPerEndpoint = -1L;

    /**
     * Boxed on purpose: absent must stay distinguishable from false, because a worker
     * that publishes nothing has to keep serving. See WorkerStatus#cpAcceptsExternal.
     */
    @JsonProperty("cp_accepts_external")
    @JsonAlias("cpAcceptsExternal")
    private Boolean cpAcceptsExternal;

    @JsonProperty("route_fanout")
    @JsonAlias("routeFanout")
    private long routeFanout = -1L;

    @JsonProperty("alive")
    private boolean alive;

    @JsonProperty("version")
    private long version;

    @JsonProperty("message")
    private String message;

    @JsonProperty("address")
    private String address;

    @JsonProperty("results")
    private List<WorkerStatusResponse> results;

    public Map<String, TaskInfo> getRunningTaskInfo() {
        if (!hasWaitingTasksInRunningReport()) {
            return runningTaskInfo;
        }
        Map<String, TaskInfo> running = new LinkedHashMap<>();
        runningTaskInfo.forEach((id, task) -> {
            if (!isReportedWaiting(task)) {
                running.put(id, task);
            }
        });
        return running;
    }

    public Map<String, TaskInfo> getWaitingTaskInfo() {
        if (!hasWaitingTasksInRunningReport()) {
            return waitingTaskInfo;
        }
        Map<String, TaskInfo> waiting = new LinkedHashMap<>();
        if (waitingTaskInfo != null) {
            waiting.putAll(waitingTaskInfo);
        }
        runningTaskInfo.forEach((id, task) -> {
            if (isReportedWaiting(task)) {
                waiting.putIfAbsent(id, task);
            }
        });
        return waiting;
    }

    private boolean hasWaitingTasksInRunningReport() {
        return runningTaskInfo != null && runningTaskInfo.values().stream()
                .anyMatch(WorkerStatusResponse::isReportedWaiting);
    }

    private static boolean isReportedWaiting(TaskInfo task) {
        return task instanceof ReportedTaskInfo && ((ReportedTaskInfo) task).waiting;
    }

    // Protobuf JSON combines pending and running tasks in running_task_info.
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ReportedTaskInfo extends TaskInfo {
        @JsonProperty("is_waiting")
        private boolean waiting;
    }

    public static class TaskInfoMapDeserializer extends JsonDeserializer<Map<String, TaskInfo>> {
        @Override
        public Map<String, TaskInfo> deserialize(
                JsonParser parser, DeserializationContext context) throws IOException {
            JsonNode node = parser.getCodec().readTree(parser);
            Map<String, TaskInfo> tasks = new LinkedHashMap<>();
            if (node == null || node.isNull()) {
                return tasks;
            }
            if (node.isObject()) {
                var fields = node.fields();
                while (fields.hasNext()) {
                    var field = fields.next();
                    tasks.put(field.getKey(), parser.getCodec().treeToValue(field.getValue(), ReportedTaskInfo.class));
                }
                return tasks;
            }
            if (node.isArray()) {
                for (JsonNode item : node) {
                    TaskInfo task = parser.getCodec().treeToValue(item, ReportedTaskInfo.class);
                    tasks.put(String.valueOf(task.getRequestId()), task);
                }
                return tasks;
            }
            context.handleUnexpectedToken(Map.class, parser);
            return tasks;
        }
    }
}
