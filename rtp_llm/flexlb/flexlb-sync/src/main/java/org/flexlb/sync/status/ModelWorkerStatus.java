package org.flexlb.sync.status;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
@NoArgsConstructor
public class ModelWorkerStatus {

    /**
     * Non-PD separation mode
     */
    private Map<String/*ipPort*/, WorkerStatus> pdFusionStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> prefillStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> decodeStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> vitStatusMap = new ConcurrentHashMap<>();

    private volatile String activeProfileName;

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {
        return switch (roleType) {
            case DECODE -> decodeStatusMap;
            case PREFILL -> prefillStatusMap;
            case PDFUSION -> pdFusionStatusMap;
            case VIT -> vitStatusMap;
            case null -> Map.of();
        };
    }

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType, String profileName) {
        Map<String, WorkerStatus> roleStatusMap = getRoleStatusMap(roleType);
        if (profileName == null || profileName.trim().isEmpty()) {
            return roleStatusMap;
        }
        return roleStatusMap.entrySet()
                .stream()
                .filter(entry -> entry.getValue() != null && entry.getValue().matchesProfile(profileName))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    public List<RoleType> getRoleTypeList() {
        List<RoleType> roleTypeList = new ArrayList<>();
        if (!pdFusionStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.PDFUSION);
        }
        if (!decodeStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.DECODE);
        }
        if (!prefillStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.PREFILL);
        }
        if (!vitStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.VIT);
        }
        return roleTypeList;
    }

    public Set<String> getKnownProfiles() {
        return allWorkers()
                .filter(workerStatus -> workerStatus != null)
                .map(WorkerStatus::getProfileKey)
                .filter(profile -> profile != null && !profile.isBlank())
                .collect(Collectors.toCollection(TreeSet::new));
    }

    private Stream<WorkerStatus> allWorkers() {
        return Stream.of(pdFusionStatusMap, prefillStatusMap, decodeStatusMap, vitStatusMap)
                .flatMap(statusMap -> statusMap.values().stream());
    }

    public int getWorkerTotalCount() {
        return pdFusionStatusMap.size() + decodeStatusMap.size() + prefillStatusMap.size() + vitStatusMap.size();
    }
}
