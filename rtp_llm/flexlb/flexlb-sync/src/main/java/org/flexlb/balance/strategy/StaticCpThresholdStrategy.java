package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Production baseline: offline-profiled static seq_len ladder.
 *
 * <p>Two configuration modes, checked in order:
 * <ol>
 *   <li>{@code FLEXLB_CP_TIERS} — multi-tier ladder, comma-separated
 *       {@code <maxSeqExclusive>:<cpSize>} pairs. Example:
 *       {@code 1024:2,8192:4,-1:8} means seq&lt;1024 → cp=2, 1024≤seq&lt;8192 → cp=4,
 *       seq≥8192 → cp=8. Use {@code -1} for the unbounded catch-all tier.</li>
 *   <li>{@code FLEXLB_CP_THRESHOLD} — legacy single threshold; seq&lt;threshold uses the
 *       smallest-CP tier available, otherwise the largest-CP tier.</li>
 * </ol>
 *
 * <p>If the desired CP tier isn't present in the live fleet, falls back to the
 * nearest larger CP (safer for long prompts), then nearest smaller CP.
 *
 * <p>Within a tier the selection is delegated to {@link ShortestTTFTStrategy},
 * so the intra-tier policy is not a reimplementation of the production router but
 * the production router itself: the same {@code tokens - 0.7*hit} estimate, the
 * same queue term, and the same top-30%-then-fairness tie-break. That is what
 * makes this an honest baseline. The length ladder is the only thing it adds, and
 * therefore the only thing a comparison against it measures: parallel degree
 * chosen once per length band, offline, versus chosen per request from the
 * request's own predicted cost. Set {@code FLEXLB_CP_STATIC_INNER=LRU} to fall
 * back to round-robin within a tier, which additionally blinds the baseline to
 * cache and queue.
 *
 * <p>CP size is read from {@link WorkerStatus#getCpSize()} when available and
 * falls back to {@link WorkerStatus#getTpSize()} when the engine does not yet
 * emit cp_size.
 */
@Component("staticCpThresholdStrategy")
public class StaticCpThresholdStrategy implements LoadBalancer {

    private static final long DEFAULT_THRESHOLD_TOKENS = 2860L;
    private static final String ENV_THRESHOLD = "FLEXLB_CP_THRESHOLD";
    private static final String ENV_TIERS = "FLEXLB_CP_TIERS";

    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final ShortestTTFTStrategy intraTierStrategy;
    private final boolean intraTierUsesRtp;
    private final long thresholdTokens;
    private final List<Tier> tiers;

    /** A tier is: seq &lt; maxSeqExclusive → prefer cpSize. Sorted ascending by maxSeqExclusive. */
    private record Tier(long maxSeqExclusive, long cpSize) {}

    public StaticCpThresholdStrategy(EngineWorkerStatus engineWorkerStatus,
                                     ResourceMeasureFactory resourceMeasureFactory,
                                     ShortestTTFTStrategy intraTierStrategy) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.intraTierStrategy = intraTierStrategy;
        this.intraTierUsesRtp = !"LRU".equalsIgnoreCase(String.valueOf(System.getenv("FLEXLB_CP_STATIC_INNER")));
        this.tiers = parseTiersFromEnv();
        this.thresholdTokens = readThresholdFromEnv();
        if (!tiers.isEmpty()) {
            Logger.info("StaticCpThresholdStrategy initialized (multi-tier), tiers={}, intraTier={}",
                    tiers, intraTierUsesRtp ? "SHORTEST_TTFT" : "LRU");
        } else {
            Logger.info("StaticCpThresholdStrategy initialized (single-threshold), threshold={} tokens, intraTier={}",
                    thresholdTokens, intraTierUsesRtp ? "SHORTEST_TTFT" : "LRU");
        }
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.STATIC_CP_THRESHOLD, this);
    }

    private static long readThresholdFromEnv() {
        String raw = System.getenv(ENV_THRESHOLD);
        if (raw == null || raw.trim().isEmpty()) {
            return DEFAULT_THRESHOLD_TOKENS;
        }
        try {
            return Long.parseLong(raw.trim());
        } catch (NumberFormatException ex) {
            Logger.warn("Invalid {}={}, falling back to default {}", ENV_THRESHOLD, raw, DEFAULT_THRESHOLD_TOKENS);
            return DEFAULT_THRESHOLD_TOKENS;
        }
    }

    private static List<Tier> parseTiersFromEnv() {
        String raw = System.getenv(ENV_TIERS);
        if (raw == null || raw.trim().isEmpty()) {
            return List.of();
        }
        List<Tier> parsed = new ArrayList<>();
        for (String piece : raw.split(",")) {
            String p = piece.trim();
            if (p.isEmpty()) {
                continue;
            }
            String[] kv = p.split(":");
            if (kv.length != 2) {
                Logger.warn("Invalid {} entry {} (expected max:cp), ignoring", ENV_TIERS, p);
                continue;
            }
            try {
                long max = Long.parseLong(kv[0].trim());
                long cp = Long.parseLong(kv[1].trim());
                if (cp <= 0) {
                    Logger.warn("Invalid {} entry {} (cp<=0), ignoring", ENV_TIERS, p);
                    continue;
                }
                parsed.add(new Tier(max < 0 ? Long.MAX_VALUE : max, cp));
            } catch (NumberFormatException ex) {
                Logger.warn("Invalid {} entry {} (not numeric), ignoring", ENV_TIERS, p);
            }
        }
        if (parsed.isEmpty()) {
            return List.of();
        }
        parsed.sort(Comparator.comparingLong(Tier::maxSeqExclusive));
        if (parsed.getLast().maxSeqExclusive() != Long.MAX_VALUE) {
            Logger.warn("{} has no catch-all tier (-1:cp); seq_len >= {} will fall back to largest CP",
                    ENV_TIERS, parsed.getLast().maxSeqExclusive());
        }
        return List.copyOf(parsed);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("Static-CP-threshold selection failed", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();

        FlexlbConfig config = balanceContext.getConfig();
        List<WorkerStatus> available = getAvailableWorkers(
                roleType, group, config.getResourceMeasureIndicator(roleType), requestId);
        if (CollectionUtils.isEmpty(available)) {
            Logger.warn("Static-CP-threshold: no available workers for role {}", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        long desiredCp = desiredCpForSeq(seqLen, available);
        long targetCp = resolveAvailableCp(desiredCp, available);

        List<WorkerStatus> tier = available.stream()
                .filter(w -> effectiveCpSize(w) == targetCp)
                .toList();
        if (tier.isEmpty()) {
            tier = available;
        }

        WorkerStatus chosen;
        long hitCacheTokens = 0L;
        if (intraTierUsesRtp) {
            Map<String, Integer> cacheMatchResults =
                    intraTierStrategy.getCacheMatchResults(balanceContext, roleType, group);
            ScoredWorker best = intraTierStrategy.selectBestWorker(
                    intraTierStrategy.scoreWorkers(tier, cacheMatchResults, seqLen));
            chosen = best == null ? null : best.worker();
            hitCacheTokens = best == null ? 0L : best.hitCacheTokens();
        } else {
            chosen = pickLeastRecentlyUsed(tier);
        }
        if (chosen == null) {
            available.forEach(worker -> worker.unpinDpRank(requestId));
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        final WorkerStatus selected = chosen;
        available.stream()
                .filter(worker -> worker != selected)
                .forEach(worker -> worker.unpinDpRank(requestId));

        int dpRank = chosen.resolveDpRank(requestId);
        Logger.info("ROUTE_DECISION strategy=STATIC_CP_THRESHOLD requestId={} seqLen={} hit={} worker={}:{} cp={} tierCp={}",
                requestId, seqLen, hitCacheTokens, chosen.getIp(), chosen.getPort(),
                effectiveCpSize(chosen), desiredCp);
        Logger.debug("Static-CP-threshold picked ip={} port={} cp={} dpRank={} hit={} (seq={}, desiredCp={}, targetCp={})",
                chosen.getIp(), chosen.getPort(), effectiveCpSize(chosen), dpRank, hitCacheTokens,
                seqLen, desiredCp, targetCp);

        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(seqLen);
        task.setPrefixLength(hitCacheTokens);
        task.setDpRank(dpRank);
        chosen.putLocalTask(requestId, task);

        return buildServerStatus(chosen, roleType, requestId, seqLen);
    }

    private List<WorkerStatus> getAvailableWorkers(
            RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator,
            long requestId) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            return new ArrayList<>();
        }
        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            Logger.warn("Static-CP-threshold: no ResourceMeasure for indicator {}", indicator);
            return new ArrayList<>();
        }
        List<WorkerStatus> available = new ArrayList<>();
        for (WorkerStatus worker : workerStatusMap.values()) {
            if (worker == null || !worker.isAlive()) {
                continue;
            }

            // Availability belongs to a concrete DP lane, not to the
            // requestId-hashed rank. Select and pin the least-loaded live lane
            // before filtering the physical worker.
            worker.selectAndPinDpRank(requestId);
            if (resourceMeasure.isResourceAvailable(worker, requestId)) {
                available.add(worker);
            } else {
                worker.unpinDpRank(requestId);
            }
        }
        return available;
    }

    private long desiredCpForSeq(long seqLen, List<WorkerStatus> available) {
        if (!tiers.isEmpty()) {
            for (Tier t : tiers) {
                if (seqLen < t.maxSeqExclusive()) {
                    return t.cpSize();
                }
            }
            return tiers.getLast().cpSize();
        }
        long minCp = available.stream().mapToLong(StaticCpThresholdStrategy::effectiveCpSize).min().orElse(1L);
        long maxCp = available.stream().mapToLong(StaticCpThresholdStrategy::effectiveCpSize).max().orElse(1L);
        return seqLen >= thresholdTokens ? maxCp : minCp;
    }

    private static long resolveAvailableCp(long desiredCp, List<WorkerStatus> available) {
        long[] cps = available.stream().mapToLong(StaticCpThresholdStrategy::effectiveCpSize).distinct().sorted().toArray();
        if (cps.length == 0) {
            return desiredCp;
        }
        for (long cp : cps) {
            if (cp == desiredCp) {
                return cp;
            }
        }
        long nearestUp = -1L;
        for (long cp : cps) {
            if (cp > desiredCp) {
                nearestUp = cp;
                break;
            }
        }
        if (nearestUp > 0) {
            return nearestUp;
        }
        return cps[cps.length - 1];
    }

    private static long effectiveCpSize(WorkerStatus worker) {
        long cp = worker.getCpSize();
        if (cp > 0) {
            return cp;
        }
        long tp = worker.getTpSize();
        return tp > 0 ? tp : 1L;
    }

    private static WorkerStatus pickLeastRecentlyUsed(List<WorkerStatus> tier) {
        List<WorkerStatus> sorted = tier.stream()
                .sorted(Comparator.comparingLong(w -> w.getLastSelectedTime().get()))
                .toList();
        long now = System.nanoTime() / 1000;
        for (WorkerStatus candidate : sorted) {
            long expected = candidate.getLastSelectedTime().get();
            if (candidate.getLastSelectedTime().compareAndSet(expected, now)) {
                return candidate;
            }
        }
        return sorted.isEmpty() ? null : sorted.getFirst();
    }

    private ServerStatus buildServerStatus(WorkerStatus worker, RoleType roleType, long requestId, long seqLen) {
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setRequestId(requestId);
            result.setPrefillTime(seqLen);
            result.setGroup(worker.getGroup());
            result.setServerIp(worker.getIp());
            result.setHttpPort(worker.getPort());
            int dpRank = worker.resolveDpRank(requestId);
            result.setGrpcPort(dpRankGrpcPort(worker, dpRank));
            result.setProfileName(worker.getProfileKey());
            result.setCpSize(worker.getCpSize());
            result.setTpSize(worker.getTpSize());
        } catch (Exception e) {
            Logger.error("Static-CP-threshold: failed to build server status for requestId {}", requestId, e);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
            result.setSuccess(false);
        }
        return result;
    }

    private static int dpRankGrpcPort(WorkerStatus worker, int dpRank) {
        int workerPortStride = readIntEnv("WORKER_INFO_PORT_NUM", 8);
        long worldRank = (long) dpRank * Math.max(1L, worker.getTpSize());
        return CommonUtils.toGrpcPort(worker.getPort())
                + Math.toIntExact(worldRank * workerPortStride);
    }

    private static int readIntEnv(String envName, int defaultValue) {
        String raw = System.getenv(envName);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            Logger.warn("Invalid {}={}, fallback to {}", envName, raw, defaultValue);
            return defaultValue;
        }
    }
}
