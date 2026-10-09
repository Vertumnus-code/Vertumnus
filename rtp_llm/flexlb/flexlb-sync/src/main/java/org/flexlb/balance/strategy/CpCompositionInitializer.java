package org.flexlb.balance.strategy;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.util.JsonUtils;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Selects the initial worker composition from a representative historical
 * workload.
 *
 * <p>The historical record carries the aggregate window-level demand and how that
 * demand splits over CP degrees once each request is attributed to the degree it
 * would be served at, which is what determines the expected mixture of small- and
 * large-degree workers. The profiled capacities then decide whether a candidate
 * composition can accommodate that demand. Among historically suitable
 * compositions with sufficient capacity the one containing more large-degree
 * workers is preferred, so that more parallelism is available to
 * latency-sensitive, compute-intensive requests. If no such composition has
 * sufficient capacity, the composition with the largest aggregate profiled
 * capacity is selected.
 */
final class CpCompositionInitializer {

    private static final int MAX_CANDIDATES = 200_000;

    /** Aggregate demand and per-degree demand mixture of a historical workload. */
    record HistoricalWorkload(long observationWindowMs,
                              double aggregateWindowDemandSeconds,
                              Map<Integer, Double> windowDemandByCp,
                              double longRequestFraction,
                              String source) {

        static HistoricalWorkload absent() {
            return new HistoricalWorkload(0L, Double.NaN, Map.of(), Double.NaN, "absent");
        }

        static HistoricalWorkload load(String path) {
            if (path == null || path.isBlank()) {
                return absent();
            }
            try {
                String json = new String(Files.readAllBytes(Path.of(path.trim())), StandardCharsets.UTF_8);
                return parse(json, path.trim());
            } catch (Exception e) {
                throw new IllegalArgumentException(
                        "unreadable CP historical workload " + path + ": " + e.getMessage(), e);
            }
        }

        static HistoricalWorkload parse(String json, String source) {
            JsonNode root = JsonUtils.toTreeNode(json);
            if (root == null || !root.isObject()) {
                throw new IllegalArgumentException("CP historical workload must be a JSON object");
            }
            Map<Integer, Double> byCp = new LinkedHashMap<>();
            JsonNode mixture = root.path("window_demand_seconds_by_cp");
            if (mixture.isObject()) {
                mixture.fields().forEachRemaining(entry -> {
                    int cp = Integer.parseInt(entry.getKey().trim());
                    double demand = entry.getValue().asDouble(Double.NaN);
                    if (cp <= 0 || !Double.isFinite(demand) || demand < 0.0) {
                        throw new IllegalArgumentException("invalid historical demand entry: " + entry.getKey());
                    }
                    byCp.put(cp, demand);
                });
            }
            double aggregate = root.path("aggregate_window_demand_seconds")
                    .asDouble(byCp.values().stream().mapToDouble(Double::doubleValue).sum());
            if (!Double.isFinite(aggregate) || aggregate < 0.0) {
                throw new IllegalArgumentException("invalid aggregate historical demand");
            }
            return new HistoricalWorkload(root.path("observation_window_ms").asLong(0L),
                    aggregate, byCp,
                    root.path("long_request_fraction").asDouble(Double.NaN), source);
        }

        boolean isPresent() {
            return Double.isFinite(aggregateWindowDemandSeconds);
        }

        double demandAt(int cpSize) {
            return windowDemandByCp.getOrDefault(cpSize, 0.0);
        }

        Map<String, Object> snapshot() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("source", source);
            result.put("observation_window_ms", observationWindowMs);
            result.put("aggregate_window_demand_seconds", aggregateWindowDemandSeconds);
            Map<String, Object> mixture = new LinkedHashMap<>();
            new TreeMap<>(windowDemandByCp)
                    .forEach((cp, demand) -> mixture.put(String.valueOf(cp), demand));
            result.put("window_demand_seconds_by_cp", mixture);
            result.put("long_request_fraction", longRequestFraction);
            return result;
        }
    }

    record Selection(CpComposition composition, String reason,
                     double aggregateCapacitySeconds, double requiredDemandSeconds) {

        Map<String, Object> snapshot() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("reason", reason);
            result.put("composition", composition == null ? null : composition.describe());
            Map<String, Object> vector = new LinkedHashMap<>();
            if (composition != null) {
                composition.vector().forEach((cp, count) -> vector.put(String.valueOf(cp), count));
            }
            result.put("workers_by_cp", vector);
            result.put("aggregate_capacity_seconds", aggregateCapacitySeconds);
            result.put("required_demand_seconds", requiredDemandSeconds);
            return result;
        }
    }

    private CpCompositionInitializer() {
    }

    static Selection select(List<Integer> ladder, int gpuCount,
                            CpCapacityProfile profile, HistoricalWorkload historical) {
        List<Integer> degrees = ladder.stream().distinct().sorted().toList();
        if (gpuCount <= 0 || degrees.isEmpty()) {
            return new Selection(null, "unknown_gpu_count", Double.NaN, Double.NaN);
        }
        if (!profile.covers(degrees)) {
            return new Selection(null, "missing_profiled_capacity", Double.NaN, Double.NaN);
        }
        List<CpComposition> candidates = enumerate(degrees, gpuCount);
        if (candidates.isEmpty()) {
            return new Selection(null, "no_composition_fits_gpu_count", Double.NaN, Double.NaN);
        }
        Comparator<CpComposition> favorLargeDegrees = favorLargeDegrees(degrees);
        if (!historical.isPresent()) {
            CpComposition fallback = candidates.stream().max(favorLargeDegrees).orElseThrow();
            return new Selection(fallback, "no_historical_workload",
                    fallback.profiledWindowCapacity(profile), Double.NaN);
        }

        double required = historical.aggregateWindowDemandSeconds();
        List<CpComposition> eligible = candidates.stream()
                .filter(candidate -> isHistoricallySuitable(candidate, profile, historical, degrees))
                .filter(candidate -> candidate.profiledWindowCapacity(profile) >= required)
                .toList();
        if (!eligible.isEmpty()) {
            CpComposition selected = eligible.stream().max(favorLargeDegrees).orElseThrow();
            return new Selection(selected, "historically_suitable_with_capacity",
                    selected.profiledWindowCapacity(profile), required);
        }
        CpComposition widest = candidates.stream()
                .max(Comparator.comparingDouble((CpComposition candidate) ->
                                candidate.profiledWindowCapacity(profile))
                        .thenComparing(favorLargeDegrees))
                .orElseThrow();
        return new Selection(widest, "largest_aggregate_capacity",
                widest.profiledWindowCapacity(profile), required);
    }

    /**
     * A composition is historically suitable when every degree the historical
     * workload puts demand on has enough profiled capacity at that degree to carry
     * it, which is what ties the worker mixture to the request-length distribution.
     */
    private static boolean isHistoricallySuitable(CpComposition candidate, CpCapacityProfile profile,
                                                  HistoricalWorkload historical, List<Integer> degrees) {
        for (Integer cp : degrees) {
            double demand = historical.demandAt(cp);
            if (demand <= 0.0) {
                continue;
            }
            if (candidate.workerCount(cp) * profile.windowDemand(cp) < demand) {
                return false;
            }
        }
        return true;
    }

    /** Orders compositions by worker count at the largest degree first. */
    private static Comparator<CpComposition> favorLargeDegrees(List<Integer> degrees) {
        Comparator<CpComposition> comparator = null;
        for (int i = degrees.size() - 1; i >= 0; i--) {
            int cp = degrees.get(i);
            Comparator<CpComposition> next =
                    Comparator.comparingInt(candidate -> candidate.workerCount(cp));
            comparator = comparator == null ? next : comparator.thenComparing(next);
        }
        return comparator;
    }

    /** Every positive worker vector over the ladder whose GPU count is exact. */
    static List<CpComposition> enumerate(List<Integer> degrees, int gpuCount) {
        List<CpComposition> result = new ArrayList<>();
        expand(degrees, 0, gpuCount, new LinkedHashMap<>(), result);
        return result;
    }

    private static void expand(List<Integer> degrees, int index, int remainingGpus,
                               Map<Integer, Integer> current, List<CpComposition> result) {
        if (result.size() >= MAX_CANDIDATES) {
            return;
        }
        if (index == degrees.size()) {
            if (remainingGpus == 0 && !current.isEmpty()) {
                result.add(CpComposition.of(current));
            }
            return;
        }
        int cp = degrees.get(index);
        for (int count = remainingGpus / cp; count >= 1; count--) {
            current.put(cp, count);
            expand(degrees, index + 1, remainingGpus - count * cp, current, result);
        }
        current.remove(cp);
    }
}
