package org.flexlb.balance.strategy;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NavigableSet;
import java.util.TreeMap;

/**
 * Offline capacity profile of a single worker at each supported CP degree.
 *
 * <p>For a degree-C worker, kappa_C is the largest sum of E_C service seconds
 * admitted in one profiling window while the worker still satisfies the target
 * TTFT SLO attainment. Runtime applies the same degree-local pricing to every
 * request after the scheduler has selected its worker. All values therefore have
 * the common physical unit of worker-service seconds, while preserving the fact
 * that the same request has a different service time at a different CP degree.
 */
final class CpCapacityProfile {

    private static final CpCapacityProfile EMPTY =
            new CpCapacityProfile("", 0L, Double.NaN, Double.NaN, Map.of(), "absent");

    private final String model;
    private final long profilingWindowMs;
    private final double sloTtftSeconds;
    private final double attainmentTarget;
    private final Map<Integer, Double> windowDemandByCp;
    private final String source;

    private CpCapacityProfile(String model,
                              long profilingWindowMs,
                              double sloTtftSeconds,
                              double attainmentTarget,
                              Map<Integer, Double> windowDemandByCp,
                              String source) {
        this.model = model == null ? "" : model;
        this.profilingWindowMs = profilingWindowMs;
        this.sloTtftSeconds = sloTtftSeconds;
        this.attainmentTarget = attainmentTarget;
        this.windowDemandByCp = Map.copyOf(windowDemandByCp);
        this.source = source;
    }

    static CpCapacityProfile absent() {
        return EMPTY;
    }

    /**
     * Reads the profiling artifact, falling back to the inline capacity rates.
     *
     * @param profilePath  path of the artifact written by the offline profiler
     * @param inlineRates  {@code C:rate} pairs expressed in demand seconds per
     *                     wall second, the form a profile-less deployment uses
     * @param runtimeWindowMs the runtime observation window, adopted as H when
     *                     only inline rates are given
     */
    static CpCapacityProfile load(String profilePath, String inlineRates, long runtimeWindowMs) {
        if (profilePath != null && !profilePath.isBlank()) {
            try {
                String json = new String(Files.readAllBytes(Path.of(profilePath.trim())),
                        StandardCharsets.UTF_8);
                return parse(json, runtimeWindowMs, profilePath.trim());
            } catch (Exception e) {
                throw new IllegalArgumentException(
                        "unreadable CP capacity profile " + profilePath + ": " + e.getMessage(), e);
            }
        }
        Map<Integer, Double> rates = CpCompositionController.parseCapacityRates(inlineRates);
        if (rates.isEmpty()) {
            return EMPTY;
        }
        return fromDemandRates(rates, runtimeWindowMs, "FLEXLB_CP_PROFILED_CAPACITY");
    }

    static CpCapacityProfile parse(String json, long runtimeWindowMs, String source) {
        JsonNode root = JsonUtils.toTreeNode(json);
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("CP capacity profile must be a JSON object");
        }
        long windowMs = root.path("observation_window_ms").asLong(0L);
        JsonNode kappa = root.path("capacity_window_demand_seconds");
        JsonNode rates = root.path("capacity_demand_rate");
        Map<Integer, Double> windowDemand = new LinkedHashMap<>();
        if (kappa.isObject()) {
            if (windowMs <= 0L) {
                throw new IllegalArgumentException(
                        "CP capacity profile must declare observation_window_ms alongside kappa");
            }
            kappa.fields().forEachRemaining(entry ->
                    windowDemand.put(positiveDegree(entry.getKey()), positiveValue(entry.getValue().asDouble(Double.NaN))));
        } else if (rates.isObject()) {
            long effectiveWindowMs = windowMs > 0L ? windowMs : runtimeWindowMs;
            rates.fields().forEachRemaining(entry ->
                    windowDemand.put(positiveDegree(entry.getKey()),
                            positiveValue(entry.getValue().asDouble(Double.NaN)) * effectiveWindowMs / 1000.0));
            windowMs = effectiveWindowMs;
        } else {
            throw new IllegalArgumentException("CP capacity profile declares no capacity");
        }
        if (windowDemand.isEmpty()) {
            throw new IllegalArgumentException("CP capacity profile declares no capacity");
        }
        if (root.has("demand_reference_cp")) {
            throw new IllegalArgumentException(
                    "reference-CP capacity profiles are incompatible with selected-CP demand pricing; re-profile every degree using E_C");
        }
        String demandPricing = root.path("demand_pricing").asText("selected_cp");
        if (!"selected_cp".equalsIgnoreCase(demandPricing)
                && !"worker_degree".equalsIgnoreCase(demandPricing)) {
            throw new IllegalArgumentException(
                    "unsupported CP demand pricing: " + demandPricing);
        }
        return new CpCapacityProfile(root.path("model").asText(""),
                windowMs,
                root.path("slo_ttft_seconds").asDouble(Double.NaN),
                root.path("attainment_target").asDouble(Double.NaN),
                windowDemand,
                source);
    }

    static CpCapacityProfile fromDemandRates(Map<Integer, Double> demandRateByCp,
                                             long windowMs, String source) {
        long effectiveWindowMs = Math.max(1L, windowMs);
        Map<Integer, Double> windowDemand = new LinkedHashMap<>();
        demandRateByCp.forEach((cp, rate) ->
                windowDemand.put(positiveDegree(String.valueOf(cp)),
                        positiveValue(rate) * effectiveWindowMs / 1000.0));
        return new CpCapacityProfile("", effectiveWindowMs, Double.NaN, Double.NaN,
                windowDemand, source);
    }

    boolean isPresent() {
        return !windowDemandByCp.isEmpty();
    }

    boolean covers(Collection<Integer> degrees) {
        return isPresent() && windowDemandByCp.keySet().containsAll(degrees);
    }

    NavigableSet<Integer> degrees() {
        return new TreeMap<>(windowDemandByCp).navigableKeySet();
    }

    long profilingWindowMs() {
        return profilingWindowMs;
    }

    /** kappa_C, in predicted service seconds per observation window. */
    double windowDemand(int cpSize) {
        Double value = windowDemandByCp.get(cpSize);
        return value == null ? Double.NaN : value;
    }

    /** kappa_C normalized to predicted service seconds per wall second. */
    double demandRate(int cpSize) {
        double kappa = windowDemand(cpSize);
        return profilingWindowMs > 0L ? kappa * 1000.0 / profilingWindowMs : Double.NaN;
    }

    /** Logs the deviation the paper's shared-H assumption forbids. */
    void warnOnWindowMismatch(long runtimeWindowMs) {
        if (isPresent() && profilingWindowMs > 0L && profilingWindowMs != runtimeWindowMs) {
            Logger.warn("CP capacity profile was measured over H={}ms but the runtime observation window is {}ms; capacities are rescaled per wall second",
                    profilingWindowMs, runtimeWindowMs);
        }
    }

    Map<String, Object> snapshot() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("source", source);
        result.put("model", model);
        result.put("observation_window_ms", profilingWindowMs);
        result.put("slo_ttft_seconds", sloTtftSeconds);
        result.put("attainment_target", attainmentTarget);
        result.put("demand_pricing", "selected_cp");
        Map<String, Object> kappa = new LinkedHashMap<>();
        Map<String, Object> rate = new LinkedHashMap<>();
        for (Integer cp : degrees()) {
            kappa.put(String.valueOf(cp), windowDemand(cp));
            rate.put(String.valueOf(cp), demandRate(cp));
        }
        result.put("capacity_window_demand_seconds", kappa);
        result.put("capacity_demand_rate", rate);
        return result;
    }

    private static int positiveDegree(String raw) {
        int cp = Integer.parseInt(raw.trim());
        if (cp <= 0) {
            throw new IllegalArgumentException("CP degree must be positive: " + raw);
        }
        return cp;
    }

    private static double positiveValue(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException("profiled capacity must be finite and positive: " + value);
        }
        return value;
    }
}
