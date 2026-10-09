package org.flexlb.cache.prefix;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public record PrefixCacheSettings(String policy, String metadataMode, String namespace,
                                   long reportTtlMillis, int maxNodes, int maxObservations,
                                   long scanMillis, double interThreshold, double intraThreshold,
                                   double deleteThreshold, double decay, int operationsPerList,
                                   long destinationBudgetBytes, long wireBudgetBytes, long safetyBytes,
                                   long nodeIdleTtlMillis, long observationTtlMillis, Set<String> enabledActions,
                                   String routingCacheView) {
    public static final Set<String> ACTION_KINDS = Set.of("inter", "intra", "reclaim");

    public static PrefixCacheSettings fromEnvironment() {
        return from(System.getenv());
    }

    public static PrefixCacheSettings from(Map<String, String> environment) {
        String policy = environment.getOrDefault("FLEXLB_PREFIX_CACHE_POLICY", "native");
        String metadata = environment.getOrDefault("FLEXLB_CACHE_METADATA_MODE", "legacy_optimistic");
        String namespace = environment.getOrDefault("PREFIX_CACHE_NAMESPACE", "");
        if (!policy.equals("native") && !policy.equals("observe") && !policy.equals("adaptive")) {
            throw new IllegalArgumentException("unknown prefix cache policy");
        }
        if (!metadata.equals("legacy_optimistic") && !metadata.equals("confirmed")) {
            throw new IllegalArgumentException("unknown cache metadata mode");
        }
        if ((!policy.equals("native") || metadata.equals("confirmed")) && namespace.isBlank()) {
            throw new IllegalArgumentException("PREFIX_CACHE_NAMESPACE is required");
        }
        if (!policy.equals("native") && !metadata.equals("confirmed")) {
            throw new IllegalArgumentException("prefix observation and adaptive actions require confirmed metadata");
        }
        String transferMode = environment.getOrDefault("PREFIX_TRANSFER_MODE", "disabled");
        if (policy.equals("adaptive") ? !Set.of("tcp", "rdma").contains(transferMode) : !transferMode.equals("disabled")) {
            throw new IllegalArgumentException("only explicitly configured adaptive policy can use TCP or RDMA prefix actions");
        }
        return new PrefixCacheSettings(policy, metadata, namespace,
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_REPORT_TTL_MS", "5000")),
                Integer.parseInt(environment.getOrDefault("FLEXLB_PREFIX_MAX_NODES", "100000")),
                Integer.parseInt(environment.getOrDefault("FLEXLB_PREFIX_MAX_OBSERVATIONS", "10000")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_SCAN_MS", "10000")),
                Double.parseDouble(environment.getOrDefault("FLEXLB_PREFIX_THETA_INTER", "1.0")),
                Double.parseDouble(environment.getOrDefault("FLEXLB_PREFIX_THETA_INTRA", "10.0")),
                Double.parseDouble(environment.getOrDefault("FLEXLB_PREFIX_THETA_DELETE", "0.1")),
                Double.parseDouble(environment.getOrDefault("FLEXLB_PREFIX_DECAY", "0.5")),
                Integer.parseInt(environment.getOrDefault("FLEXLB_PREFIX_OPS_PER_LIST", "16")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_DESTINATION_BUDGET_BYTES", "67108864")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_WIRE_BUDGET_BYTES", "67108864")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_SAFETY_BYTES", "16777216")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_NODE_IDLE_TTL_MS", "3600000")),
                Long.parseLong(environment.getOrDefault("FLEXLB_PREFIX_OBSERVATION_TTL_MS", "300000")),
                parseActions(environment.getOrDefault("FLEXLB_PREFIX_ENABLED_ACTIONS", "inter,intra,reclaim")),
                environment.getOrDefault("FLEXLB_PREFIX_ROUTING_CACHE_VIEW", "confirmed"));
    }

    public PrefixCacheSettings(String policy, String metadataMode, String namespace,
                               long reportTtlMillis, int maxNodes, int maxObservations,
                               long scanMillis, double interThreshold, double intraThreshold,
                               double deleteThreshold, double decay, int operationsPerList,
                               long destinationBudgetBytes, long wireBudgetBytes, long safetyBytes,
                               long nodeIdleTtlMillis, long observationTtlMillis, Set<String> enabledActions) {
        this(policy, metadataMode, namespace, reportTtlMillis, maxNodes, maxObservations,
                scanMillis, interThreshold, intraThreshold, deleteThreshold, decay, operationsPerList,
                destinationBudgetBytes, wireBudgetBytes, safetyBytes, nodeIdleTtlMillis,
                observationTtlMillis, enabledActions, "confirmed");
    }

    private static Set<String> parseActions(String value) {
        if (value.trim().equals("none")) { return Set.of(); }
        Set<String> actions = new HashSet<>();
        for (String action : value.split(",", -1)) {
            if (!ACTION_KINDS.contains(action.trim()) || !actions.add(action.trim())) {
                throw new IllegalArgumentException("invalid or duplicate prefix action kind");
            }
        }
        return Set.copyOf(actions);
    }

    public PrefixCacheSettings {
        enabledActions = Collections.unmodifiableSet(new TreeSet<>(enabledActions));
        if (!ACTION_KINDS.containsAll(enabledActions)) {
            throw new IllegalArgumentException("unknown prefix action kind");
        }
        if (!java.util.Set.of("native", "observe", "adaptive").contains(policy)
                || !java.util.Set.of("legacy_optimistic", "confirmed").contains(metadataMode)
                || !java.util.Set.of("confirmed", "main_compatible").contains(routingCacheView)
                || (routingCacheView.equals("main_compatible") && !metadataMode.equals("confirmed"))
                || ((!policy.equals("native") || metadataMode.equals("confirmed")) && namespace.isBlank())
                || (!policy.equals("native") && !metadataMode.equals("confirmed"))
                || reportTtlMillis <= 0 || maxNodes <= 0 || maxObservations <= 0 || scanMillis <= 0
                || !Double.isFinite(interThreshold) || interThreshold < 0
                || !Double.isFinite(intraThreshold) || intraThreshold <= 0
                || !Double.isFinite(deleteThreshold) || deleteThreshold < 0 || deleteThreshold >= intraThreshold
                || !(decay > 0 && decay < 1) || operationsPerList < 0
                || destinationBudgetBytes < 0 || wireBudgetBytes < 0 || safetyBytes < 0
                || nodeIdleTtlMillis <= 0 || observationTtlMillis <= 0) {
            throw new IllegalArgumentException("invalid prefix cache limits/thresholds");
        }
    }

    public boolean confirmed() { return metadataMode.equals("confirmed"); }
    public boolean mainCompatibleRouting() { return routingCacheView.equals("main_compatible"); }
    public boolean observe() { return policy.equals("observe") || adaptive(); }
    public boolean adaptive() { return policy.equals("adaptive"); }
}
