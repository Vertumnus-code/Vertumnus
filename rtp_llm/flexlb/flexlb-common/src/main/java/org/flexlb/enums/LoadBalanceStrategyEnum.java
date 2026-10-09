package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // Random assignment

    SHORTEST_TTFT("ShortestTTFT"),  // Shortest Time-To-First-Token

    WEIGHTED_CACHE("WeightedCache"),  // Lowest cache usage strategy

    CP_AWARE_TTFT("CpAwareTtft"),  // CP-aware analytic TTFT minimization

    STATIC_CP_THRESHOLD("StaticCpThreshold")  // Offline-profile static seq_len threshold routing

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
