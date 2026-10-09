package org.flexlb.httpserver;

import org.flexlb.balance.strategy.CpAwareTtftStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import java.util.Map;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class PrefixCacheStatusServer {
    private final CacheAwareService cacheAwareService;
    private final ConfigService configService;
    private final EngineGrpcClient grpcClient;
    private final MasterEngineSynchronizer synchronizer;

    public PrefixCacheStatusServer(CacheAwareService cacheAwareService, ConfigService configService,
                                  EngineGrpcClient grpcClient, MasterEngineSynchronizer synchronizer) {
        this.cacheAwareService = cacheAwareService;
        this.configService = configService;
        this.grpcClient = grpcClient;
        this.synchronizer = synchronizer;
        if (cacheAwareService.prefixObserver().settings().observe()
                && configService.loadBalanceConfig().getStrategyForRoleType(RoleType.PREFILL)
                    != LoadBalanceStrategyEnum.CP_AWARE_TTFT) {
            throw new IllegalArgumentException("prefix observe requires CP_AWARE_TTFT routing");
        }
    }

    @Bean
    public RouterFunction<ServerResponse> prefixCacheStatus() {
        return route().GET("/rtp_llm/prefix_cache_status", request -> {
            var observer = cacheAwareService.prefixObserver();
            return ServerResponse.ok().bodyValue(Map.of(
                    "scheduler", CpAwareTtftStrategy.effectiveConfiguration(),
                    "routing", configService.loadBalanceConfig(),
                    "prefix_cache", observer.settings(),
                    "control_rpc_latency", grpcClient.prefixRpcLatency(),
                    "inventory_refresh", synchronizer.prefixInventoryRefreshStatus(),
                    "observation", observer.status()));
        }).build();
    }
}
