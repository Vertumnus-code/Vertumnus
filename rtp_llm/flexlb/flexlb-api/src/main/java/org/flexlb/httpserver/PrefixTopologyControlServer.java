package org.flexlb.httpserver;

import org.flexlb.sync.prefix.PrefixTopologyControl;
import org.flexlb.sync.prefix.PrefixTransferRecoveryService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Map;
import java.util.function.Supplier;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
@ConditionalOnProperty(name = "prefix-cache.topology.control.enabled", havingValue = "true")
public final class PrefixTopologyControlServer {
    private static final Logger LOG = LoggerFactory.getLogger(PrefixTopologyControlServer.class);
    private final PrefixTransferRecoveryService service;
    private final byte[] authorization;

    public PrefixTopologyControlServer(PrefixTransferRecoveryService service, Environment environment) {
        this.service = service;
        String token = environment.getRequiredProperty("prefix-cache.topology.control.token");
        if (!token.matches("[A-Za-z0-9_-]{32,256}")) { throw new IllegalArgumentException("invalid topology control credential configuration"); }
        authorization = ("Bearer " + token).getBytes(StandardCharsets.UTF_8);
    }

    @Bean
    public RouterFunction<ServerResponse> prefixTopologyControl() {
        return route().POST("/rtp_llm/prefix_topology/plans", request -> authorized(request)
                ? request.bodyToMono(PrefixTopologyControl.Request.class).flatMap(plan -> execute(() -> service.topologyControl().prepare(plan)))
                : denied())
                .POST("/rtp_llm/prefix_topology/plans/{id}/advance", request -> authorized(request)
                        ? execute(() -> service.topologyControl().advance(request.pathVariable("id"))) : denied())
                .POST("/rtp_llm/prefix_topology/plans/{id}/admit", request -> authorized(request)
                        ? execute(() -> service.topologyControl().admit(request.pathVariable("id"))) : denied())
                .GET("/rtp_llm/prefix_topology/plans/{id}", request -> authorized(request)
                        ? execute(() -> service.topologyControl().read(request.pathVariable("id"))) : denied()).build();
    }

    private boolean authorized(ServerRequest request) {
        String value = request.headers().firstHeader("Authorization");
        return value != null && value.length() <= 263 && MessageDigest.isEqual(authorization, value.getBytes(StandardCharsets.UTF_8));
    }

    private Mono<ServerResponse> denied() { return ServerResponse.status(403).bodyValue(Map.of("error", "forbidden")); }

    private Mono<ServerResponse> execute(Supplier<PrefixTopologyControl.Status> operation) {
        return Mono.fromCallable(operation::get).subscribeOn(Schedulers.boundedElastic()).flatMap(value -> ServerResponse.ok().bodyValue(value))
                .onErrorResume(IllegalArgumentException.class, failure -> ServerResponse.badRequest().bodyValue(Map.of("error", "invalid_immutable_request")))
                .onErrorResume(IllegalStateException.class, failure -> {
                    LOG.debug("Topology control operation is not confirmed", failure);
                    return ServerResponse.status(409).bodyValue(Map.of("error", "topology_not_confirmed"));
                });
    }
}
