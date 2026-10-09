package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.balance.strategy.CpCompositionController;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Slf4j
@Component
public class HttpLoadBalanceServer {
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");
    private final GeneralHttpNettyService generalHttpNettyService;
    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final QueueManager queueManager;
    private final ActiveRequestCounter activeRequestCounter;
    private final EngineWorkerStatus engineWorkerStatus;
    private final CpCompositionController cpCompositionController;

    public HttpLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                 RouteService routeService,
                                 LBStatusConsistencyService lbStatusConsistencyService,
                                 EngineHealthReporter engineHealthReporter,
                                 QueueManager queueManager,
                                 ActiveRequestCounter activeRequestCounter,
                                 EngineWorkerStatus engineWorkerStatus,
                                 CpCompositionController cpCompositionController) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.queueManager = queueManager;
        this.activeRequestCounter = activeRequestCounter;
        this.engineWorkerStatus = engineWorkerStatus;
        this.cpCompositionController = cpCompositionController;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
                .POST("/rtp_llm/schedule", accept(MediaType.APPLICATION_JSON),
                        this::scheduleRequest)
                .POST("/rtp_llm/master/info", accept(MediaType.APPLICATION_JSON),
                        this::responseMasterInfo)
                .POST("/rtp_llm/schedule_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::dumpLBStatus)
                .POST("/rtp_llm/notify_master", accept(MediaType.APPLICATION_JSON),
                        this::notifyParticipant)
                .POST("/rtp_llm/update_log_level", accept(MediaType.APPLICATION_JSON),
                        this::debugMode)
                .GET("/rtp_llm/queue_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::queueSnapshot)
                .GET("/rtp_llm/worker_profiles", accept(MediaType.APPLICATION_JSON),
                        this::workerProfiles)
                .GET("/rtp_llm/cp_composition", accept(MediaType.APPLICATION_JSON),
                        this::cpComposition)
                .POST("/rtp_llm/cp_composition/scheduled_switch", accept(MediaType.APPLICATION_JSON),
                        this::scheduledCpSwitch)
                .POST("/rtp_llm/cp_composition/reconcile", accept(MediaType.APPLICATION_JSON),
                        this::reconcileCpComposition)
                .POST("/rtp_llm/switch_profile", accept(MediaType.APPLICATION_JSON),
                        this::switchProfile)
                .build();
    }

    /**
     * Handles load balancing request scheduling.
     *
     * @param request the HTTP request containing the model inference request
     * @return a reactive response containing the load balancing result
     */
    public Mono<ServerResponse> scheduleRequest(ServerRequest request) {
        BalanceContext ctx = new BalanceContext();
        return request.bodyToMono(Request.class)
                .flatMap(req -> {
                    if (req.getRequestId() == 0) {
                        throw new IllegalArgumentException("requestId is 0");
                    }
                    ctx.setRequest(req);
                    return Mono.using(
                            activeRequestCounter::acquire,
                            ignored -> processScheduledRequest(ctx, req),
                            ActiveRequestCounter.RequestToken::close);
                })
                .onErrorResume(e -> handleRequestError(ctx, e))
                .doFinally(signal -> finalizeRequestContext(ctx));
    }

    private Mono<ServerResponse> processScheduledRequest(BalanceContext ctx, Request req) {
        engineHealthReporter.reportArriveDelayTime(ctx);

        if (lbStatusConsistencyService.isNeedConsistency() && !lbStatusConsistencyService.isMaster()) {
            return forwardRequestToMaster(ctx, req);
        }

        return routeService.route(ctx)
                .flatMap(response -> handleRoutingResult(ctx, response))
                .doOnCancel(() -> {
                    ctx.setSuccess(false);
                    ctx.setErrorMessage("REQUEST_CANCELLED");
                    routeService.cancel(ctx);
                });
    }

    private Mono<ServerResponse> debugMode(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(logLevelUpdateRequest -> {
                    Logger.setGlobalLogLevel(logLevelUpdateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just("Success! logLevel=" + Logger.getGlobalLogLevel()), String.class);
                }).onErrorResume(e -> {
                    Logger.error("update logLevel error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private Mono<ServerResponse> responseMasterInfo(ServerRequest request) {
        return request.bodyToMono(Request.class)
                .flatMap((Function<Request, Mono<ServerResponse>>) req -> {
                    Response result = new Response();
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    result.setQueueLength(queueManager.getQueue().size());
                    result.setCode(200);
                    result.setSuccess(true);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(result), Response.class);
                }).onErrorResume(e -> {
                    Logger.error("responseMasterInfo error", e);
                    Response errorResponse = new Response();
                    errorResponse.setSuccess(false);
                    errorResponse.setCode(500);
                    errorResponse.setErrorMessage(e.getMessage());
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(errorResponse), Response.class);
                });
    }

    public Mono<ServerResponse> notifyParticipant(ServerRequest request) {
        return request.bodyToMono(MasterChangeNotifyReq.class)
                .flatMap(masterChangeNotifyReq -> {
                    MasterChangeNotifyResp resp = lbStatusConsistencyService.handleMasterChange(masterChangeNotifyReq);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(resp), MasterChangeNotifyReq.class);
                }).onErrorResume((Function<Throwable, Mono<ServerResponse>>) e -> {
                    Logger.error("notifyParticipant error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    public Mono<ServerResponse> dumpLBStatus(ServerRequest request) {
        return request.bodyToMono(SyncLBStatusReq.class)
                .flatMap(syncLBStatusReq -> {
                    SyncLBStatusResp resp = lbStatusConsistencyService.dumpLBStatus();
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(resp), SyncLBStatusResp.class);
                }).onErrorResume(e -> {
                    Logger.error("dumpLBStatus error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    public Mono<ServerResponse> queueSnapshot(ServerRequest request) {
        try {
            QueueSnapshotResponse response = queueManager.snapshotQueue();
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(response), QueueSnapshotResponse.class);
        } catch (Exception e) {
            Logger.error("queueSnapshot error", e);
            return ServerResponse.status(500)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(e.getMessage()), String.class);
        }
    }

    public Mono<ServerResponse> workerProfiles(ServerRequest request) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(engineWorkerStatus.profileSnapshot());
    }

    public Mono<ServerResponse> cpComposition(ServerRequest request) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(cpCompositionController.snapshot());
    }

    public Mono<ServerResponse> scheduledCpSwitch(ServerRequest request) {
        String configuredToken = System.getenv("FLEXLB_CP_ORACLE_TOKEN");
        String authorization = request.headers().firstHeader("Authorization");
        boolean enabled = Boolean.parseBoolean(System.getenv("FLEXLB_CP_ORACLE_ENABLED"));
        boolean authorized = configuredToken != null && !configuredToken.isBlank()
                && ("Bearer " + configuredToken.trim()).equals(authorization);
        if (!enabled || !authorized) {
            return ServerResponse.status(403)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("ok", false,
                            "error", "scheduled CP switching is disabled or unauthorized"));
        }
        return request.bodyToMono(Map.class)
                .flatMap(body -> Mono.fromCallable(() -> {
                            Object rawHost = body.get("host");
                            Object rawTarget = body.get("target_cp");
                            if (rawHost == null || !(rawTarget instanceof Number)) {
                                throw new IllegalArgumentException("host and numeric target_cp are required");
                            }
                            return cpCompositionController.executeScheduledSwitch(
                                    rawHost.toString(), ((Number) rawTarget).intValue());
                        })
                        .subscribeOn(Schedulers.boundedElastic())
                        .flatMap(result -> Boolean.TRUE.equals(result.get("ok"))
                                ? ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                                        .bodyValue(result)
                                : ServerResponse.status(409).contentType(MediaType.APPLICATION_JSON)
                                        .bodyValue(result)))
                .onErrorResume(e -> {
                    Logger.error("scheduledCpSwitch error", e);
                    return ServerResponse.status(400)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(Map.of("ok", false,
                                    "error", String.valueOf(e.getMessage())));
                });
    }

    public Mono<ServerResponse> reconcileCpComposition(ServerRequest request) {
        return Mono.fromCallable(cpCompositionController::reconcileNow)
                .subscribeOn(Schedulers.boundedElastic())
                .flatMap(result -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(result))
                .onErrorResume(e -> {
                    Logger.error("reconcileCpComposition error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    public Mono<ServerResponse> switchProfile(ServerRequest request) {
        return request.bodyToMono(Map.class)
                .flatMap(body -> Mono.fromCallable(() -> {
                            Object profileName = body.get("profile_name");
                            if (profileName == null) {
                                profileName = body.get("profileName");
                            }
                            long timeoutMs = longValue(body.get("timeout_ms"), 600_000L);
                            long pollIntervalMs = longValue(body.get("poll_interval_ms"), 200L);
                            return engineWorkerStatus.switchActiveProfileAfterDrain(
                                    profileName == null ? null : profileName.toString(),
                                    timeoutMs,
                                    pollIntervalMs);
                        })
                        .subscribeOn(Schedulers.boundedElastic())
                        .flatMap(snapshot -> ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .bodyValue(snapshot)))
                .onErrorResume(e -> {
                    Logger.error("switchProfile error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private long longValue(Object value, long defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return Long.parseLong(value.toString());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /**
     * Forwards the request to the active master node.
     *
     * @param ctx the balance context
     * @param request the master request to forward
     * @return response from the master node
     */
    private Mono<ServerResponse> forwardRequestToMaster(BalanceContext ctx, Request request) {
        String master = lbStatusConsistencyService.getMasterHostIpPort();
        if (master == null) {
            Logger.error("Master unreachable, routing locally");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return fallbackToLocalRouting(ctx);
        }
        Logger.info("Forwarding request to master: {}, request: {}", master, request);
        URI uri = URI.create("http://" + master);
        return generalHttpNettyService.request(request, uri, "/rtp_llm/schedule", Response.class)
                .flatMap(resp -> {
                            engineHealthReporter.reportForwardToMasterResult(uri.getHost(), String.valueOf(resp.getCode()));
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .bodyValue(resp);
                        }
                )
                .onErrorResume(e -> {
                    String errorCode = e instanceof TimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
                    Logger.error("[Fallback] Master unreachable, routing locally: {}, errorCode: {}", e.getMessage(), errorCode);
                    engineHealthReporter.reportForwardToMasterResult("LOCAL", errorCode);
                    return fallbackToLocalRouting(ctx);
                });
    }

    private Mono<ServerResponse> fallbackToLocalRouting(BalanceContext ctx) {
        return routeService.route(ctx)
                .flatMap(response -> handleRoutingResult(ctx, response))
                .onErrorResume(e -> {
                    Logger.error("[Fallback] Local routing failed", e);
                    Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(errorResponse);
                });
    }

    /**
     * Processes the routing response and builds the appropriate HTTP response.
     *
     * @param ctx the balance context
     * @param response the routing response
     * @return HTTP response based on routing success or failure
     */
    private Mono<ServerResponse> handleRoutingResult(BalanceContext ctx, Response response) {

        response.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());

        if (response.isSuccess()) {
            return buildSuccessResponse(response);
        } else {
            Logger.error("Routing failed with error code: {}", response.getErrorMessage());
            ctx.setSuccess(false);
            ctx.setErrorMessage("error_code:" + response.getErrorMessage());
            return buildErrorResponse(response);
        }
    }

    /**
     * Builds a successful HTTP response.
     *
     * @param result the master response containing the result
     * @return successful HTTP response
     */
    private Mono<ServerResponse> buildSuccessResponse(Response result) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), Response.class);
    }

    /**
     * Builds an error HTTP response.
     *
     * @param result the master response containing the error
     * @return error HTTP response
     */
    private Mono<ServerResponse> buildErrorResponse(Response result) {
        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), Response.class);
    }

    /**
     * Handles global request errors.
     *
     * @param ctx the balance context
     * @param throwable the error that occurred
     * @return error response
     */
    private Mono<ServerResponse> handleRequestError(BalanceContext ctx, Throwable throwable) {
        Logger.error("Request processing error", throwable);
        ctx.setSuccess(false);
        ctx.setErrorMessage(throwable.getMessage());

        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(throwable.getMessage()), String.class);
    }

    /**
     * Finalizes the request context by reporting metrics.
     *
     * @param ctx the balance context to finalize
     */
    private void finalizeRequestContext(BalanceContext ctx) {
        engineHealthReporter.reportBalancingService(ctx);
        logPvRecord(ctx);
    }

    /**
     * Logs the PV record with appropriate log level based on success status.
     *
     * @param ctx the balance context containing PV log data
     */
    private void logPvRecord(BalanceContext ctx) {

        PvLogData pvLogData = new PvLogData(ctx);

        try {
            String jsonLog = JsonUtils.toStringOrEmpty(pvLogData);
            if (pvLogData.isSuccess()) {
                pvLogger.info(jsonLog);
            } else {
                pvLogger.error(jsonLog);
            }
        } catch (Exception ex) {
            Logger.error("Failed to serialize PV log data", ex);
        }
    }

}
