package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
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
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Worker selection based on a simple CP-aware three-term cost model.
 *
 * <p>Paper form: F_{i,w} = lambda_L * L_w + lambda_S * D_{i,w} + lambda_P * M_{i,w}
 * where L_w is the max-normalized queued attention work
 * (sum of R*P + R(R+1)/2 over remaining requests, divided by C_w),
 * D_{i,w} is the log-distance between the worker CP size and the ideal CP size
 * derived from the cache-adjusted equivalent length R~ (solution of
 * R~(R~+1)/2 = R*P + R(R+1)/2), and M_{i,w} = R/L is the prefix-cache miss
 * ratio. The legacy prefill-time estimate is populated only as response
 * metadata and never participates in selection.
 */
@Component("cpAwareTtftStrategy")
public class CpAwareTtftStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final CpCompositionController cpCompositionController;
    private static final long DEFAULT_BLOCK_SIZE_TOKENS = 64L;
    private static final long SCORE_SCALE = 1000L;
    /**
     * Resolution for the shared-denominator score. The denominator is the
     * busiest candidate's predicted completion, which reaches tens of seconds
     * under load; at {@link #SCORE_SCALE} one point would cover tens of
     * milliseconds and near-idle candidates would collapse into ties.
     */
    private static final long UNIFIED_SCORE_SCALE = 1_000_000L;
    /** Milliseconds per score point when the unified score is left in seconds. */
    private static final long SECONDS_SCORE_SCALE = 1000L;
    /**
     * Divide the unified score by the busiest candidate's predicted completion.
     * The denominator is shared by every candidate of a decision, so this only
     * rescales the score; it cannot reorder candidates. Leaving it off keeps
     * the score in seconds, matching the queue and execution-time inputs.
     */
    private static final boolean UNIFIED_NORMALIZE =
            "true".equalsIgnoreCase(readStringEnv("FLEXLB_CP_UNIFIED_NORMALIZE", "false"));
    /**
     * Weight of the CP-mismatch term in the unified score. It defaults to zero
     * because the effect it approximates is already priced: CP size enters both
     * the queue and the request's own cost through the sharded coefficients
     * {@code g_shard/C} and {@code alpha/C}. Being dimensionless, a non-zero
     * weight also competes with terms measured in seconds.
     */
    private static final double UNIFIED_LENGTH_WEIGHT =
            readDoubleEnv("FLEXLB_CP_UNIFIED_LENGTH_WEIGHT", 0.0);
    private static final double LOAD_WEIGHT = readDoubleEnv("FLEXLB_CP_SIMPLE_LOAD_WEIGHT", 1.0);
    private static final double LENGTH_WEIGHT = readDoubleEnv("FLEXLB_CP_SIMPLE_LENGTH_WEIGHT", 1.0);
    private static final double CACHE_WEIGHT = readDoubleEnv("FLEXLB_CP_SIMPLE_CACHE_WEIGHT", 1.0);
    private static final double TOKENS_PER_CP_UNIT = readDoubleEnv(
            "FLEXLB_CP_SIMPLE_TOKENS_PER_CP_UNIT", 1024.0);

    /**
     * Prefill latency model, in seconds:
     * <pre>
     *   T_C(R, P) = d0 + (g_shard/C + g_fix) * R~ + g_cache * P~ + (alpha/C) * W~
     *   R~ = R/1024,  P~ = P/1024,  W~ = (R*P + R(R+1)/2) / 1024^2
     * </pre>
     * R is the uncached prefill length, P the prefix-cache hit, C the CP size.
     * Only the sharded terms carry 1/C: attention work splits across ranks
     * (alpha/C) as does the parallel part of the MoE/linear cost (g_shard/C),
     * while the serial remainder (g_fix), the ALL_GATHER cache replay (g_cache)
     * and the fixed launch cost (d0) do not shard. One coefficient set covers
     * every CP size; C enters the model only as 1/C.
     *
     * Defaults fitted over 176 measured points (CP2/CP4/CP8 x 8 prefix lengths
     * x 10 sequence lengths) on Qwen3-30B-A3B: MAPE 2.53%, leave-one-prefix-out
     * cross-validated MAPE 2.89%.
     */
    private static final double LAT_D0 = readDoubleEnv("FLEXLB_CP_LAT_D0", 0.09632);
    private static final double LAT_G_SHARD = readDoubleEnv("FLEXLB_CP_LAT_G_SHARD", 0.02109);
    private static final double LAT_G_FIX = readDoubleEnv("FLEXLB_CP_LAT_G_FIX", 0.03382);
    private static final double LAT_G_CACHE = readDoubleEnv("FLEXLB_CP_LAT_G_CACHE", 0.00294);
    private static final double LAT_ALPHA = readDoubleEnv("FLEXLB_CP_LAT_ALPHA", 0.00587);

    /**
     * Saturation floor, in seconds: no prefill is predicted faster than this
     * however few tokens it carries. {@code 0} disables it, which reproduces
     * the pure polynomial above exactly, so the Qwen3-30B-A3B coefficients and
     * their 2.53% MAPE are unaffected by this term existing.
     *
     * <p>DeepSeek-V4-Flash needs it. Below roughly 4k tokens (CP4) or 6k (CP8)
     * its measured prefill stops responding to length at all and pins at
     * ~229 ms: at CP8, r=1024 and r=4096 both cost ~230 ms. Work is spread over
     * C ranks and then over 256 routed experts at 6 experts per token, so at
     * r=1024 and C=8 each expert sees fewer than 4 tokens and the grouped GEMM
     * is launch-bound, not FLOP-bound -- 43 layers of kernel launches cost the
     * same no matter how little data each carries. Above the knee the GPU is
     * FLOP-bound and the polynomial applies again.
     *
     * <p>Forcing a plain intercept to absorb that flat region is what makes the
     * unsaturated fit fail: it lands at MAPE 7.08% with a negative g_cache
     * (i.e. claiming a prefix-cache hit makes a request slower). With the floor
     * the same five terms fit 182 points across CP4/CP8, 11 lengths from 1k to
     * 48k and 12 partial-cache shapes at MAPE 2.16%, leave-one-shape-out CV
     * 2.37% -- at par with the Qwen model. One global scalar, not a per-CP
     * table: the CP4 and CP8 floors differ by 3%.
     *
     * <p>DeepSeek-V4-Flash coefficients (seconds):
     * FLEXLB_CP_LAT_FLOOR=0.229279 FLEXLB_CP_LAT_D0=0.082177
     * FLEXLB_CP_LAT_G_SHARD=0.166855 FLEXLB_CP_LAT_G_FIX=0.0119768
     * FLEXLB_CP_LAT_G_CACHE=0.000124114 FLEXLB_CP_LAT_ALPHA=0.000615125
     *
     * <p>alpha comes out 9.5x smaller than Qwen's, which is the predicted
     * signature of V4's sparse attention: index_topk=512 plus a 128-token
     * sliding window caps the keys any query attends, so the attention matmuls
     * go linear in r, but the indexer still scores every context key for every
     * query and that scoring keeps the full causal r*p + r(r+1)/2 shape. The
     * quadratic term survives, relocated from attention to the indexer, with a
     * much smaller coefficient because the indexer is one head of dim 128
     * against 64 heads of dim 576.
     */
    private static final double LAT_FLOOR = readDoubleEnv("FLEXLB_CP_LAT_FLOOR", 0.0);

    /**
     * Selects how a candidate is priced.
     * <ul>
     *   <li>{@code LEGACY} — normalised three-term score, queue measured as sharded attention work.</li>
     *   <li>{@code HYBRID} — same three-term score, queue measured in predicted seconds.</li>
     *   <li>{@code UNIFIED} — three-term score whose load and cache terms share one denominator.</li>
     *   <li>{@code GPU_SECONDS} — continuous queue, execution-time and excess GPU-seconds cost.</li>
     * </ul>
     */
    private static final String COST_MODEL = readStringEnv("FLEXLB_CP_COST_MODEL", "LEGACY");
    private static final boolean ADDITIONAL_GPU_SECONDS = Boolean.parseBoolean(
            readStringEnv("FLEXLB_CP_ADDITIONAL_GPU_SECONDS", "false"));
    private static final boolean QUEUE_IN_SECONDS =
            "HYBRID".equalsIgnoreCase(COST_MODEL) || "GPU_SECONDS".equalsIgnoreCase(COST_MODEL);

    /**
     * Whether the fixed per-forward-pass term of the latency model is charged
     * once for a worker's whole backlog rather than once per queued request.
     * Queued prefills are batched, so the backlog pays the constant once.
     */
    private static final boolean QUEUE_D0_ONCE =
            !"false".equalsIgnoreCase(readStringEnv("FLEXLB_CP_QUEUE_D0_ONCE", "true"));
    /** Resolution and upper bound for the continuous cost score. */
    private static final long COST_SCORE_SCALE = 100_000L;
    private static final double COST_SECONDS_CAP = 999.0;
    /**
     * How close two candidates' costs have to be before the cost stops deciding
     * between them and least-recently-used does.
     *
     * <p>Worker queue depths arrive by periodic sync, so within one interval every
     * request sees the same snapshot. A cost gap narrower than the work that can
     * arrive unobserved in that interval is not information, and an argmin over it
     * sends a whole burst to whichever worker happened to look best in the stale
     * snapshot. Rounding the comparison to this quantum and breaking the resulting
     * tie by {@code lastSelectedTime} spreads the burst instead.
     *
     * <p>Rotation is confined to candidates of the same parallel degree, so it
     * decides which worker inside a pool serves the request and never which pool.
     */
    private static final double TIE_QUANTUM_COST =
            readDoubleEnv("FLEXLB_CP_TIE_QUANTUM", 0.0);
    private static final long TIE_QUANTUM_POINTS =
            Math.round(COST_SCORE_SCALE * Math.max(0.0, TIE_QUANTUM_COST));
    /**
     * What a second of this request's queueing is worth in gpu-seconds of
     * system capacity. The two terms of the cost pull against each other:
     * gpu-seconds shrink with cache reuse and with a smaller parallel degree,
     * queueing shrinks with an emptier worker. Large values reduce the cost to
     * shortest-queue, zero reduces it to pure cache affinity.
     */
    private static final double QUEUE_PRICE = readDoubleEnv("FLEXLB_CP_QUEUE_PRICE", 1.0);
    /**
     * What one gpu-second of system capacity is worth in seconds of this
     * request's own completion time, at an idle fleet.
     * <p>
     * The cost is written in seconds, so the capacity a request occupies has to
     * be converted into seconds to enter it, and this is the exchange rate. On a
     * fleet of one parallel degree it changes no ordering that is not already
     * there: {@code cpSize} is then a common factor and the term is a rescaling
     * of {@code own}. It exists for mixed degrees, where the two halves of the
     * cost disagree. A request occupies strictly more gpu-seconds at every larger
     * degree, at every length, so a positive price is the only thing that ever
     * holds a request back from the largest degree available, and the size of the
     * price is what decides at which length it stops holding it back.
     * <p>
     * The price at which the two degrees of a pair tie has a closed form. Writing
     * {@code u} for the uncached length and {@code p} for the prefix a candidate
     * holds, the seconds a larger degree saves are
     * {@code lat_lo - lat_hi = g_shard*(1/C_lo - 1/C_hi)*u_hat + alpha*(1/C_lo - 1/C_hi)*W_hat},
     * while the gpu-seconds it costs are {@code C_hi*lat_hi - C_lo*lat_lo}, in
     * which the quadratic term cancels exactly and leaves
     * {@code (C_hi - C_lo)*(d0 + g_fix*u_hat + g_cache*p_hat)}. The tie price is
     * the first over the second, quadratic over linear, so it rises with length:
     * one price therefore names one switch length, and the degree chosen is
     * non-decreasing in length rather than alternating with it. The same ratio
     * falls with the prefix a candidate holds, because a hit shrinks {@code u} and
     * the quadratic term with it, so a well-cached long request stays on the
     * smaller degree while an uncached one of the same length does not, without
     * either being a special case. For the profiled DSV4 cp4/cp8 pair the ratio is
     * 0.162 at 4k uncached tokens, 0.238 at 8k and 0.318 at 17k.
     */
    private static final double CAPACITY_PRICE = readDoubleEnv("FLEXLB_CP_CAPACITY_PRICE", 0.0);
    /** Ceiling on the backlog-adjusted capacity price, in the same units. */
    private static final double CAPACITY_PRICE_CEILING =
            readDoubleEnv("FLEXLB_CP_CAPACITY_PRICE_CEILING", 0.318);
    /**
     * How much the measured backlog raises the price of a gpu-second.
     * <p>
     * A gpu-second is only worth something when something else wants it. On an
     * idle fleet the capacity a larger degree consumes would otherwise have gone
     * unused, so buying latency with it is very nearly free and the request should
     * take it; on a full fleet the same gpu-seconds come out of requests that are
     * already waiting, and spending them makes the queue that is the actual
     * problem longer. Pricing capacity at a constant gets one of those two ends
     * wrong whichever constant is chosen. The backlog the candidates report is
     * already an estimate of how contended the capacity is, in seconds, so the
     * price is scaled by it, and the mixture of degrees the fleet is asked for
     * moves with load without anything in the rule being retuned: on the profiled
     * DSV4 trace the share of gpu-seconds demanded on cp8 falls from 72.5% at a
     * quarter of the measured load to 67.2% at eight times it, and further with a
     * higher base price. Zero recovers a constant price.
     */
    private static final double CAPACITY_PRICE_LOAD_SLOPE =
            readDoubleEnv("FLEXLB_CP_CAPACITY_PRICE_LOAD_SLOPE", 1.0);
    /**
     * Price of the parallel degree a candidate holds in excess of the cheapest
     * degree the fleet offers, in units of the request's own prefill time.
     * <p>
     * The homogeneous rule that was validated is {@code queue + OWN_WEIGHT*own},
     * and reproducing it exactly on the cheapest degree is what fixes the form of
     * the degree term: it has to be charged on the excess degree rather than on
     * the degree itself, because a term in {@code cpSize} charges the cheapest
     * degree too and reweights queue against own as a side effect. Writing the
     * excess as a ratio rather than a difference,
     * <pre>
     *   cost_w = queue_w + OWN_WEIGHT * own_w * (1 + mu * (C_w/C_min - 1))
     * </pre>
     * the cheapest degree is literally the validated rule, and the whole degree
     * choice between a pair of degrees collapses to one scalar test: at equal
     * queue a candidate at {@code C_hi} is preferred over one at {@code C_min}
     * exactly when {@code own_hi/own_lo < 1 / (1 + mu*(C_hi/C_min - 1))}. For a
     * doubling of degree that is {@code K = 1/(1+mu)}, so one scalar names the
     * escalation rate directly and names the same rate for the cp4/cp8 pair and
     * for the cp2/cp4 pair, which are the two fleets in use.
     * <p>
     * {@code own_hi/own_lo} falls with the work a request actually carries -- it
     * tends to {@code C_min/C_hi} once the quadratic attention term dominates and
     * to 1 at the floor where neither degree helps -- so {@code K} lies in
     * {@code (C_min/C_hi, 1]} and the degree chosen is non-decreasing in uncached
     * length at every {@code mu}, with no ceiling needed to keep it monotone. The
     * ratio also falls with the prefix a candidate holds, so a well-cached long
     * request stays on the cheap degree while an uncached one of the same length
     * does not.
     * <p>
     * Zero leaves the older absolute form in {@link #CAPACITY_PRICE} in charge.
     */
    private static final double ESCALATION_MU = readDoubleEnv("FLEXLB_CP_ESCALATION_MU", 0.0);
    /**
     * Ceiling on {@link #ESCALATION_MU}, and the one number in the rule that is
     * conservation rather than preference.
     * <p>
     * Per-pool gpu-second demand scales as the reciprocal of the measurement
     * window, so the {@code mu} at which the two pools carry equal utilisation is
     * the same at every offered load -- only the level moves. Above that value the
     * fleet is being asked for more capacity on the expensive pool than on the
     * cheap one, which no amount of queueing can undo, and the expensive pool is
     * also the smaller one and so the one that queues worse at equal utilisation.
     * The default is that balancing value for the profiled DSV4 cp4/cp8 trace,
     * {@code mu = 0.5728}, i.e. {@code K = 0.6358}.
     */
    private static final double ESCALATION_MU_CEILING =
            readDoubleEnv("FLEXLB_CP_ESCALATION_MU_CEILING", 0.5728);
    /**
     * Backlog on the wider-degree pool, in seconds, at which its gpu-seconds are
     * priced at {@link #ESCALATION_MU_CEILING}. Set it to one escalated service
     * time and the price is full once that pool is a request deep. Zero prices
     * escalation off the fleet-wide backlog instead.
     */
    private static final double ESCALATION_Q_REFERENCE =
            readDoubleEnv("FLEXLB_CP_ESCALATION_QREF", 0.0);
    /**
     * What a second of the request's own prefill is worth against a second of its
     * queueing. Four is the ratio the homogeneous gate validated on both models.
     */
    private static final double OWN_WEIGHT = readDoubleEnv("FLEXLB_CP_OWN_WEIGHT", 4.0);
    /**
     * What a gpu-second is worth in seconds of the latency it buys, at the point
     * where the fleet's backlog reaches {@link #GPU_SECOND_PRICE_Q_REFERENCE}.
     *
     * <p>A wider degree buys wall time with gpu-seconds and the exchange rate is
     * measurable: it is {@code C*own}, and for this model a request served at
     * twice the degree costs between 1.20x and 2.00x what the narrow degree would
     * have spent, the 2.00x falling on requests short enough that both degrees
     * return the same own. So the term to add is the excess gpu-seconds the choice
     * actually spends, priced by what a gpu-second is worth:
     *
     * <pre>
     *   cost_w = queue_w + OWN_WEIGHT * own_w + lambda * (C_w * own_w - C_min * own_w_at_C_min)
     * </pre>
     *
     * <p>The bracket is zero on the cheapest degree by construction, so on a
     * uniform fleet this is the validated homogeneous rule unchanged. It is
     * charged against the same request's cost at the cheapest degree and at the
     * same prefix hit, so a cache hit never pays the degree price. Unlike a price
     * on the degree ratio, this one needs no length input to order requests
     * correctly: at equal queue the wider degree wins exactly when
     * {@code (own_min - own) / (C/C_min * own - own_min) > lambda}, a quantity
     * that rises monotonically with length from zero below the execution-time
     * floor -- where a wider degree returns the same own and is charged the whole
     * of the narrow degree's bill for it -- to about 2 at the longest prompts. One
     * scalar therefore fixes one escalation length, and the map stays monotone.
     */
    private static final double GPU_SECOND_PRICE_CEILING =
            readDoubleEnv("FLEXLB_CP_GS_LAMBDA_MAX", 0.0);
    /**
     * Price of a gpu-second on an idle fleet. Zero is the honest value: capacity
     * that would have gone unused costs the cluster nothing, so at zero backlog
     * every degree is free and the rule reduces to picking the shortest completion
     * -- which pools the two degrees into one queue while there is slack, and lets
     * a short request take an idle wide worker even though it gains nothing there.
     */
    private static final double GPU_SECOND_PRICE_FLOOR =
            readDoubleEnv("FLEXLB_CP_GS_LAMBDA_MIN", 0.0);
    /**
     * Mean candidate backlog, in seconds, at which a gpu-second reaches
     * {@link #GPU_SECOND_PRICE_CEILING}. Below it the price ramps linearly from
     * {@link #GPU_SECOND_PRICE_FLOOR}, so the fleet gives its spare capacity away
     * cheaply and withdraws it as the queues fill.
     */
    private static final double GPU_SECOND_PRICE_Q_REFERENCE =
            readDoubleEnv("FLEXLB_CP_GS_LAMBDA_QREF", 1.0);
    /**
     * Weight on the delay a placement inflicts on the requests behind it.
     *
     * <p>Every other term prices what the request itself pays: its own wait, its own
     * execution, its own excess gpu-seconds. On a shared queue that is only half the
     * bill. A request placed on worker w also delays every request that arrives at w
     * before the queue drains, and in a first-come server it delays each of them by
     * exactly {@code own_w}. With arrivals to w at {@code lam_w} per second and a
     * drain time of about {@code queue_w + own_w}, that is
     *
     * <pre>
     *   social_w = own_w * lam_w * (queue_w + own_w)
     * </pre>
     *
     * <p>seconds of latency imposed elsewhere, in the same unit as the rest of the
     * cost, so the weight on it is dimensionless and {@code 1.0} is the exact
     * marginal social cost rather than a tuned constant. Values below one are the
     * honest setting for an engine that batches prefill, since a long prompt sharing
     * a batch with a short one does not block it for the whole of its own execution.
     *
     * <p>What it changes is the shape of the degree choice. The gpu-second term is
     * linear in {@code own}, so doubling the degree repays a request at most the
     * fraction of {@code own} the extra shards remove. This term is quadratic in
     * {@code own}, so halving a long request's execution quarters what it costs
     * everyone else, and the wider degree wins by a margin that grows with length
     * without any length appearing in the rule. It is also what stops the price from
     * spending the narrow pool on requests that do not belong there: a long request
     * whose prefix is already cached on a narrow worker has a small {@code own} there
     * and is still welcome, while the same length with nothing cached is not, and the
     * two are separated by their execution time rather than by a threshold on either
     * length or hit.
     */
    private static final double SOCIAL_NU = readDoubleEnv("FLEXLB_CP_SOCIAL_NU", 0.0);
    /**
     * Seconds of history in the per-worker arrival-rate estimate that
     * {@link #SOCIAL_NU} multiplies. One selection contributes
     * {@code 1/SOCIAL_RATE_WINDOW} to the rate and the rest decays at that
     * timescale, so the estimate follows a rate change within a window and does not
     * carry a burst from a minute ago into the current price.
     */
    private static final double SOCIAL_RATE_WINDOW =
            readDoubleEnv("FLEXLB_CP_SOCIAL_RATE_WINDOW_SECONDS", 5.0);
    /**
     * Per-worker arrival rate in requests per second, keyed by ip:port:dpRank.
     *
     * <p>Only the selected worker is updated, so what the map holds is the rate at
     * which each worker is actually being handed work -- which is the {@code lam_w}
     * the social cost needs, and is not recoverable from the worker's reported queue.
     */
    private static final Map<String, double[]> SOCIAL_RATE = new ConcurrentHashMap<>();

    /** Identity of one worker rank, which is the unit work is queued against. */
    private static String socialKey(WorkerStatus workerStatus, int dpRank) {
        return workerStatus.getIp() + ":" + workerStatus.getPort() + ":" + dpRank;
    }

    /** Requests per second currently being routed to one worker rank. */
    private static double socialRate(String key) {
        double[] state = SOCIAL_RATE.get(key);
        return state == null ? 0.0 : state[0];
    }

    /** Fold one selection of {@code key} at {@code nowMs} into its arrival rate. */
    private static void noteSocialArrival(String key, long nowMs) {
        SOCIAL_RATE.compute(key, (k, state) -> {
            if (state == null) {
                return new double[]{1.0 / SOCIAL_RATE_WINDOW, nowMs};
            }
            double gap = Math.max(0.0, (nowMs - state[1]) / 1000.0);
            double decay = Math.exp(-gap / SOCIAL_RATE_WINDOW);
            state[0] = state[0] * decay + 1.0 / SOCIAL_RATE_WINDOW;
            state[1] = nowMs;
            return state;
        });
    }

    @Autowired
    public CpAwareTtftStrategy(EngineWorkerStatus engineWorkerStatus,
                               EngineHealthReporter engineHealthReporter,
                               CacheAwareService cacheAwareService,
                               ResourceMeasureFactory resourceMeasureFactory,
                               CpCompositionController cpCompositionController) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        var observer = cacheAwareService.prefixObserver();
        if (observer != null && observer.settings().observe()
                && (!"GPU_SECONDS".equalsIgnoreCase(COST_MODEL)
                    || !(ADDITIONAL_GPU_SECONDS || GPU_SECOND_PRICE_CEILING > 0.0))) {
            throw new IllegalArgumentException("prefix observe requires GPU_SECONDS additional-GPU-seconds branch");
        }
        Logger.info("CP_AWARE_EFFECTIVE_CONFIG {}", effectiveConfiguration());
        this.cpCompositionController = cpCompositionController;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.CP_AWARE_TTFT, this);
    }

    /** Kept for focused strategy tests that do not start the composition controller. */
    public CpAwareTtftStrategy(EngineWorkerStatus engineWorkerStatus,
                        EngineHealthReporter engineHealthReporter,
                        CacheAwareService cacheAwareService,
                        ResourceMeasureFactory resourceMeasureFactory) {
        this(engineWorkerStatus, engineHealthReporter, cacheAwareService,
                resourceMeasureFactory, null);
    }

    private static double readDoubleEnv(String envName, double defaultValue) {
        String raw = System.getenv(envName);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            Logger.warn("Invalid {}={}, fallback to {}", envName, raw, defaultValue);
            return defaultValue;
        }
    }

    private static String readStringEnv(String envName, String defaultValue) {
        String raw = System.getenv(envName);
        return raw == null || raw.isBlank() ? defaultValue : raw.trim();
    }

    private final Object selectLock = new Object();
    private final List<CandidateCostBreakdown> candidateCosts = new ArrayList<>();
    private Map<String, org.flexlb.cache.prefix.ConfirmedCacheDirectory.Snapshot> frozenInventory = Map.of();
    private Map<String, Integer> frozenConfirmedMatches = Map.of();
    private RouteObservationInput routeObservation;
    private record RouteObservationInput(String attempt, long requestId, long inputTokens, int blockSize,
            List<Long> keys, int preferredDegree, String selectedLane,
            Map<String, org.flexlb.cache.prefix.ConfirmedCacheDirectory.Snapshot> inventory) { }

    public record CandidateCostBreakdown(String lane, int degree, long inputTokens, long reuseTokens,
                                          double queueSeconds, double serviceSeconds,
                                          double nonQueueCost, double totalCost, double frozenLambda) { }

    public static Map<String, Object> effectiveConfiguration() {
        Map<String, Object> values = new java.util.TreeMap<>();
        for (var field : CpAwareTtftStrategy.class.getDeclaredFields()) {
            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())
                    && java.lang.reflect.Modifier.isFinal(field.getModifiers())
                    && (field.getType().isPrimitive() || field.getType() == String.class)) {
                try {
                    values.put(field.getName(), field.get(null));
                } catch (IllegalAccessException exception) {
                    throw new IllegalStateException(exception);
                }
            }
        }
        values.put("strategy", "CP_AWARE_TTFT");
        values.put("gpu_seconds_branch", ADDITIONAL_GPU_SECONDS || GPU_SECOND_PRICE_CEILING > 0.0
                ? "additional" : ESCALATION_MU > 0.0 ? "escalation" : "capacity");
        return java.util.Collections.unmodifiableMap(values);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        var routingObserver = cacheAwareService.prefixObserver();
        try (var permit = routingObserver == null || !routingObserver.settings().confirmed()
                ? null : routingObserver.directory().acquireRouting()) {
            ServerStatus result;
            RouteObservationInput observation;
            synchronized (selectLock) {
                routeObservation = null;
                result = doSelect(balanceContext, roleType, group);
                observation = routeObservation;
            }
            if (observation != null && result.isSuccess()) {
                var observer = cacheAwareService.prefixObserver();
                try {
                    observer.observeAndAccept(observation.attempt(), observation.requestId(),
                            observation.inputTokens(), observation.blockSize(), observation.keys(),
                            observation.preferredDegree(), prefillCostModel(),
                            observation.selectedLane(), observation.inventory());
                } catch (RuntimeException exception) {
                    Logger.warn("prefix observation rejected", exception);
                }
            }
            return result;
        } catch (Exception e) {
            Logger.warn("CP-aware TTFT selection failed", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        if (cacheAwareService.prefixObserver() != null) {
            cacheAwareService.prefixObserver().cancel(requestId);
        }
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectAllProfileModelWorkerStatus(RoleType.PREFILL, null);
        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        candidateCosts.clear();
        var observer = cacheAwareService.prefixObserver();
        var inventoryView = observer == null ? Map.<String, org.flexlb.cache.prefix.ConfirmedCacheDirectory.Snapshot>of()
                : observer.directory().view();
        frozenInventory = inventoryView;
        frozenConfirmedMatches = Map.of();

        FlexlbConfig config = balanceContext.getConfig();
        List<WorkerStatus> availableWorkers = getAvailableWorkers(
                roleType, group, config.getResourceMeasureIndicator(roleType), requestId);
        if (CollectionUtils.isEmpty(availableWorkers)) {
            Logger.warn("No available workers for role: {}", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(
                balanceContext, roleType, availableWorkers, requestId);
        if (observer != null && observer.settings().confirmed()) {
            if (!observer.settings().namespace().equals(balanceContext.getRequest().getPrefixCacheNamespace())) {
                cacheMatchResults = Map.of();
            } else {
                var candidateRoutes = availableWorkers.stream()
                        .map(worker -> worker.getDpRankCacheKey(requestId)).toList();
                frozenConfirmedMatches = observer.directory().measuredMatches(
                        inventoryView, balanceContext.getRequest().getBlockCacheKeys(), candidateRoutes);
                if (observer.settings().mainCompatibleRouting()) {
                    var combined = new HashMap<>(cacheMatchResults);
                    frozenConfirmedMatches.forEach((route, blocks) -> combined.merge(route, blocks, Math::max));
                    cacheMatchResults = combined;
                } else {
                    cacheMatchResults = observer.routingMatches(
                            inventoryView, balanceContext.getRequest().getBlockCacheKeys(), candidateRoutes);
                }
            }
        }

        List<ScoredWorker> scoredWorkers = scoreWorkers(
                availableWorkers, cacheMatchResults, seqLen, requestId);

        ScoredWorker bestWorker = selectBestWorker(scoredWorkers);
        if (bestWorker == null) {
            availableWorkers.forEach(worker -> worker.unpinDpRank(requestId));
            Logger.warn("CP-aware TTFT: failed to find best worker for role: {}", roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        availableWorkers.stream()
                .filter(worker -> worker != bestWorker.worker())
                .forEach(worker -> worker.unpinDpRank(requestId));

        if (observer != null && observer.settings().observe() && roleType != RoleType.DECODE
                && observer.settings().namespace().equals(balanceContext.getRequest().getPrefixCacheNamespace())) {
            var preferred = candidateCosts.stream().min(java.util.Comparator
                    .comparingDouble(CandidateCostBreakdown::nonQueueCost)
                    .thenComparingInt(CandidateCostBreakdown::degree)
                    .thenComparing(CandidateCostBreakdown::lane)).orElse(null);
            var selectedCache = inventoryView.get(bestWorker.worker().getDpRankCacheKey(requestId));
            if (preferred != null && selectedCache != null) {
                routeObservation = new RouteObservationInput(balanceContext.getRequest().getPrefixCacheAttempt(),
                        requestId, seqLen, selectedCache.report().blockSize(),
                        List.copyOf(balanceContext.getRequest().getBlockCacheKeys()), preferred.degree(),
                        bestWorker.worker().getDpRankCacheKey(requestId), inventoryView);
            }
        }
        ServerStatus result = finalizeWorkerSelection(bestWorker, balanceContext, roleType, requestId, seqLen);
        return result;
    }

    private List<WorkerStatus> getAvailableWorkers(
            RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator,
            long requestId) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectAllProfileModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            return new ArrayList<>();
        }

        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            Logger.warn("No ResourceMeasure registered for indicator: {}", indicator);
            return new ArrayList<>();
        }

        List<WorkerStatus> available = new ArrayList<>();
        for (WorkerStatus worker : workerStatusMap.values()) {
            if (worker == null || !worker.isAlive()) {
                continue;
            }

            // Resource availability is rank-local for a DP worker. Pin the
            // least-loaded live lane first; otherwise requestId % dpSize can
            // reject the entire physical worker just because that arbitrary
            // lane is busy while another lane is idle.
            worker.selectAndPinDpRank(requestId);
            var observer = cacheAwareService.prefixObserver();
            boolean topologyAvailable = observer == null || !observer.settings().confirmed()
                    || observer.directory().routingAvailable(worker.getIpPort(), worker.getDpRankCacheKey(requestId),
                            worker.getCpSize(), Math.max(1L, worker.getDpSize()));
            if (topologyAvailable && resourceMeasure.isResourceAvailable(worker, requestId)) {
                available.add(worker);
            } else {
                worker.unpinDpRank(requestId);
            }
        }
        return available;
    }

    private Map<String, Integer> getCacheMatchResults(BalanceContext balanceContext,
                                                           RoleType roleType,
                                                           List<WorkerStatus> candidateWorkers,
                                                           long requestId) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        List<String> candidateIpPorts = candidateWorkers.stream()
                .map(worker -> worker.getDpRankCacheKey(requestId))
                .filter(ipPort -> ipPort != null && !ipPort.isBlank())
                .toList();
        Map<String, Integer> fleetResults = cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, candidateIpPorts);
        return fleetResults != null ? fleetResults : Map.of();
    }

    private List<ScoredWorker> scoreWorkers(List<WorkerStatus> workers,
                                            Map<String, Integer> cacheMatchResults,
                                            long seqLen,
                                            long requestId) {
        List<CandidateEstimate> estimates = workers.stream()
                .filter(WorkerStatus::isAlive)
                .map(workerStatus -> buildCandidateEstimate(
                        workerStatus, cacheMatchResults, seqLen, requestId))
                .collect(Collectors.toList());
        if (estimates.isEmpty()) {
            return List.of();
        }

        if ("GPU_SECONDS".equalsIgnoreCase(COST_MODEL)) {
            return scoreByGpuSeconds(estimates, seqLen);
        }
        if ("UNIFIED".equalsIgnoreCase(COST_MODEL)) {
            return scoreBySharedDenominator(estimates, seqLen);
        }

        Map<WorkerStatus, Double> queuedWorkByWorker = estimates.stream()
                .collect(Collectors.toMap(
                        CandidateEstimate::worker,
                        estimate -> QUEUE_IN_SECONDS
                                ? calculateQueueSeconds(estimate.worker(), estimate.dpRank())
                                : calculateQueuedAttentionWork(estimate.worker(), estimate.dpRank())));
        double maxQueuedWork = queuedWorkByWorker.values().stream()
                .mapToDouble(Double::doubleValue)
                .max()
                .orElse(0.0);
        long minCpSize = estimates.stream()
                .mapToLong(estimate -> Math.max(1L, estimate.worker().getCpSize()))
                .min()
                .orElse(1L);
        long maxCpSize = estimates.stream()
                .mapToLong(estimate -> Math.max(1L, estimate.worker().getCpSize()))
                .max()
                .orElse(minCpSize);

        return estimates.stream()
                .map(estimate -> {
                    WorkerStatus workerStatus = estimate.worker();
                    double queuedWork = queuedWorkByWorker.getOrDefault(workerStatus, 0.0);
                    double loadCost = maxQueuedWork > 0.0 ? queuedWork / maxQueuedWork : 0.0;
                    double lengthCost = calculateLengthCpMismatch(
                            seqLen, estimate.hitCacheTokens(),
                            workerStatus.getCpSize(), minCpSize, maxCpSize);
                    double prefixCacheCost = calculatePrefixCacheMissRatio(
                            seqLen, estimate.hitCacheTokens());
                    long score = Math.max(0L, Math.round(SCORE_SCALE * (
                            LOAD_WEIGHT * loadCost
                                    + LENGTH_WEIGHT * lengthCost
                                    + CACHE_WEIGHT * prefixCacheCost)));
                    long lastSelectedTime = workerStatus.getLastSelectedTime().get();
                    Logger.info("CP-aware simple cost - ip: {}, port: {}, profile: {}, cpSize: {}, dpRank: {}, hit: {}, metadataPrefillMs: {}, metadataQueueMs: {}, score: {}, loadCost: {}, lengthCost: {}, prefixCacheCost: {}, queuedWork: {}",
                            workerStatus.getIp(), workerStatus.getPort(), workerStatus.getProfileKey(),
                            workerStatus.getCpSize(), estimate.dpRank(), estimate.hitCacheTokens(),
                            estimate.requestMs(), estimate.queueMs(), score,
                            loadCost, lengthCost, prefixCacheCost, queuedWork);
                    return new ScoredWorker(workerStatus, estimate.ttft(), score, estimate.hitCacheTokens(), lastSelectedTime);
                })
                .collect(Collectors.toList());
    }

    private CandidateEstimate buildCandidateEstimate(WorkerStatus workerStatus,
                                                     Map<String, Integer> cacheMatchResults,
                                                     long seqLen,
                                                     long requestId) {
        int dpRank = workerStatus.resolveDpRank(requestId);
        long hitCacheTokens = calculatePrefixMatchLength(
                workerStatus, cacheMatchResults, requestId);
        var observer = cacheAwareService.prefixObserver();
        if (observer != null && observer.settings().confirmed() && !observer.settings().mainCompatibleRouting()) {
            var snapshot = frozenInventory.get(workerStatus.getDpRankCacheKey(requestId));
            if (snapshot == null) {
                hitCacheTokens = 0;
            } else {
                long blockSize = snapshot.report().blockSize();
                hitCacheTokens = Math.min(blockSize * cacheMatchResults.getOrDefault(
                        workerStatus.getDpRankCacheKey(requestId), 0),
                        org.flexlb.cache.prefix.PrefillCostModel.maxReusableTokens(seqLen, blockSize));
            }
        }
        long requestMs = TaskInfo.estimatePrefillTimeMs(seqLen, hitCacheTokens);
        long queueMs = estimateLegacyQueueMs(workerStatus, dpRank);
        return new CandidateEstimate(workerStatus, dpRank, requestMs, queueMs,
                safeAdd(requestMs, queueMs), hitCacheTokens);
    }

    /**
     * Continuous CP-aware placement cost:
     * <pre>
     *   own_w   = predicted prefill seconds at w, including w's prefix hit
     *   queue_w = predicted remaining prefill work already assigned to w
     *   deltaG  = C_w * own_w - C_min * own_at_C_min
     *   cost_w  = queue_w + OWN_WEIGHT * own_w + lambda(queue_wide) * deltaG
     * </pre>
     * The cost contains no latency target or request-length bucket. Cache locality
     * enters through {@code own_w}; CP efficiency enters through {@code deltaG}; and
     * the wide-pool backlog raises the price of its scarce GPU-seconds.
     */
    private List<ScoredWorker> scoreByGpuSeconds(List<CandidateEstimate> estimates, long seqLen) {
        int n = estimates.size();
        double[] own = new double[n];
        double[] queue = new double[n];
        double[] completion = new double[n];
        long[] cp = new long[n];
        long[] hits = new long[n];
        long cheapestCp = Long.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            CandidateEstimate estimate = estimates.get(i);
            cp[i] = Math.max(1L, estimate.worker().getCpSize());
            hits[i] = Math.max(0L, Math.min(estimate.hitCacheTokens(), seqLen));
            own[i] = prefillSeconds(cp[i], seqLen - hits[i], hits[i]);
            queue[i] = calculateQueueSeconds(estimate.worker(), estimate.dpRank());
            completion[i] = queue[i] + own[i];
            cheapestCp = Math.min(cheapestCp, cp[i]);
        }

        // What this request would have cost the fleet at the narrowest degree on
        // offer, at the prefix hit this candidate would give it. Subtracting it
        // from a candidate's own gpu-seconds leaves what the degree choice adds,
        // which is zero for the narrowest degree and never negative.
        double[] cheapGpuSeconds = new double[n];
        for (int i = 0; i < n; i++) {
            cheapGpuSeconds[i] = cheapestCp * (cp[i] == cheapestCp
                    ? own[i]
                    : prefillSeconds(cheapestCp, seqLen - hits[i], hits[i]));
        }

        double queueSum = 0.0;
        double highQueueSum = 0.0;
        int highCount = 0;
        for (int i = 0; i < n; i++) {
            queueSum += queue[i];
            if (cp[i] > cheapestCp) {
                highQueueSum += queue[i];
                highCount++;
            }
        }
        double backlog = queueSum / n;
        double highBacklog = highCount > 0 ? highQueueSum / highCount : 0.0;
        double lambda = Math.min(CAPACITY_PRICE_CEILING,
                CAPACITY_PRICE * (1.0 + CAPACITY_PRICE_LOAD_SLOPE * backlog));
        // What a gpu-second of the wider-degree pool is worth. A gpu-second that
        // pool would otherwise have spent idle costs the cluster nothing, so the
        // price starts at zero and rises with that pool's own backlog rather than
        // with the fleet's: below the execution-time floor a request gains nothing
        // from the wider degree, and pricing degree instead of scarcity locks it
        // out of an idle pool anyway, which splits one queue into two that cannot
        // lend to each other. With the price on scarcity the split closes when
        // there is slack and reappears when there is not, and the ordering falls
        // out of the cost: a request pays OWN_WEIGHT * mu * own to escalate and is
        // repaid OWN_WEIGHT * (own_cheap - own) in service time, so the ones with
        // nothing to gain are the first to be priced out.
        double mu = ESCALATION_Q_REFERENCE > 0.0
                ? ESCALATION_MU + (ESCALATION_MU_CEILING - ESCALATION_MU)
                        * Math.min(1.0, highBacklog / ESCALATION_Q_REFERENCE)
                : Math.min(ESCALATION_MU_CEILING,
                        ESCALATION_MU * (1.0 + CAPACITY_PRICE_LOAD_SLOPE * backlog));
        // What a gpu-second is worth in seconds of latency. Read off the WIDE pool's own
        // backlog, not the fleet's: the only thing this price is asked to ration is the
        // wide pool, since the excess it multiplies is zero everywhere else. Reading the
        // fleet average makes the price rise with congestion wherever it sits, so a cheap
        // pool at its limit next to an idle wide pool raises the price of exactly the
        // capacity that is free -- the split between the pools stops closing at the moment
        // closing it is worth the most, and the fleet pays for two pools while queueing
        // like the smaller one. On the wide pool's own queue the price says what it is
        // meant to say: this pool is full, stop sending, and otherwise it is not, so spill.
        // Measured offline on the 40-gpu mix this is worth +0.3 req/s of sustainable rate
        // on both the count and the token axis. On a uniform fleet there is no wide pool
        // and the excess is zero, so the fleet average is kept there and changes nothing.
        double gsBacklog = highCount > 0 ? highBacklog : backlog;
        double gsLambda = GPU_SECOND_PRICE_FLOOR
                + (GPU_SECOND_PRICE_CEILING - GPU_SECOND_PRICE_FLOOR)
                        * Math.min(1.0, gsBacklog / Math.max(1e-9, GPU_SECOND_PRICE_Q_REFERENCE));

        List<ScoredWorker> scored = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            CandidateEstimate estimate = estimates.get(i);
            WorkerStatus workerStatus = estimate.worker();
            double gpuSeconds = cp[i] * own[i];
            double socialSeconds = 0.0;
            if (SOCIAL_NU > 0.0) {
                double workerRate = socialRate(
                        socialKey(workerStatus, estimate.dpRank()));
                socialSeconds = SOCIAL_NU * own[i] * workerRate * (queue[i] + own[i]);
            }
            double cost;
            if (ADDITIONAL_GPU_SECONDS || GPU_SECOND_PRICE_CEILING > 0.0) {
                cost = queue[i] + OWN_WEIGHT * own[i]
                        + gsLambda * (gpuSeconds - cheapGpuSeconds[i]) + socialSeconds;
            } else if (ESCALATION_MU > 0.0) {
                double excess = (double) cp[i] / cheapestCp - 1.0;
                cost = queue[i] + OWN_WEIGHT * own[i] * (1.0 + mu * excess);
            } else {
                cost = QUEUE_PRICE * queue[i] + own[i] + lambda * gpuSeconds;
            }
            long score = costPoints(cost);
            if (cacheAwareService.prefixObserver() != null && cacheAwareService.prefixObserver().settings().observe()) {
                candidateCosts.add(new CandidateCostBreakdown(workerStatus.getDpRankCacheKey(estimate.dpRank()),
                        (int) cp[i], seqLen, hits[i], queue[i], own[i],
                        OWN_WEIGHT * own[i] + gsLambda * (gpuSeconds - cheapGpuSeconds[i]) + socialSeconds,
                        cost, gsLambda));
            }

            Logger.info("CP-aware gpu-second cost - ip: {}, port: {}, profile: {}, cpSize: {}, dpRank: {}, hit: {}, confirmedHit: {}, score: {}, cost: {}, queueSeconds: {}, ownSeconds: {}, completionSeconds: {}, gpuSeconds: {}, excessGpuSeconds: {}, gsLambda: {}, socialSeconds: {}, socialNu: {}, workerRate: {}, lambda: {}, mu: {}, backlogSeconds: {}, highBacklogSeconds: {}",
                    workerStatus.getIp(), workerStatus.getPort(), workerStatus.getProfileKey(),
                    workerStatus.getCpSize(), estimate.dpRank(), estimate.hitCacheTokens(),
                    confirmedReuseTokens(workerStatus.getDpRankCacheKey(estimate.dpRank()), seqLen),
                    score, cost, queue[i], own[i], completion[i], gpuSeconds,
                    gpuSeconds - cheapGpuSeconds[i], gsLambda, socialSeconds, SOCIAL_NU,
                    socialRate(socialKey(workerStatus, estimate.dpRank())),
                    lambda, mu, backlog, highBacklog);
            scored.add(new ScoredWorker(workerStatus, estimate.ttft(), score,
                    estimate.hitCacheTokens(), workerStatus.getLastSelectedTime().get()));
        }
        return scored;
    }

    private long confirmedReuseTokens(String route, long inputTokens) {
        var snapshot = frozenInventory.get(route);
        if (snapshot == null) { return 0; }
        long blockSize = snapshot.report().blockSize();
        return Math.min(blockSize * frozenConfirmedMatches.getOrDefault(route, 0),
                org.flexlb.cache.prefix.PrefillCostModel.maxReusableTokens(inputTokens, blockSize));
    }

    private static long costPoints(double cost) {
        double bounded = Math.max(0.0, Math.min(cost, COST_SECONDS_CAP));
        return Math.round(COST_SCORE_SCALE * bounded);
    }

    /**
     * Three-term score in which the load and cache terms are divided by one
     * shared denominator instead of each by its own maximum:
     * <pre>
     *   denom     = max_w ( queue_w + own_w )
     *   loadCost  = queue_w / denom
     *   cacheCost = own_w   / denom
     * </pre>
     * Both terms stay in [0, 1] and sum to at most 1, so the weights keep their
     * meaning, while the ratio between them is the ratio of the seconds they
     * stand for. A cache hit is then worth exactly the queueing it saves: on a
     * 1k request that is tens of milliseconds, on a 128k request it is seconds.
     * Dividing by a shared positive constant leaves the ordering unchanged, so
     * the ranking is that of {@code queue_w + own_w} in seconds.
     */
    private List<ScoredWorker> scoreBySharedDenominator(List<CandidateEstimate> estimates, long seqLen) {
        int n = estimates.size();
        double[] queueSeconds = new double[n];
        double[] ownSeconds = new double[n];
        double denom = 0.0;
        for (int i = 0; i < n; i++) {
            CandidateEstimate estimate = estimates.get(i);
            double cpSize = Math.max(1L, estimate.worker().getCpSize());
            long hit = Math.max(0L, Math.min(estimate.hitCacheTokens(), seqLen));
            queueSeconds[i] = calculateQueueSeconds(estimate.worker(), estimate.dpRank());
            ownSeconds[i] = prefillSeconds(cpSize, seqLen - hit, hit);
            denom = Math.max(denom, queueSeconds[i] + ownSeconds[i]);
        }
        long minCpSize = estimates.stream()
                .mapToLong(estimate -> Math.max(1L, estimate.worker().getCpSize()))
                .min()
                .orElse(1L);
        long maxCpSize = estimates.stream()
                .mapToLong(estimate -> Math.max(1L, estimate.worker().getCpSize()))
                .max()
                .orElse(minCpSize);

        List<ScoredWorker> scored = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            CandidateEstimate estimate = estimates.get(i);
            WorkerStatus workerStatus = estimate.worker();
            double loadCost = denom > 0.0 ? queueSeconds[i] / denom : 0.0;
            double cacheCost = denom > 0.0 ? ownSeconds[i] / denom : 0.0;
            double lengthCost = calculateLengthCpMismatch(
                    seqLen, estimate.hitCacheTokens(),
                    workerStatus.getCpSize(), minCpSize, maxCpSize);
            long score = UNIFIED_NORMALIZE
                    ? Math.max(0L, Math.round(UNIFIED_SCORE_SCALE * (
                            LOAD_WEIGHT * loadCost
                                    + UNIFIED_LENGTH_WEIGHT * lengthCost
                                    + CACHE_WEIGHT * cacheCost)))
                    : Math.max(0L, Math.round(SECONDS_SCORE_SCALE * (
                            LOAD_WEIGHT * queueSeconds[i]
                                    + UNIFIED_LENGTH_WEIGHT * lengthCost
                                    + CACHE_WEIGHT * ownSeconds[i])));

            Logger.info("CP-aware unified cost - ip: {}, port: {}, profile: {}, cpSize: {}, dpRank: {}, hit: {}, score: {}, loadCost: {}, lengthCost: {}, cacheCost: {}, queueSeconds: {}, ownSeconds: {}, denom: {}",
                    workerStatus.getIp(), workerStatus.getPort(), workerStatus.getProfileKey(),
                    workerStatus.getCpSize(), estimate.dpRank(), estimate.hitCacheTokens(),
                    score, loadCost, lengthCost, cacheCost, queueSeconds[i], ownSeconds[i], denom);
            scored.add(new ScoredWorker(workerStatus, estimate.ttft(), score,
                    estimate.hitCacheTokens(), workerStatus.getLastSelectedTime().get()));
        }
        return scored;
    }

    static double prefillSeconds(double cpSize, double uncachedLen, double prefixLen) {
        return prefillCostModel().seconds(cpSize, uncachedLen, prefixLen);
    }

    private static org.flexlb.cache.prefix.PrefillCostModel prefillCostModel() {
        return new org.flexlb.cache.prefix.PrefillCostModel(
                LAT_FLOOR, LAT_D0, LAT_G_SHARD, LAT_G_FIX, LAT_G_CACHE, LAT_ALPHA);
    }

    /**
     * The fixed cost a co-scheduled batch pays once rather than per request.
     *
     * <p>Where the model saturates, that cost is the floor rather than d0: a
     * queue of four 1k requests on DeepSeek-V4-Flash does not take 4 x 229 ms,
     * because the reason a 1k request costs 229 ms is that its kernel launches
     * dominate, and batched requests share those launches. Taking the larger of
     * the two is exact at both ends -- LAT_FLOOR when the whole queue sits in
     * the launch-bound region, LAT_D0 when the floor is disabled -- and never
     * drives a per-request term negative, since prefillSeconds cannot return
     * less than either.
     */
    private static final double QUEUE_FIXED_ONCE = Math.max(LAT_FLOOR, LAT_D0);

    public static double calculateQueueSeconds(WorkerStatus worker, int dpRank) {
        if (worker == null || worker.getLocalTaskMap() == null) {
            return 0.0;
        }
        double cpSize = Math.max(1L, worker.getCpSize());
        double seconds = 0.0;
        boolean queued = false;
        for (TaskInfo task : worker.getLocalTaskMap().values()) {
            if (task == null || task.getDpRank() != dpRank) {
                continue;
            }
            if (task.getTaskState() == TaskStateEnum.RUNNING && task.getIterateCount() > 0) {
                continue;
            }
            double p = Math.max(0L, task.getPrefixLength());
            double r = Math.max(0L, task.getInputLength() - task.getPrefixLength());
            seconds += prefillSeconds(cpSize, r, p) - (QUEUE_D0_ONCE ? QUEUE_FIXED_ONCE : 0.0);
            queued = true;
        }
        if (queued && QUEUE_D0_ONCE) {
            seconds += QUEUE_FIXED_ONCE;
        }
        return Math.max(0.0, seconds);
    }

    private double calculateQueuedAttentionWork(WorkerStatus worker, int dpRank) {
        if (worker == null || worker.getLocalTaskMap() == null) {
            return 0.0;
        }
        double cpSize = Math.max(1L, worker.getCpSize());
        double work = 0.0;
        for (TaskInfo task : worker.getLocalTaskMap().values()) {
            if (task == null || task.getDpRank() != dpRank) {
                continue;
            }
            if (task.getTaskState() == TaskStateEnum.RUNNING && task.getIterateCount() > 0) {
                continue;
            }
            // Paper attention work: W_attn(R, P) = R*P + R(R+1)/2, then / C_w.
            // Scaled by 1/1024^2 (K^2 units) purely for readable magnitudes;
            // the scale cancels in the max-normalization.
            double r = Math.max(0L, task.getInputLength() - task.getPrefixLength());
            double p = Math.max(0L, task.getPrefixLength());
            work += (r * p + r * (r + 1.0) / 2.0) / cpSize / (1024.0 * 1024.0);
        }
        return Math.max(0.0, work);
    }

    /**
     * Cache-adjusted equivalent uncached length R~ from the paper:
     * solve R~(R~+1)/2 = R*P + R(R+1)/2, i.e. the uncached length whose
     * attention work equals this request's residual work on the worker.
     */
    private double calculateEquivalentLength(long seqLen, long hitCacheTokens) {
        double p = Math.max(0.0, Math.min(hitCacheTokens, seqLen));
        double r = Math.max(0.0, seqLen - p);
        return (Math.sqrt(1.0 + 8.0 * r * p + 4.0 * r * (r + 1.0)) - 1.0) / 2.0;
    }

    private double calculateLengthCpMismatch(
            long seqLen, long hitCacheTokens, long cpSize, long minCpSize, long maxCpSize) {
        double minCp = Math.max(1L, minCpSize);
        double maxCp = Math.max(minCp, maxCpSize);
        if (maxCp <= minCp) {
            return 0.0;
        }
        double tokenUnit = Math.max(1.0, TOKENS_PER_CP_UNIT);
        double equivalentLen = calculateEquivalentLength(seqLen, hitCacheTokens);
        double targetCp = Math.max(minCp, Math.min(maxCp, equivalentLen / tokenUnit));
        double actualCp = Math.max(minCp, Math.min(maxCp, Math.max(1L, cpSize)));
        double range = Math.log(maxCp / minCp) / Math.log(2.0);
        double distance = Math.abs(Math.log(actualCp / targetCp) / Math.log(2.0));
        return Math.min(1.0, distance / range);
    }

    private double calculatePrefixCacheMissRatio(long seqLen, long hitCacheTokens) {
        if (seqLen <= 0L) {
            return 0.0;
        }
        double hitRatio = Math.max(0.0, Math.min(1.0, hitCacheTokens / (double) seqLen));
        return 1.0 - hitRatio;
    }

    private long safeAdd(long base, long... values) {
        if (base == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        long result = base;
        for (long value : values) {
            if (value > 0 && result > Long.MAX_VALUE - value) {
                return Long.MAX_VALUE;
            }
            result += value;
        }
        return result;
    }

    private long estimateLegacyQueueMs(WorkerStatus worker, int dpRank) {
        if (worker == null || worker.getLocalTaskMap() == null) {
            return 0L;
        }
        return worker.getLocalTaskMap().values().stream()
                .filter(task -> task.getDpRank() == dpRank)
                .mapToLong(TaskInfo::estimatePrefillTime)
                .sum();
    }

    private record CandidateEstimate(WorkerStatus worker,
                                     int dpRank,
                                     long requestMs,
                                     long queueMs,
                                     long ttft,
                                     long hitCacheTokens) {
    }

    private ServerStatus finalizeWorkerSelection(ScoredWorker selectedWorker,
                                                 BalanceContext balanceContext,
                                                 RoleType roleType,
                                                 long requestId,
                                                 long seqLen) {
        WorkerStatus workerStatus = selectedWorker.worker();
        int dpRank = workerStatus.resolveDpRank(requestId);
        noteSocialArrival(socialKey(workerStatus, dpRank), System.currentTimeMillis());

        Logger.info("CP-aware selected {} worker - ip: {}, port: {}, profile: {}, dpRank: {}, hit: {}, ttft: {}, score: {}",
                roleType, workerStatus.getIp(), workerStatus.getPort(), workerStatus.getProfileKey(),
                dpRank, selectedWorker.hitCacheTokens(), selectedWorker.ttft(), selectedWorker.score());

        // cp is on the line because the heterogeneous claim is about the joint
        // distribution of chosen degree against length and prefix hit, and those
        // three fields are enough to recover it. score, not ttft: ttft carries the
        // engine's own `tokens - 0.7*hit` estimate for downstream reporting, while
        // score is the quantity this strategy actually ordered candidates by.
        Logger.info("ROUTE_DECISION strategy=CP_AWARE_TTFT requestId={} seqLen={} hit={} worker={}:{} cp={} score={}",
                requestId, seqLen, selectedWorker.hitCacheTokens(), workerStatus.getIp(), workerStatus.getPort(),
                workerStatus.getCpSize() > 0 ? workerStatus.getCpSize() : workerStatus.getTpSize(),
                selectedWorker.score());

        double hitRate = seqLen > 0 ? selectedWorker.hitCacheTokens() / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, workerStatus.getIp(),
                selectedWorker.hitCacheTokens(), hitRate);

        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(seqLen);
        task.setPrefixLength(selectedWorker.hitCacheTokens());
        task.setDpRank(dpRank);
        workerStatus.putLocalTask(requestId, task);

        List<Long> requestKeys = balanceContext.getRequest().getBlockCacheKeys();
        String rankCacheKey = workerStatus.getDpRankCacheKey(dpRank);
        var observer = cacheAwareService.prefixObserver();
        if (observer == null || !observer.settings().confirmed()
                || observer.settings().namespace().equals(balanceContext.getRequest().getPrefixCacheNamespace())) {
            cacheAwareService.addCacheBlocks(rankCacheKey, workerStatus.getRole(), requestKeys);
        }

        if (cpCompositionController != null
                && (RoleType.PREFILL.equals(roleType) || RoleType.PDFUSION.equals(roleType))) {
            cpCompositionController.observeRequest(requestId, seqLen,
                    selectedWorker.hitCacheTokens(),
                    workerStatus.getCpSize() > 0
                            ? workerStatus.getCpSize() : workerStatus.getTpSize(),
                    requestKeys);
        }

        return buildServerStatus(selectedWorker, roleType, requestId);
    }

    private ScoredWorker selectBestWorker(List<ScoredWorker> scoredWorkers) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        ScoredWorker best = scoredWorkers.stream()
                .min(Comparator.comparingLong(ScoredWorker::score)
                        .thenComparingLong(ScoredWorker::lastSelectedTime))
                .orElse(null);
        if (best == null) {
            return null;
        }
        ScoredWorker selected = best;
        if (TIE_QUANTUM_POINTS > 0L) {
            long cpSize = best.worker().getCpSize();
            long limit = best.score() + TIE_QUANTUM_POINTS;
            selected = scoredWorkers.stream()
                    .filter(w -> w.worker().getCpSize() == cpSize && w.score() <= limit)
                    .min(Comparator.comparingLong(ScoredWorker::lastSelectedTime)
                            .thenComparingLong(ScoredWorker::score))
                    .orElse(best);
        }
        selected.worker().getLastSelectedTime().set(System.nanoTime() / 1000);
        return selected;
    }

    private ServerStatus buildServerStatus(ScoredWorker selectedWorker, RoleType roleType, long requestId) {
        WorkerStatus workerStatus = selectedWorker.worker();
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setRequestId(requestId);
            result.setPrefillTime(selectedWorker.ttft());
            result.setHitCacheTokens(selectedWorker.hitCacheTokens());
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            int dpRank = workerStatus.resolveDpRank(requestId);
            result.setGrpcPort(dpRankGrpcPort(workerStatus, dpRank));
            result.setProfileName(workerStatus.getProfileKey());
            result.setCpSize(workerStatus.getCpSize());
            result.setTpSize(workerStatus.getTpSize());
        } catch (Exception e) {
            Logger.error("CP-aware: failed to build server status for requestId: {}", requestId, e);
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

    private long calculatePrefixMatchLength(
            WorkerStatus workerStatus, Map<String, Integer> cacheMatchResults,
            long requestId) {
        if (cacheMatchResults == null) {
            return 0L;
        }
        String rankCacheKey = workerStatus.getDpRankCacheKey(requestId);
        Integer prefixMatchLength = cacheMatchResults.get(rankCacheKey);
        if (prefixMatchLength == null && workerStatus.getDpSize() <= 1) {
            prefixMatchLength = cacheMatchResults.get(workerStatus.getIpPort());
        }
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return 0L;
        }
        long blockSize = 0L;
        org.flexlb.dao.master.CacheStatus rankCache =
                workerStatus.getDpRankCacheStatusForRequest(requestId);
        if (rankCache != null) {
            blockSize = rankCache.getBlockSize();
        } else if (workerStatus.getCacheStatus() != null) {
            blockSize = workerStatus.getCacheStatus().getBlockSize();
        }
        if (blockSize <= 0L) {
            blockSize = DEFAULT_BLOCK_SIZE_TOKENS;
            Logger.info("CP-aware calculatePrefixMatchLength: worker={} using default blockSize={} (cacheStatus present={}, reportedBlockSize={})",
                    rankCacheKey, DEFAULT_BLOCK_SIZE_TOKENS,
                    rankCache != null || workerStatus.getCacheStatus() != null,
                    rankCache != null ? rankCache.getBlockSize()
                            : workerStatus.getCacheStatus() == null
                                    ? -1 : workerStatus.getCacheStatus().getBlockSize());
        }
        return blockSize * prefixMatchLength;
    }
}
