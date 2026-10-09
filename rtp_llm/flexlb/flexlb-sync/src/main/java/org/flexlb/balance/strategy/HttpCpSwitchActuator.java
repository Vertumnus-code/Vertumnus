package org.flexlb.balance.strategy;

import com.fasterxml.jackson.core.type.TypeReference;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** HTTP bridge from FlexLB to the resident Python CP switch controller. */
@Component
@org.springframework.boot.autoconfigure.condition.ConditionalOnProperty(
        name = "FLEXLB_CP_SWITCH_MODE", havingValue = "http", matchIfMissing = true)
public class HttpCpSwitchActuator implements CpSwitchActuator {

    private final String actuatorUrl;
    private final String statusUrl;
    private final String token;
    private final HttpClient client;

    public HttpCpSwitchActuator() {
        this(System.getenv("FLEXLB_CP_SWITCH_ACTUATOR_URL"),
                System.getenv("FLEXLB_CP_SWITCH_STATUS_URL"),
                System.getenv("FLEXLB_CP_SWITCH_ACTUATOR_TOKEN"));
    }

    HttpCpSwitchActuator(String actuatorUrl, String token) {
        this(actuatorUrl, null, token);
    }

    HttpCpSwitchActuator(String actuatorUrl, String statusUrl, String token) {
        this.actuatorUrl = actuatorUrl == null ? "" : actuatorUrl.trim();
        this.statusUrl = statusUrl == null || statusUrl.isBlank()
                ? deriveStatusUrl(this.actuatorUrl) : statusUrl.trim();
        this.token = token == null ? "" : token.trim();
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    }

    @Override
    public boolean isConfigured() {
        return !actuatorUrl.isBlank();
    }

    /** The switch endpoint and the status endpoint are siblings on the same service. */
    static String deriveStatusUrl(String switchUrl) {
        String base = switchUrl == null ? "" : switchUrl.trim();
        if (base.isEmpty()) {
            return "";
        }
        int query = base.indexOf('?');
        if (query >= 0) {
            base = base.substring(0, query);
        }
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        if (base.endsWith("/switch")) {
            base = base.substring(0, base.length() - "/switch".length());
        }
        return base + "/status";
    }

    @Override
    public Result switchTo(int targetCp, long timeoutMs) throws Exception {
        return new Result(false, false, -1, -1L,
                "exactly one resident host is required; whole-pool switching is disabled", "");
    }

    @Override
    public Result switchTo(int targetCp, long timeoutMs, List<String> hosts) throws Exception {
        if (!isConfigured()) {
            return new Result(false, false, -1, -1L,
                    "FLEXLB_CP_SWITCH_ACTUATOR_URL is not configured", "");
        }
        List<String> selectedHosts = hosts == null ? List.of() : hosts.stream()
                .filter(host -> host != null && !host.isBlank())
                .map(String::trim)
                .distinct()
                .toList();
        if (selectedHosts.size() != 1) {
            return new Result(false, false, -1, -1L,
                    "exactly one resident host is required; got " + selectedHosts, "");
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("target_cp", targetCp);
        body.put("protocol", "barrier");
        body.put("drain_timeout_s", Math.max(1L, (timeoutMs + 999L) / 1000L));
        // Name the host explicitly rather than letting the actuator pick its own: the
        // router has already withheld this host's endpoints and only these.
        body.put("rolling", true);
        body.put("hosts", selectedHosts);

        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(actuatorUrl))
                .timeout(Duration.ofMillis(Math.max(timeoutMs, 1_000L) + 10_000L))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JsonUtils.toString(body)));
        if (!token.isBlank()) {
            request.header("Authorization", "Bearer " + token);
        }

        HttpResponse<String> response = client.send(request.build(), HttpResponse.BodyHandlers.ofString());
        String responseBody = response.body() == null ? "" : response.body();
        Map<String, Object> payload = responseBody.isBlank()
                ? Map.of()
                : JsonUtils.toObject(responseBody, new TypeReference<Map<String, Object>>() { });
        boolean success = booleanValue(payload.get("ok"));
        boolean committed = booleanValue(payload.get("committed"));
        int activeCp = intValue(payload.get("active_cp"), -1);
        long centralEpoch = longValue(payload.get("epoch"), -1L);
        // The central actuator owns a fleet-wide request epoch, while worker status
        // publishes the selected host's engine epoch. Adoption must compare like with
        // like, so Result.epoch is deliberately the host-local value.
        long epoch = longValue(payload.get("host_epoch"), centralEpoch);
        String participantHost = stringValue(payload.get("participant_host"));
        Map<String, Integer> composition = intMap(payload.get("composition"));
        String error = stringValue(payload.get("error"));
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            success = false;
            if (error.isBlank()) {
                error = "actuator returned HTTP " + response.statusCode();
            }
        }
        Logger.info("CP actuator response target=CP{} success={} committed={} active=CP{} hostEpoch={} centralEpoch={} error={}",
                targetCp, success, committed, activeCp, epoch, centralEpoch, error);
        return new Result(success, committed, activeCp, epoch, error, responseBody,
                participantHost, composition);
    }

    @Override
    public Status status(long timeoutMs) throws Exception {
        if (!isConfigured()) {
            return Status.unreachable("FLEXLB_CP_SWITCH_ACTUATOR_URL is not configured");
        }
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(statusUrl))
                .timeout(Duration.ofMillis(Math.max(timeoutMs, 500L)))
                .GET();
        if (!token.isBlank()) {
            request.header("Authorization", "Bearer " + token);
        }

        HttpResponse<String> response = client.send(request.build(), HttpResponse.BodyHandlers.ofString());
        String responseBody = response.body() == null ? "" : response.body();
        Map<String, Object> payload = responseBody.isBlank()
                ? Map.of()
                : JsonUtils.toObject(responseBody, new TypeReference<Map<String, Object>>() { });
        boolean ok = booleanValue(payload.get("ok"));
        String error = stringValue(payload.get("error"));
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            ok = false;
            if (error.isBlank()) {
                error = "actuator status returned HTTP " + response.statusCode();
            }
        }
        return new Status(true, ok,
                booleanValue(payload.get("busy")),
                intValue(payload.get("active_cp"), -1),
                longValue(payload.get("epoch"), -1L),
                longValue(payload.get("committed_epoch"), -1L),
                stringValue(payload.get("phase")),
                error, responseBody);
    }

    private static boolean booleanValue(Object value) {
        return value instanceof Boolean bool ? bool
                : value != null && Boolean.parseBoolean(value.toString());
    }

    private static int intValue(Object value, int defaultValue) {
        long parsed = longValue(value, defaultValue);
        return parsed > Integer.MAX_VALUE || parsed < Integer.MIN_VALUE ? defaultValue : (int) parsed;
    }

    private static long longValue(Object value, long defaultValue) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return value == null ? defaultValue : Long.parseLong(value.toString());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private static String stringValue(Object value) {
        return value == null ? "" : value.toString();
    }

    private static Map<String, Integer> intMap(Object value) {
        if (!(value instanceof Map<?, ?> source)) {
            return Map.of();
        }
        Map<String, Integer> result = new LinkedHashMap<>();
        source.forEach((key, item) -> {
            int parsed = intValue(item, -1);
            if (key != null && parsed > 0) {
                result.put(key.toString(), parsed);
            }
        });
        return Map.copyOf(result);
    }
}
