package org.flexlb.sync.prefix;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Message;
import io.grpc.Status;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Endpoint;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

final class PrefixRpcTiming {
    static final String PROPERTY = "flexlb.prefix.rpcTiming";
    private static final AtomicLong SEQUENCE = new AtomicLong();
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final List<String> FIELDS = List.of("request_id", "operation_id", "lease_id", "source_lease_id",
            "epoch", "version", "expected_version", "committed_version", "deadline_unix_ms", "success", "state");
    private final Consumer<Map<String, Object>> sink;
    private final LongSupplier nanos;
    private final LongSupplier millis;

    PrefixRpcTiming(Consumer<Map<String, Object>> sink, LongSupplier nanos, LongSupplier millis) {
        this.sink = sink;
        this.nanos = Objects.requireNonNull(nanos);
        this.millis = Objects.requireNonNull(millis);
    }

    static PrefixRpcTiming configured() {
        String value = System.getProperty(PROPERTY, "false");
        if (!value.equals("true") && !value.equals("false")) {
            throw new IllegalArgumentException("invalid prefix RPC timing property");
        }
        return new PrefixRpcTiming(value.equals("true") ? PrefixRpcTiming::write : null,
                System::nanoTime, System::currentTimeMillis);
    }

    <Response extends Message> Response call(String method, Endpoint endpoint, Message request,
                                             long timeoutMs, Supplier<Response> operation) {
        if (sink == null) {
            return operation.get();
        }
        long sequence = SEQUENCE.incrementAndGet();
        long startedMillis = millis.getAsLong();
        long startedNanos = nanos.getAsLong();
        Response response = null;
        Throwable failure = null;
        try {
            response = operation.get();
            return response;
        } catch (RuntimeException | Error exception) {
            failure = exception;
            throw exception;
        } finally {
            long elapsed = nanos.getAsLong() - startedNanos;
            try {
                Map<String, Object> event = new LinkedHashMap<>();
                event.put("schema", 1);
                event.put("sequence", sequence);
                event.put("method", method);
                event.put("endpoint", endpoint.ip() + ":" + endpoint.port());
                event.put("thread_id", Thread.currentThread().threadId());
                event.put("started_unix_ms", startedMillis);
                event.put("started_nanos", startedNanos);
                event.put("elapsed_nanos", elapsed);
                event.put("timeout_ms", timeoutMs);
                event.put("outcome", failure == null ? "returned" : "threw");
                event.put("grpc_status", failure == null ? "OK" : Status.fromThrowable(failure).getCode().name());
                event.put("request", fields(request));
                event.put("response", response == null ? null : fields(response));
                sink.accept(event);
            } catch (RuntimeException ignored) {
            }
        }
    }

    private static Map<String, Object> fields(Message message) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (String name : FIELDS) {
            var field = message.getDescriptorForType().findFieldByName(name);
            if (field != null && !field.isRepeated()
                    && field.getJavaType() != com.google.protobuf.Descriptors.FieldDescriptor.JavaType.MESSAGE) {
                values.put(name, message.getField(field));
            }
        }
        for (String name : List.of("cache_keys", "missing_indices")) {
            var field = message.getDescriptorForType().findFieldByName(name);
            if (field != null && field.isRepeated()) {
                values.put(name + "_count", message.getRepeatedFieldCount(field));
            }
        }
        var transfer = message.getDescriptorForType().findFieldByName("transfer");
        if (transfer != null && message.hasField(transfer)) {
            Message identity = (Message) message.getField(transfer);
            for (String name : List.of("transfer_id", "source_worker_id", "destination_worker_id")) {
                var field = identity.getDescriptorForType().findFieldByName(name);
                values.put(name, identity.getField(field));
            }
        }
        return values;
    }

    private static void write(Map<String, Object> event) {
        try {
            System.err.println("PREFIX_RPC_TIMING " + JSON.writeValueAsString(event));
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("prefix RPC timing serialization failed", exception);
        }
    }
}
