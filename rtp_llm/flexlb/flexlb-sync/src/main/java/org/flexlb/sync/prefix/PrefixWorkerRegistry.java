package org.flexlb.sync.prefix;

import org.flexlb.cache.prefix.ConfirmedCacheDirectory;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.Endpoint;
import org.flexlb.sync.prefix.PrefixTransferCoordinator.WorkerSnapshot;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class PrefixWorkerRegistry {
    public record Registration(String route, String workerId, Endpoint controlEndpoint, Endpoint transferEndpoint, int dpLane) {
        public Registration(String route, String workerId, Endpoint controlEndpoint, Endpoint transferEndpoint) {
            this(route, workerId, controlEndpoint, transferEndpoint, 0);
        }

        public ConfirmedCacheDirectory.Lane lane() {
            return new ConfirmedCacheDirectory.Lane(workerId, dpLane);
        }

        public Registration {
            Objects.requireNonNull(route);
            Objects.requireNonNull(workerId);
            Objects.requireNonNull(controlEndpoint);
            Objects.requireNonNull(transferEndpoint);
            if (route.isBlank() || workerId.isBlank() || route.length() > 1024 || workerId.length() > 512
                    || dpLane < 0 || controlEndpoint.equals(transferEndpoint)) {
                throw new IllegalArgumentException("invalid explicit prefix Worker registration");
            }
        }
    }
    private List<Registration> registrations;

    public PrefixWorkerRegistry(List<Registration> registrations) {
        this.registrations = List.copyOf(registrations);
        validate(registrations);
    }

    private static void validate(List<Registration> registrations) {
        if (registrations.isEmpty() || registrations.size() > 4096) {
            throw new IllegalArgumentException("prefix Worker registry is empty or exceeds capacity");
        }
        var routes = new HashSet<String>();
        var workers = new HashSet<ConfirmedCacheDirectory.Lane>();
        var endpoints = new HashSet<Endpoint>();
        for (var registration : registrations) {
            if (!routes.add(registration.route()) || !workers.add(registration.lane())
                    || !endpoints.add(registration.controlEndpoint()) || !endpoints.add(registration.transferEndpoint())) {
                throw new IllegalArgumentException("prefix Worker identity or endpoint is aliased");
            }
        }
    }

    public synchronized List<Registration> registrations() { return registrations; }

    public synchronized void replaceWorld(java.util.Set<String> workers, List<Registration> replacements) {
        if (workers.isEmpty() || replacements.isEmpty() || replacements.stream().anyMatch(registration -> !workers.contains(registration.workerId()))) {
            throw new IllegalArgumentException("invalid topology registry replacement");
        }
        var updated = new java.util.ArrayList<Registration>();
        registrations.stream().filter(registration -> !workers.contains(registration.workerId())).forEach(updated::add);
        updated.addAll(replacements);
        validate(updated);
        registrations = List.copyOf(updated);
    }

    public synchronized Map<String, WorkerSnapshot> freeze(Map<String, ConfirmedCacheDirectory.Snapshot> inventory) {
        Map<String, WorkerSnapshot> result = new HashMap<>();
        for (var registration : registrations) {
            var snapshot = inventory.get(registration.route());
            if (snapshot == null) { continue; }
            var report = snapshot.report();
            if (!registration.workerId().equals(report.workerId()) || report.dpLane() != registration.dpLane() || !report.deviceReady()) { continue; }
            result.put(registration.route(), new WorkerSnapshot(report.workerId(), report.epoch(), report.version(),
                    report.layoutFingerprint(), registration.controlEndpoint(), registration.transferEndpoint(), report.cpDegree(), report.dpLane()));
        }
        return Map.copyOf(result);
    }
}
