package org.flexlb.sync.prefix;

import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.imps.CuratorFrameworkState;
import org.apache.curator.framework.recipes.leader.LeaderLatch;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.framework.state.ConnectionStateListener;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.common.PathUtils;
import org.apache.zookeeper.data.Stat;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

public final class ZookeeperPrefixControllerAuthority implements PrefixControllerAuthority, AutoCloseable {
    private record ConnectionObservation(ConnectionState state) {}

    private record Grant(EngineRpcService.PrefixControllerPB identity, String path, ZooKeeper session,
                         long sessionId, ConnectionObservation connection) {}

    private final CuratorFramework client;
    private final String namespace;
    private final String electionPath;
    private final String controllerId = UUID.randomUUID().toString();
    private volatile LeaderLatch latch;
    private final AtomicReference<ConnectionObservation> connection =
            new AtomicReference<>(new ConnectionObservation(null));
    private final AtomicReference<Grant> grant = new AtomicReference<>();
    private final ConnectionStateListener connectionListener = this::connectionChanged;
    private volatile boolean started;
    private volatile boolean closed;
    private volatile boolean rejoinPending;

    public ZookeeperPrefixControllerAuthority(CuratorFramework client, String electionRoot, String namespace) {
        this.client = Objects.requireNonNull(client);
        this.namespace = Objects.requireNonNull(namespace);
        PathUtils.validatePath(Objects.requireNonNull(electionRoot));
        if (electionRoot.equals("/") || namespace.isBlank()
                || namespace.getBytes(StandardCharsets.UTF_8).length > 512) {
            throw new IllegalArgumentException("invalid prefix election root or namespace");
        }
        try {
            electionPath = electionRoot + "/" + HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(namespace.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is required for prefix namespace isolation", exception);
        }
        latch = new LeaderLatch(client, electionPath, controllerId);
    }

    public synchronized void start() throws Exception {
        if (closed) {
            throw new IllegalStateException("prefix authority is closed");
        }
        if (started) {
            return;
        }
        if (client.getState() != CuratorFrameworkState.STARTED) {
            throw new IllegalStateException("Curator client must be started by its owner");
        }
        client.getConnectionStateListenable().addListener(connectionListener);
        try {
            latch.start();
            started = true;
        } catch (Exception exception) {
            closed = true;
            client.getConnectionStateListenable().removeListener(connectionListener);
            try {
                latch.close();
            } catch (Exception cleanupFailure) {
                exception.addSuppressed(cleanupFailure);
            }
            throw exception;
        }
    }

    public String electionPath() {
        return electionPath;
    }

    @Override
    public EngineRpcService.PrefixControllerPB snapshot(String requestedNamespace) {
        if (!namespace.equals(requestedNamespace)) {
            throw new IllegalArgumentException("prefix authority namespace mismatch");
        }
        Grant previous = grant.get();
        try {
            ConnectionObservation observed = connection.get();
            rejoinAfterSessionLoss(observed);
            if (!connectedLeader(observed)) {
                throw new IllegalStateException("prefix controller is not a connected leader");
            }
            String path = latch.getOurPath();
            ZooKeeper session = client.getZookeeperClient().getZooKeeper();
            long sessionId = session.getSessionId();
            if (path == null || sessionId == 0 || session.getState() != ZooKeeper.States.CONNECTED) {
                throw new IllegalStateException("prefix controller session is not writable");
            }
            Stat node = client.checkExists().forPath(path);
            if (node == null || node.getEphemeralOwner() != sessionId || node.getCzxid() == 0) {
                throw new IllegalStateException("prefix election node is not owned by the current session");
            }
            var identity = EngineRpcService.PrefixControllerPB.newBuilder().setNamespace(namespace)
                    .setControllerId(latch.getId()).setTerm(node.getCzxid()).build();
            Grant candidate = new Grant(identity, path, session, sessionId, observed);
            if (!valid(candidate)) {
                throw new IllegalStateException("prefix leadership changed while verifying its election node");
            }
            grant.set(candidate);
            if (!valid(candidate)) {
                grant.compareAndSet(candidate, null);
                throw new IllegalStateException("prefix leadership changed while publishing its identity");
            }
            return identity;
        } catch (InterruptedException exception) {
            grant.compareAndSet(previous, null);
            Thread.currentThread().interrupt();
            throw new IllegalStateException("prefix controller verification interrupted", exception);
        } catch (Exception exception) {
            grant.compareAndSet(previous, null);
            throw new IllegalStateException("prefix controller identity unavailable", exception);
        }
    }

    @Override
    public boolean isCurrent(EngineRpcService.PrefixControllerPB controller) {
        Grant current = grant.get();
        if (current == null || !current.identity().equals(controller)) {
            return false;
        }
        try {
            return valid(current);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            return false;
        } catch (Exception exception) {
            return false;
        }
    }

    private boolean connectedLeader(ConnectionObservation observed) {
        ConnectionState state = observed.state();
        return started && !closed && client.getState() == CuratorFrameworkState.STARTED
                && (state == null || state == ConnectionState.CONNECTED || state == ConnectionState.RECONNECTED)
                && client.getZookeeperClient().isConnected() && latch.hasLeadership();
    }

    private boolean valid(Grant candidate) throws Exception {
        return connection.get() == candidate.connection() && connectedLeader(candidate.connection())
                && candidate.path().equals(latch.getOurPath())
                && candidate.session().getState() == ZooKeeper.States.CONNECTED
                && candidate.session().getSessionId() == candidate.sessionId()
                && client.getZookeeperClient().getZooKeeper() == candidate.session()
                && connection.get() == candidate.connection() && !closed;
    }

    void connectionChanged(CuratorFramework ignored, ConnectionState state) {
        connection.set(new ConnectionObservation(state));
        if (state == ConnectionState.LOST) {
            rejoinPending = true;
            grant.set(null);
        }
    }

    private synchronized void rejoinAfterSessionLoss(ConnectionObservation observed) throws Exception {
        if (!rejoinPending || !started || closed || connection.get() != observed
                || (observed.state() != ConnectionState.CONNECTED && observed.state() != ConnectionState.RECONNECTED)
                || !client.getZookeeperClient().isConnected()) {
            return;
        }
        rejoinPending = false;
        grant.set(null);
        var previous = latch;
        var replacement = new LeaderLatch(client, electionPath, controllerId);
        latch = replacement;
        try {
            if (previous.getState() != LeaderLatch.State.CLOSED) { previous.close(); }
            replacement.start();
        } catch (Exception failure) {
            rejoinPending = true;
            try { replacement.close(); }
            catch (Exception cleanupFailure) { failure.addSuppressed(cleanupFailure); }
            throw failure;
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        grant.set(null);
        client.getConnectionStateListenable().removeListener(connectionListener);
        if (started) {
            latch.close();
        }
    }
}
