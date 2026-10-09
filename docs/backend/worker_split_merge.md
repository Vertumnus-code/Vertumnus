# Resident worker split and merge

Prefill context-parallel workers can change the partition of an existing GPU
world without restarting model processes. For example, a four-rank world can
serve one CP4 worker or two independent CP2 workers. The world size and model
weights remain fixed. Ordinary weight-sharded tensor parallelism cannot use
this operation.

## Engine setup

Configure prefill CP and the supported TP/DP partitions before starting workers:

```sh
CP_ROTATE_METHOD=ALL_GATHER
TP_SIZE=4
DP_SIZE=1
WORLD_SIZE=4
RTP_LLM_PREWARM_TOPOLOGIES=2:2,4:1
```

Prewarming creates and exercises all configured CUDA/NCCL and CPU process
groups before KV-cache sizing. Switching reuses those groups. Every rank must
receive the same configuration. This mode supports CUDA without user-buffer
communication. Leave `RTP_LLM_PREWARM_TOPOLOGIES` unset to retain the existing
process-group recreation behavior.

Communicator prewarming does not compile every model kernel. First inference
under a new partition can still trigger Triton compilation, increasing the
first switch's admission delay. When measuring steady-state switching, run
representative inference under each partition before the timed window, retain
the processes and weights, and reset cached prefixes as required by the
experiment. Report cold transitions separately.

Every rank initializes its cache policy because a follower can become a DP
leader after a split. With `RTP_PREFIX_CACHE_ENABLE_P2P=1`, topology completion
also prepares the new leader's prefix-copy endpoint before reopening inventory.
Endpoint initialization failure keeps inventory fenced. Initialized endpoints
remain available across subsequent merges and splits at their per-rank ports.

Use a short `TMPDIR` for PyTorch symmetric-memory IPC. A short local symlink to
a NAS temporary directory keeps temporary data on NAS without exceeding Unix
socket path limits. Long temporary paths caused a native `IpcChannel` startup
failure with the validated PyTorch 2.8 environment.

Each native world has one frontend route. Configure its managed topology writer
with `PREFIX_TOPOLOGY_WRITER_URL`, `PREFIX_TOPOLOGY_WRITER_TOKEN`, and
`PREFIX_TOPOLOGY_FRONTEND_ROUTE`. Set `PREFIX_TOPOLOGY_STATE_DIRECTORY` to an
existing absolute directory shared by frontend processes in that world and
distinct from other worlds. The writer token must match the FlexLB control token.

## FlexLB setup

Set `FLEXLB_CP_SWITCH_MODE=native` and `FLEXLB_CP_SUPPORTED_SIZES=2,4`.
`FLEXLB_CP_EXPECTED_GPU_COUNT` is the total number of prefill GPUs across all
registered worlds. Native switch domains are frontend endpoints, so multiple
worlds on one physical host remain independent.

Enable `enableQueueing` in `FLEXLB_CONFIG`. Requests must wait within their
original deadline when a topology admission fence temporarily withholds routes;
direct routing without the queue returns resource-unavailable errors during
these intervals.

Native switching uses the prefix-cache topology transaction protocol. Enable
the recovery service and configure its ZooKeeper connection, cache namespace,
initial worker registrations, rank control/transfer endpoints, and allowed
cross-degree copy pairs. The required topology properties are:

```properties
prefix-cache.topology.recovery.enabled=true
prefix-cache.topology.placement.enabled=true
prefix-cache.topology.writer.enabled=true
prefix-cache.topology.control.enabled=true
prefix-cache.topology.writer.fenced-ingress-only=true
```

Also configure `prefix-cache.topology.worlds`,
`prefix-cache.topology.reader-id`, `prefix-cache.topology.required-readers`, and
`prefix-cache.topology.control.token`. The writer requires prefix observation and
exclusively fenced CP-aware ingress. Each world registration must describe every
rank and the current DP lane leaders.

The composition controller is disabled by default. To enable policy-driven
changes, set `FLEXLB_CP_COMPOSITION_ENABLED=true` and provide measured capacity
for every supported degree through `FLEXLB_CP_CAPACITY_PROFILE` or
`FLEXLB_CP_PROFILED_CAPACITY`. Capacity must match the model, hardware, and
observation window. The controller applies consecutive-window confirmation and
cooldown, then ranks eligible worlds by drain time and estimated cache traffic.

`GET /rtp_llm/cp_composition` reports policy and transition state. For a fixed
experimental schedule, keep the online controller disabled and explicitly enable
`FLEXLB_CP_ORACLE_ENABLED` with `FLEXLB_CP_ORACLE_TOKEN`. The authenticated
`POST /rtp_llm/cp_composition/scheduled_switch` accepts a frontend endpoint as
`host` and an adjacent `target_cp`. This path uses the same planner, drain fence,
native transaction, and adoption checks as policy-driven switching.

## Transition and recovery

1. Stop assigning new requests to the selected world. Resident plans drain only
   that world. External-copy preservation can temporarily fence all registered
   routes while taking its drained snapshot; the request queue retains new work.
2. Cancel only engine-owned RECEIVED requests that have not allocated KV. The
   frontend can obtain a fresh master route before any output is emitted. It
   preserves the request deadline and limits repeated rerouting to eight retries.
3. Drain remaining work and create an immutable durable topology plan.
4. The authenticated frontend executes the native switch through the managed
   writer, updates its DP addresses, and admits the confirmed cache topology.
5. Publish the new composition only after participant status and frontend
   admission agree. An ambiguous timeout keeps the participant world withheld.

The controller selects resident repair only when every participant advertises
the corresponding capability. The immutable plan and native receipt bind
`resident_prefix_split_v1` for an adjacent split or `resident_leader_merge_v1`
for an adjacent merge. A split preserves the leader's confirmed mapping and
installs it on both new lanes without relocating KV. A merge retains the primary
lane's confirmed inventory and broadcasts its resident KV pool across the new
group, then retires secondary-only replica records. It does not promise to
retain a union larger than the physical cache pool. A failed repair keeps the
world quarantined; after overwriting resident KV, cancellation cannot restore
the former topology.

Resident preservation reads only the affected world's inventories. New cache
actions are paused while its topology plan is pending, but recovery of existing
durable operations remains permitted so the drain can finish. Recovery uses
the shared topology lock and immutable operation bindings; the exclusive
maintenance lock prevents it from overlapping native commit.

Placement refresh reads independent worlds' native and frontend proofs
concurrently, with at most 16 worlds in flight. It validates and publishes them
under the placement lock and rechecks the durable journal before releasing
fences. A missing, stale, or
unready proof still withholds its world. HTTP transport completion alone does
not establish topology readiness.

The drain barrier checks owned requests on the affected routes. Native reuse
events have bounded history, so a missing event does not keep a drained worker
fenced until the switch deadline. Preservation conservatively includes prefixes
that pending observations reference and the confirmed inventory still holds;
uncached suffixes are excluded. This does not commit a synthetic reuse event or
alter measured demand statistics. Requests waiting for an available route retry
at the configured resource-check interval, releasing their scheduling permit
between attempts.

The journal also retains the immutable preservation copy plan. An interrupted
advance resumes copies not yet issued and skips paths already confirmed at their
destination. A committed transfer whose path is missing remains a failed proof;
it is not silently recopied. Unresolved transfer outcomes continue to fence the
transition. The managed frontend lets an advance run up to its bounded caller
deadline, because copying several long prefixes may exceed 30 seconds; retries
retain the original switch identity and immutable plan deadline.

Inspect the durable plan identified in the transition error before recovery.
Retry the same immutable request through the frontend; do not mint a replacement
plan while the previous one is unresolved. Engine confirmation alone is not
frontend admission. `POST /rtp_llm/cp_composition/reconcile` can publish a completed
transition after all participant and admission checks succeed.
An ADMITTED journal entry does not prove that the original frontend released its
shared ingress gate. The actuator requires an `ok` acknowledgement with that
same switch identity, and replays the immutable request when the original
response was lost or rejected. A response for another plan cannot complete
recovery. Admission receipts are read under the placement projection lock so a
concurrent background refresh cannot invalidate the projection between admission
and its receipt.
Reconciliation can also resolve a durably cancelled native plan when the participant degree
and epoch, full composition, and non-participant topology still match the
pre-switch state. Old worker heartbeats alone never establish cancellation.
The actuator must also replay the original cancelled request and receive the
frontend's matching cancellation acknowledgement before releasing composition
recovery. This clears a shared ingress gate left behind by a timed-out caller.
For an interrupted attempt whose actuator state was lost on process restart,
retry that same immutable request through its original frontend after checking
the durable journal; do not edit the shared gate file.

When resident repair is unavailable, external-copy preservation requires
sufficient physically free blocks on compatible,
unaffected workers, within the configured copy budgets. Evictable cached blocks
are not counted as free capacity. When free space is insufficient, the writer
can reclaim unrelated cached suffixes through the durable reclamation protocol.
It protects every required path on every destination, orders known descendants
before parents, and only nominates victims from inventories with consistent
physical accounting and no active, resident, or operation-reserved blocks.
Native epoch, version, pin, and tree checks remain authoritative. A fresh report
must confirm both key removal and the released physical space before the writer
validates the original copy allocation; a successful RPC alone is insufficient.
Shared ancestors and all pending copies are charged against fresh physical space
without greedily moving paths onto capacity reserved for other paths. Ambiguous reclamation
is journaled and never blindly replayed. Copy budgets and required paths are
unchanged. If these conditions cannot be satisfied, preservation can still
refuse the transition; the global reader fence may remain closed until the
immutable plan is cancelled at its deadline.
A definitive reclaim rejection for an unchanged worker epoch, lane, layout,
and inventory version disables further inferred subsets on that lane. The writer
can instead reclaim its entire inventory complement of all required paths,
including descendants absent from the controller's partial tree. The complete
remainder is attempted atomically, stays within the 8,192-key bound, and still
requires native validation and fresh physical proof. Free destinations remain
preferred; after a rejection this fallback takes precedence over other inferred
subsets. It can evict more unrelated cache blocks. A rejected complete remainder
is excluded until the inventory changes; unknown outcomes are never retried.
Rejection evidence is durable, so controller retries retain these safeguards.
Enable DEBUG for `org.flexlb.httpserver.PrefixTopologyControlServer` to inspect
the cause behind a retryable control response.
Composition observation and reconciliation use a delay after each completed
control task. A slow switch therefore cannot cause timer catch-up to produce
millisecond demand windows or consume cooldown without real observation time.
Durable topology evidence retains full inventory keys and confirmation identity,
but excludes historical reuse/copy telemetry and duplicate legacy cache keys so
normal request history does not consume the journal payload budget.
Large topology journal entries use gzip while retaining the same JSON schema
and all inventory evidence. Existing uncompressed entries remain readable. Stored
entries remain limited to 512 KiB, and decoded JSON is limited to 8 MiB; corrupt
or oversized entries fail closed. Older binaries cannot read compressed entries,
so downgrade requires draining the controller and migrating its journal first.

After a controller crash, reader registration waits for the preceding ZooKeeper
session to release its ephemeral node, for at most twice the negotiated session
timeout (capped at 60 seconds). It never replaces another reader's node. A lost
session invalidates routing observation; after reconnection the reader registers
under the new session and publishes a fresh acknowledgement before becoming
healthy. Cohorts captured under the expired session remain invalid.

The legacy external-controller HTTP actuator remains available with
`FLEXLB_CP_SWITCH_MODE=http` (the default).
