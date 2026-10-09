# Main-compatible routing cache view

`FLEXLB_PREFIX_ROUTING_CACHE_VIEW` explicitly selects the routing cache view when
`FLEXLB_CACHE_METADATA_MODE=confirmed`:

- `confirmed` is the default and preserves the existing confirmed-only routing.
- `main_compatible` restores the original main dispatch/report cache bookkeeping
  for routing. CP-aware routing augments that index with a frozen, physically
  confirmed prefix snapshot. Cost formulas and scheduler coefficients do not change.

The compatibility mode requires confirmed metadata and the existing namespace
configuration. Requests with a different namespace cannot consume or populate its
routing index. Native/legacy configurations without this option remain unchanged.

## Separate authorities

The routing index is intentionally optimistic: dispatching a request records its
keys before the worker confirms physical residency. Legacy key reports remain
merge-only, with the existing per-worker bounded LRU. Such membership is not a
physical cache hit, a promise of future residency, or permission to copy a block.

Prefix observation and COPY continue to use the confirmed directory. Actual reuse
events drive access accounting. Epochs, versions, all-rank readiness, topology and
namespace fences, true-free capacity and COPY safety margins remain unchanged.
The routing index is never imported into that directory.

Compatibility mode retains legacy cache-key reports alongside confirmed inventory;
the inventory-only RPC optimization is disabled for this mode even if requested.
Set `FLEXLB_PREFIX_INVENTORY_ONLY_RPC=false` in experiment manifests to record that
choice explicitly. Do not enable new inflight/history hints for a main-alignment
experiment; this mode restores existing bookkeeping rather than adding a new model.

## Measurement

GPU-second candidate logs expose two different values:

- `hit`: the routing/pricing metadata input, which may exceed physical readiness.
- `confirmedHit`: the reusable prefix in the frozen confirmed inventory view.

Physical-copy oracles and admission-to-native retention audits must use
`confirmedHit`, not the optimistic pricing value. End-to-end reuse remains the
actual native reused tokens divided by actual input tokens. Metadata matches,
copy counts and offline lookup differences are not substitutes for that metric
or for measured mean/P90 TTFT.
