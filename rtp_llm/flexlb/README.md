# FlexLB

FlexLB is the scheduling and control plane for Vertumnus. It implements CP-aware request routing, resident worker merge and split, and prefix cache management.

Build with JDK 21 and Maven from this directory:

```bash
mvn -Popensource clean package
```

The application is packaged as `flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar`.
See the [project quick start](../../README.md#quick-start) for launch commands,
[worker reconfiguration](../../docs/backend/worker_split_merge.md) for cluster configuration,
and [prefix cache routing](docs/prefix_cache_routing_view.md) for cache semantics.
