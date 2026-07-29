# Execution backends

The local executor supports:

- synchronous execution for debugging and low overhead;
- threads for I/O-bound partitions;
- spawned processes for CPU-bound, pickle-compatible top-level blocks.

Every node declares its own parallelism and `ResourceRequirements` (CPU, GPU,
GPU-memory hint, RAM, scratch, custom resources, concurrency group,
affinity/exclusivity and duration estimate).

The Prefect adapter wraps domain execution in a flow and maps failures without
exposing Prefect states. The Ray adapter translates resources and resolves
`ObjectRef` before returning. Prefect-Ray uses the official task runner.
Backend switching does not change user block imports.
