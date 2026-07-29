# Performance guide

Choose artifact granularity large enough to amortize serializer, hash and SQL
cost, but small enough to resume useful partitions. Batch scalar metrics rather
than committing one transaction each in high-frequency loops.

Avoid database arrays, repeated serialization between local neighbors, loading
whole datasets in a driver, and passing stored objects through Ray. Reuse
manifest hashes for large files. Bound event payloads.

The benchmark script measures lightweight block overhead, partition overhead,
cache lookup, hashing and SQLite concurrency. PostgreSQL/Ray results are
environment-dependent and reported only when those services are available.

Run the checked-in local benchmark with:

```bash
python benchmarks/benchmark_core.py
```

The recorded baseline and its interpretation are in the repository-level
`PERFORMANCE_REPORT.md`. These figures measure framework overhead on one
development machine; they are not claims about backend or workload throughput.
