# Runweaver performance baseline

Baseline date: 2026-07-29. Environment: CPython 3.12.3,
Linux 6.17 x86-64. The benchmark was run from an editable install with
`python benchmarks/benchmark_core.py`.

| Operation | Median | Observed p95 |
|---|---:|---:|
| One identity block, ephemeral execution | 1.1231 ms | 1.1610 ms |
| Durable SQLite cache lookup | 4.6553 ms | 5.1162 ms |
| 1,000 fingerprints of 64 scalars | 57.3186 ms | 59.5975 ms |
| Commit a 1 MiB local artifact | 2.0124 ms | 2.4431 ms |

The cache run reported `identity` as a cache hit. Values are seven local
samples (five for hashing and artifact commits), not a distributed throughput
claim. Filesystem cache, CPU scheduling and SQLite settings can materially
change them.

The practical conclusion is to avoid representing scalar inner-loop work as
individual durable blocks. Durable mode is intended for work large enough to
justify a few milliseconds of state, serialization and integrity overhead.
Large remote objects, concurrent PostgreSQL writers, GPU transfers, Prefect and
Ray require environment-specific benchmarks; no synthetic result for those
systems is claimed here.
