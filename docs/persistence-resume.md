# Persistence, caching and resume

SQL transitions are transactional and use an optimistic version. Worker leases
record owner, acquire/heartbeat/expiration times and attempts. Reconciliation
marks expired running work stale; it never silently repeats an unsafe
non-idempotent side effect.

Artifact commit writes a temporary object, flushes, hashes, moves to an
immutable content address, commits a manifest, registers lineage, and only then
marks work complete. Resume verifies manifests and hashes.

Cache keys include block semantic/version and code fingerprint, resolved
parameters, input/model hashes, serializer and environment versions, seed and
declared external dependencies. Runtime IDs are excluded. A code/input change
naturally invalidates the affected node and all downstream keys.

The first interrupt requests graceful cancellation and a safe point. A repeated
signal raises hard termination. CLI/API resume validates durable state and
reuses completed blocks/partitions; it is not an unconditional script rerun.
