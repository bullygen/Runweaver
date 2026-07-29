# Tracking and observability

Structured runtime fields are experiment, round, trial, pipeline, block,
partition, attempt, backend and event type. SQL events record transitions,
retries and leases. Results include cache hits and resumed partition IDs.

MLflow is an optional sink for flattened parameters, full JSON plans, scalar
metrics and complex-metric/artifact references. Library IDs are tags. Turning
tracking off never changes computation.

Prefect UI shows orchestration state. The domain store remains authoritative.
OpenTelemetry can be added as an optional event sink; it is not required
locally.
