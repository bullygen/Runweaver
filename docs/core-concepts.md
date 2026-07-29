# Core concepts

`DataRef` and `ModelRef` point to materialized values with a schema version,
semantic role and content hash. Large payloads live outside SQL.

A `Block` consumes and returns Pydantic models. `BlockSpec` declares version,
role, determinism, idempotency, cache/retry/checkpoint policy, side effects,
serializer and resource requirements.

A `Pipeline` is a DAG. `.then()` is the linear convenience API; `.add()` and
dependency IDs build branches and fan-in. Validation checks cycles, required
fields, map cardinality and unsafe policy combinations before execution.

An `Experiment` contains identity and objectives. A planner produces immutable
`TrialPlan` values; a run never lets a backend mutate them. Observations are
`MetricRecord` values. Selection is an immutable `DecisionRecord`, and
refinement returns a child `ParameterSpace` version.

The state hierarchy is Experiment → planning round → trial → pipeline → block
→ partition → checkpoint. Artifact lineage ties outputs back to code,
parameters, input hashes, environment and serializers.
