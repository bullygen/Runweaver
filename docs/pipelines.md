# Pipeline construction

`.then(block)` creates a sequence. `.add(block, id=..., depends_on=(...))`
creates an arbitrary DAG; `.join()` selects several predecessors for the next
node. Fan-in merges non-colliding output fields.

Node options include `map_over`, `parallelism` and an optional resource override.
Execution patterns are `SINGLE`, `MAP`, `BATCH`, `REDUCE`, `MAP_REDUCE`,
`SERVICE_CALL` and `SUBPROCESS`. `STREAM` is deliberately rejected until
backpressure/checkpoint behavior is supplied.

Validation covers unique IDs, dependency existence, cycles, Pydantic field
compatibility, map fields and unsupported semantics. Reusable subpipelines can
be constructed as functions returning a validated `Pipeline`.
