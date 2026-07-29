# Authoring blocks

Use `function_block` for a pure typed function and `BaseBlock` when state or
helper methods are useful. Blocks receive `RunContext`, not an ORM session,
Prefect context or Ray handle.

Context services include stable IDs, explicit seed streams, workdir, artifacts,
checkpoint manager, metric reporter, logger, cancellation token, allocated
resources, named secrets and a testable clock.

Long operations call `context.cancellation.raise_if_cancelled()` at safe
points. Stateful algorithms snapshot versioned cursor/RNG/model state through
the checkpoint manager. A map block consumes a declared list field; partitions
are independently persisted and collected in source order.

Declare every external side effect. Non-idempotent irreversible blocks cannot
enable automatic retry. `SERVICE_CALL` and `SUBPROCESS` are patterns, but an
application adapter must define safe authentication, argument vectors and
recovery semantics.
