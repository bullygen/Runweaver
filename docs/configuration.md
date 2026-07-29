# Configuration reference

`ExperimentConfig` version 1 contains experiment metadata, pipeline blocks and
plugins, parameters, planner, executor, storage, tracking, initial input,
stop conditions, observability and secret names.

JSON is guaranteed. Generate the full schema:

```bash
runweaver schema --output experiment.schema.json
```

Operational artifact/database settings may use `${ENV:NAME}`. Embedded
password/token/key fields are rejected unless they contain a `secret://name`
reference. Secret values never enter snapshots.

See `examples/experiment.json` for an annotated small graph.
