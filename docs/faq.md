# FAQ and troubleshooting

## Why another database when Prefect/MLflow have one?

Their records answer orchestration/tracking questions. Runweaver owns immutable
plans, domain transitions, partition leases, artifact commit order and decision
history.

## Why did pipeline validation reject my two schemas?

The immediate upstream output must provide every required downstream input
field with a compatible annotation. Use a small adapter block or fan-in model.

## Why cannot my process block be serialized?

Use a top-level function/class and serializable fields, or select threads.
Large values should be artifact references.

## Why was a checkpoint rejected?

Code, parameters, input lineage or serializer compatibility changed. Inspect
the lineage and start the affected block again instead of forcing unsafe state.

## Does resume repeat completed work?

No. It verifies committed artifacts and restores completed blocks/partitions.
Missing, stale or corrupt work is recomputed only when policy allows.
