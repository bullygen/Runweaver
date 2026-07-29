# CLI reference

```text
runweaver validate CONFIG
runweaver schema --output SCHEMA
runweaver plan CONFIG --output PLAN
runweaver run CONFIG [--resume]
runweaver status RUN_ID
runweaver inspect trial TRIAL_ID
runweaver pause RUN_ID
runweaver resume CONFIG
runweaver retry RUN_ID
runweaver reconcile [--requeue]
runweaver cancel RUN_ID
runweaver artifacts verify RUN_ID
runweaver lineage ARTIFACT_ID
runweaver export RUN_ID --output EXPORT
```

Success is exit 0. Validation/configuration errors and failed artifact
verification are nonzero. Commands accept explicit database/artifact settings
where they inspect an existing durable deployment.
