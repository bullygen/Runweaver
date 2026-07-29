# Evaluation, metrics, selection and refinement

`MetricRecord` supports scalar/vector or artifact-backed values, direction,
step, split, aggregation, uncertainty, unit, tags and provenance. Objectives,
diagnostics and constraints remain observations; they do not make decisions.

Policies include top-k, best feasible, thresholds and Pareto fronts. Applications
can express instability rejection, promotion, replication requests and stop
decisions using the same immutable record contract.

`EliteZoomStrategy` uses elite/top-k samples, quantile bounds, expansion,
minimum width, global bounds and categorical retention. It records a global
exploration fraction to avoid collapse. Neighborhood, trust-region, replication
and robustness strategies also return new space versions rather than mutating
history.
