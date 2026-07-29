# Runweaver

Runweaver supplies the experiment-and-pipeline domain layer between typed user
computations and mature orchestration, optimization and tracking components.
Its focus is a reproducible lifecycle: generate, process, evaluate, select,
refine, stop safely and resume only missing work.

Use it when stages have explicit contracts, trials must be planned immutably,
partial work is expensive, or experiment decisions need provenance. A plain
function call is preferable for a tiny one-off computation; a framework-native
trainer is preferable when training-loop abstraction is the only need.

Runweaver complements:

- PyTorch/Lightning for tensor computation and training loops;
- Prefect for orchestration and UI state;
- Ray for distributed resources;
- Optuna for adaptive suggestions/pruning;
- MLflow for comparison and visualization.

The SQL domain store remains authoritative for experiment semantics and
partition recovery. MLflow and Prefect are projections, never the only source
of resume truth.
