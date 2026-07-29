# Runweaver architecture

Runweaver is a domain layer for typed computational pipelines and iterative
experiments. It composes mature execution, optimization and tracking systems;
it does not replace their numerical or operational capabilities.

```mermaid
flowchart TB
  U[Python API / versioned JSON / CLI] --> D[Domain: blocks, DAG, plans, metrics, decisions]
  D --> E[Execution service]
  E --> L[Local sync / thread / process]
  E --> P[Prefect adapter]
  E --> R[Ray adapter]
  D --> S[(SQLAlchemy domain store)]
  D --> A[(fsspec artifact store)]
  D --> O[Optuna planner adapter]
  D --> M[MLflow tracking sink]
  D --> T[PyTorch helpers]
  P -. runtime states .-> S
  M -. display only .-> S
```

The dependency rule is strict: core modules do not import a user application,
PyTorch, Ray, Optuna or MLflow. Optional adapters import domain contracts.
Backend-native state and handles are translated before reaching the public API.

## Domain boundaries

- A `Block` owns typed computation and declares determinism, idempotency,
  resources, retry/cache/checkpoint policy and side effects.
- A `Pipeline` owns a validated DAG, node-local parallelism and fan-in/out.
- A `TrialPlan` is immutable after a planner materializes it.
- Metrics describe observations. `DecisionPolicy` selects. A
  `RefinementStrategy` returns a new parameter-space version.
- The SQL store is authoritative for experiment, trial, pipeline, block and
  partition state. Prefect and MLflow are projections.
- The artifact store owns large payloads, content hashes, commit manifests and
  lineage; SQL stores references and searchable metadata.

## Execution sequence

```mermaid
sequenceDiagram
  participant User
  participant Engine
  participant State as Domain DB
  participant Store as Artifact store
  User->>Engine: run(pipeline, plan, resume=True)
  Engine->>State: create/read deterministic run hierarchy
  loop topological blocks
    Engine->>State: acquire PLANNED→QUEUED→RUNNING
    Engine->>Store: verify resume/cache artifact
    alt valid artifact
      Engine->>State: CACHED/COMPLETED
    else compute
      Engine->>Engine: execute missing partitions only
      Engine->>Store: temp write → hash → object → manifest
      Engine->>State: register artifact + COMPLETED
    end
  end
  Engine-->>User: backend-neutral PipelineResult
```

## State machine

```mermaid
stateDiagram-v2
  [*] --> PLANNED
  PLANNED --> QUEUED
  QUEUED --> RUNNING
  RUNNING --> COMPLETED
  RUNNING --> CACHED
  RUNNING --> RETRY_WAIT
  RETRY_WAIT --> QUEUED
  RUNNING --> PAUSING
  PAUSING --> PAUSED
  PAUSED --> QUEUED
  RUNNING --> FAILED
  FAILED --> QUEUED
  RUNNING --> STALE
  STALE --> QUEUED
  STALE --> ORPHANED
  RUNNING --> CANCELLED
  RUNNING --> PRUNED
```

All transitions are centralized and persisted with an optimistic version
column. Running workers can hold expiring leases and emit heartbeats.
Reconciliation marks expired work `STALE`; recovery policy decides whether it
is safe to requeue.

## Artifact commit

```mermaid
flowchart LR
  W[write temporary URI] --> F[flush/fsync when available]
  F --> H[SHA-256]
  H --> O[move/copy immutable object]
  O --> M[commit manifest marker]
  M --> D[register reference in SQL transaction]
  D --> C[mark run COMPLETED]
```

The cache key contains semantic block/version, code fingerprint, resolved
parameters, upstream hashes, model hash when present, serializer/environment
versions, seed and declared external dependencies. Runtime IDs are excluded.

## Pause and resume

```mermaid
flowchart TD
  S[SIGINT / API cancellation] --> N[stop scheduling new work]
  N --> P[block reaches safe point]
  P --> K[commit checkpoint / completed partitions]
  K --> X[PAUSED]
  X --> V[validate code, inputs, manifests]
  V -->|compatible| R[requeue missing work]
  V -->|incompatible| E[actionable compatibility error]
```

## Global-to-local experiment loop

```mermaid
flowchart LR
  G[Grid / random / LHS / Sobol / Halton / Optuna] --> T[immutable TrialPlans]
  T --> M[metrics + uncertainty]
  M --> D[selection / Pareto / feasibility]
  D --> R[versioned refinement]
  R --> T
```
