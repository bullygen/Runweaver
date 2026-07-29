# Implementation report

## Result

The repository was migrated from the application-specific SMBH CV D1–D6
prototype to **Runweaver 0.1.0**, a neutral Python library for typed, durable
computational pipelines and iterative ML experiments. The old package, command
wrappers, experiment configurations, tests and article protocol documents were
removed. Their supported concepts are represented by the new library,
documentation and executable tutorial 12.

Runweaver deliberately composes Pydantic v2, SQLAlchemy/Alembic, fsspec,
NumPy/SciPy, Prefect 3 and Typer. PyTorch/TorchMetrics, Optuna, Ray/Prefect-Ray,
MLflow and S3 remain explicit extras instead of being reimplemented.

## Implemented surface

- immutable Pydantic domain contracts for blocks, ports, resources, artifacts,
  metrics, plans, experiments, decisions, histories and checkpoints;
- typed linear and DAG pipeline builder with cycle, dependency and port
  validation;
- synchronous, thread and process local execution, node-local mapping, stable
  fan-in, retries, cooperative cancellation and pause/resume;
- ephemeral and durable modes behind the same block API;
- SQLAlchemy run hierarchy, transition validation, optimistic versioning,
  leases, heartbeats, stale reconciliation, events, metrics and checkpoints;
- fsspec content-addressed artifacts with SHA-256 manifests, temporary writes,
  commit markers, corruption checks and lineage-aware cache keys;
- deterministic grid, random, full-factorial, LHS, Sobol, Halton and imported
  design-matrix planners, plus an optional persistent Optuna adapter;
- selection, threshold, feasibility, Pareto and stability policies separated
  from metrics and refinement;
- elite zoom, neighborhood, trust-region, replication and robustness
  refinement strategies that create versioned child spaces;
- backend-neutral adapters for Prefect, Ray, MLflow and PyTorch;
- versioned JSON configuration, generated JSON Schema and a Typer CLI for
  validation, planning, execution, inspection, cancellation, retry,
  reconciliation and export;
- entry-point based block plugins and runtime registries for custom planners,
  decision policies, refinement strategies, serializers and tracking sinks.

## Verification

| Gate | Result |
|---|---|
| Unit/contract/integration/property/failure tests | 36 passed |
| Core branch coverage | 85.62%, threshold 80% |
| Domain models | 91% |
| Domain decisions | 90% |
| Ruff | passed |
| strict mypy core/contracts | passed, 14 source files |
| MkDocs strict build | passed after final link validation |
| Tutorial scripts | core paths and both article protocols passed |
| Tutorial notebooks | all 12 executed with `nbconvert` |
| CLI validate/plan/run/schema | passed |
| Wheel and sdist | built and checked |
| Clean wheel installation | CLI and import smoke-tested |

Coverage excludes CLI entry glue and optional adapter modules from the core
threshold. This is intentional: those surfaces have isolated smoke paths, while
their full behavior depends on external runtimes. Individual low-level graph
and execution modules remain below 90%; the stated 90% gate is met for the
aggregate domain contracts, not for every source file.

## Article workflow migration

`tutorials/12_migrating_article_workflows.py` and its paired notebook are the
supported image-processing example. They preserve the meaningful workflow
shape:

1. deterministic synthetic image generation;
2. D1 preprocessing;
3. D2 circular candidate detection;
4. D3 candidate construction;
5. D4 initialization;
6. D5 SciPy fitting;
7. D6 pruning;
8. metric reporting, selection and versioned local refinement.

`article_v1` uses an LHS plan and one-ring selection. `article_v2` uses a Sobol
plan, conditional parameters and up to two rings. Both execute as small,
self-contained studies with durable artifacts and state.

Exact numerical compatibility with the deleted detector was intentionally not
retained under the authorization to replace the prototype. The compatibility
matrix records this boundary.

## Operational boundaries

- SQLite plus local fsspec storage is the fully exercised reference runtime.
  PostgreSQL and remote object stores use the same contracts but require
  deployment-specific integration tests.
- Prefect remains the default orchestration dependency; Ray, Optuna, MLflow,
  PyTorch and S3 are extras. Missing extras produce an actionable error or a
  documented tutorial `SKIP`.
- The Ray adapter maps CPU/GPU/RAM/custom resource declarations but is not a
  cluster scheduler of its own.
- MLflow is a tracking projection. SQL state remains authoritative.
- Streaming is rejected until an application supplies explicit backpressure
  and checkpoint semantics.
- The package is marked alpha. Public contracts are curated in
  `runweaver.api`, but semantic-version compatibility begins with a later
  stable release.

## Release and reproduction

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,docs,optuna]'
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest --cov=runweaver --cov-report=term -q
.venv/bin/mkdocs build --strict
.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=180 tutorials/*.ipynb
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

The same gates are encoded in `noxfile.py` and `.github/workflows/ci.yml`.
