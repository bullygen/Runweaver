# Changelog

All notable changes follow Semantic Versioning.

## 0.1.0 - 2026-07-29

- Extracted a neutral typed pipeline and experiment domain model.
- Added local sync/thread/process execution and partition-level durable resume.
- Added SQLAlchemy/Alembic state, fsspec artifacts, cache and lineage.
- Added DOE planners, decisions and refinement strategies.
- Added optional Prefect, Ray, Optuna, MLflow and PyTorch adapters.
- Added versioned JSON configuration, Typer CLI, MkDocs and twelve tutorials.
- Replaced the former SMBH prototype with an executable article v1/v2
  migration tutorial under the user's explicit replacement authorization.

## Versioning policy

Public root exports are stable within a minor release. Experimental APIs live
outside the package root. Deprecations warn for at least one minor release.
Configuration and database schemas carry independent versions and migrations.
