# ADR 0003: Own a domain StateStore

Status: accepted

## Context

Prefect records orchestration and MLflow displays experiments, but neither is
the authoritative model for immutable plans, partition resume, artifact commit
ordering or decision records.

## Decision

Use SQLAlchemy 2.x repositories and migrations as the domain source of truth.
Prefect and MLflow IDs are mappings/projections.

## Alternatives

Use Prefect or MLflow alone; retain JSON status files.

## Consequences

Transactions, optimistic locking and leases are available. Operators must run
schema migrations and reconciliation in production.
