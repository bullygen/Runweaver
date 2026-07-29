# ADR 0002: Ray is an optional executor

Status: accepted

## Context

Local use must work without a cluster, while large experiments need
resource-aware distributed execution.

## Decision

Ray Core and Prefect-Ray live in the `ray` extra. Domain
`ResourceRequirements` are translated to Ray options, and `ObjectRef` never
leaves the adapter.

## Alternatives

Require Ray for every run; make Ray the durable store.

## Consequences

Core imports stay small and cross-platform. Distributed operation has an
explicit installation/runtime boundary.
