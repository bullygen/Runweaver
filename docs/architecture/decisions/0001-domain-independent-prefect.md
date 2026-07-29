# ADR 0001: Domain model independent of Prefect

Status: accepted

## Context

Prefect has excellent orchestration states, but an experiment also needs
immutable plans, scientific metrics, decisions, partition lineage and recovery
rules that must remain stable across backends.

## Decision

Runweaver domain objects and exceptions are Pydantic/Python types. The Prefect
adapter translates flows, run IDs and failures at the boundary.

## Alternatives

Expose Prefect task/flow states directly; build an orchestration engine from
scratch.

## Consequences

Blocks run locally or under another backend unchanged. Reconciliation code is
required, and some Prefect-native details remain adapter-only.
