# ADR 0007: Separate planning, decision and refinement

Status: accepted

## Context

Generating candidates, measuring them, selecting them and changing the search
space are different policies with different evidence and versioning.

## Decision

Use independent `ExperimentPlanner`, `DecisionPolicy` and
`RefinementStrategy` contracts. Decisions and child spaces are immutable.

## Alternatives

One adaptive optimizer object that owns all experiment semantics.

## Consequences

Policies are testable and composable, including non-Optuna DOE workflows.
Applications must explicitly wire the loop.
