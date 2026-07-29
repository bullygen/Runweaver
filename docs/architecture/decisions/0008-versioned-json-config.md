# ADR 0008: Versioned JSON configuration

Status: accepted

## Context

Experiments need portable, reviewable snapshots and predictable migrations.

## Decision

JSON is guaranteed and validated by a versioned Pydantic schema. Environment
interpolation is restricted to operational store settings. Secrets are named
references.

## Alternatives

Arbitrary Python configs; unversioned YAML.

## Consequences

Snapshots are reproducible and safe to inspect. Advanced block implementations
are registered through entry points rather than executed from config paths.
