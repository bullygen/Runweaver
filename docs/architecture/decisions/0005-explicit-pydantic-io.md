# ADR 0005: Explicit Pydantic block I/O

Status: accepted

## Context

An `Any -> Any` block API moves schema failures deep into expensive runs.

## Decision

Every block declares named Pydantic input/output schemas and semantic metadata.
The DAG validates required fields before execution.

## Alternatives

Untyped dictionaries; framework-specific tensors as the universal contract.

## Consequences

Configuration and JSON Schema are machine-readable. Authors write small schema
classes and materialize large payloads as references.
