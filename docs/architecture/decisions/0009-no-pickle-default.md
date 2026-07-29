# ADR 0009: Unsafe pickle is disabled

Status: accepted

## Context

Unpickling can execute code and is unsuitable for untrusted remote artifacts.

## Decision

Core ships JSON, bytes, text and non-pickle NumPy serializers. A future unsafe
pickle plugin must be explicit and visibly marked.

## Alternatives

Use pickle/cloudpickle for every object.

## Consequences

Durable schemas are more explicit. Complex users provide safe serializers or
accept a deliberate trust-boundary plugin.
