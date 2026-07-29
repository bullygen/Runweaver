# ADR 0006: Core does not depend on PyTorch

Status: accepted

## Context

Many computational pipelines use NumPy, external services or non-ML
algorithms, while PyTorch is large and platform-sensitive.

## Decision

Core models data, processing state and checkpoints generically. PyTorch blocks,
TorchMetrics and state-dict/RNG helpers live in the `pytorch` extra.

## Alternatives

Make every block an `nn.Module`; require a universal trainer.

## Consequences

The same pipeline layer supports multiple frameworks. Runweaver does not try to
replace Lightning or user-owned training loops.
