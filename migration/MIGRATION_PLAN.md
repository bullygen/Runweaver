# Migration plan

## Strangler sequence

1. Record the former stage graph, I/O files, deterministic seeds, planning
   rules, worker controls and resume behavior.
2. Introduce Runweaver without importing the SMBH package.
3. Characterize the reusable contract with toy and failure-injection tests.
4. Move orchestration to typed blocks, a validated DAG, immutable trial plans,
   SQL state and fsspec artifacts.
5. Recreate article v1/v2 as a small, executable image-processing tutorial
   using only public Runweaver imports.
6. Remove the legacy package, launchers, configs and generated scientific
   material after the tutorial and library checks pass.
7. Keep exact scientific detector parity out of the release claim; it was not
   requested after the scope change and was not fully runnable at baseline.

## Component mapping

| Prototype mechanism | Runweaver abstraction |
|---|---|
| `PipelineConfig` dataclass | versioned Pydantic `ExperimentConfig` |
| `STAGE_FUNCS` | registered typed `Block` implementations |
| loop over D1–D6 | `Pipeline` DAG compilation |
| `_pool_map` | local thread/process partition executor |
| stage output existence | `PartitionRun` state + committed artifact manifest |
| `run_signature.json` | trial, code, input and environment fingerprints |
| candidate dictionaries | immutable `TrialPlan` |
| v1 LHS-like sampling | SciPy `LatinHypercubePlanner` |
| v2 Sobol samples | SciPy `SobolPlanner` |
| ranking/freeze | `DecisionPolicy` |
| local candidates | versioned `RefinementStrategy` |
| summary JSON/CSV | `MetricRecord` + tracker adapters |
| domain directories | content-addressed `FsspecArtifactStore` |

## Removal gate

Legacy code can be removed when:

- the public package imports without optional ML integrations;
- the local sequential and process-map paths pass;
- durable resume reuses completed partitions;
- article v1/v2 tutorial runs from a clean temporary directory;
- the migration documents state the numerical-parity limitation.
