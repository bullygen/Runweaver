# Compatibility matrix

| Legacy workflow | New workflow | Inputs | Outputs | Status / allowed difference |
|---|---|---|---|---|
| `smbh-cv-run all` | `python tutorials/12_migrating_article_workflows.py --protocol article_v1` | seed, image count, protocol | typed stage outputs, metrics, artifacts | Architectural parity; reduced image algorithm |
| article v2 runner | same tutorial with `--protocol article_v2` | conditional Sobol space | immutable plans, ranking, refinement | Planning semantics retained; reduced trial count |
| per-stage worker flags | `Pipeline.then(..., parallelism=N)` | node-local count | ordered results | Equivalent mechanism |
| stage-file resume | durable `LocalExecutor(..., resume=True)` | SQL URL + artifact root | only missing partitions rerun | Stronger integrity semantics |
| `run_signature.json` | `TrialPlan.fingerprint` and cache lineage | resolved plan/input/code | deterministic SHA-256 | Stronger and inspectable |
| ranking JSON | `DecisionRecord` | `ExperimentHistory` | immutable selected IDs/explanation | Policy separated from metrics |
| local candidate generation | `EliteZoomStrategy` / `NeighborhoodGridStrategy` | selected plans | versioned child space | Original space is not mutated |
| stage summary JSON | SQL event log + CLI export | run IDs | structured events/export JSON | Different layout |
| generated detector plots | tutorial summary JSON | small synthetic images | metrics/artifacts | Plot parity intentionally not retained |

Exact D1–D6 numerical output compatibility is marked **not retained** following
the explicit authorization to replace the prototype. Tutorial 12 is the
supported migration example and is tested as an executable program.
