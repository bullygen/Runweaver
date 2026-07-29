# Prototype audit

This audit describes the SMBH CV prototype that occupied the repository before
Runweaver extraction. The source was inspected at commit `f3f0879`; it was
removed only after its observable contracts had been recorded here and its
portable experiment shape had been moved to Tutorial 12.

## 1. Package and module map

| Area | Former modules | Classification |
|---|---|---|
| Synthetic scenes | `generation.py`, `models.py`, `article_data.py` | `DOMAIN_COMPUTATION`, `I/O` |
| Preprocessing | `d1_preprocess.py` | `DOMAIN_COMPUTATION`, `I/O` |
| Sparse voting | `d2_hough.py` | `DOMAIN_COMPUTATION`, `I/O` |
| Candidate selection | `d3_candidates.py` | `DOMAIN_COMPUTATION`, `DECISION`, `I/O` |
| Initialization and fitting | `d4_initialization.py`, `d5_fit.py` | `DOMAIN_COMPUTATION`, `I/O` |
| Pruning | `d6_prune.py` | `DOMAIN_COMPUTATION`, `DECISION`, `I/O` |
| Evaluation | `statistics.py`, `quality_gate.py` | `EVALUATION`, `DECISION` |
| Visualization | `visualization.py` | `VISUALIZATION`, `I/O` |
| Pipeline driver | `cli.py` | `ORCHESTRATION`, `STATE_MANAGEMENT`, `CLI` |
| Experiment drivers | `experiment_runner.py`, `experiment_v2.py` | `PLANNING`, `ORCHESTRATION`, `DECISION`, `STATE_MANAGEMENT` |
| Configuration | `config.py`, JSON files under `experiments/` | `STATE_MANAGEMENT` |
| Legacy launchers | `smbh_cv_*.py` | `LEGACY_GLUE`, `CLI` |

Dependency direction was mostly stage module → config/I/O. The CLI imported
every stage and bound them to `STAGE_FUNCS`; experiment drivers imported that
map and its multiprocessing helper. This made domain functions reusable, but
made execution, status files, retry behavior and scheduling inseparable from
the SMBH package.

## 2. Entry points

The root `pyproject.toml` exposed:

- `smbh-cv-run`;
- `smbh-cv-dataset`;
- `smbh-cv-experiments`;
- `smbh-cv-experiments-v2`;
- `smbh-cv-quality`.

Equivalent root scripts called the same `main()` functions. No notebooks were
present. Search plans in `experiments/article_v1` and `article_v2` were the
experiment entry configurations.

## 3. Actual stage order and contracts

The normal path was:

`generate → D1 → D2 → D3 → D4 → D5 → D6 → statistics → visualization`.

Every per-image stage accepted `(PipelineConfig as dict, image_id)` and returned
a small status dictionary. The real inter-stage values were files:

| Stage | Input | Output |
|---|---|---|
| Generate | seed and scene config | `image.npy`, truth/config JSON |
| D1 | image array | `d1_edges.npz` |
| D2 | D1 edge pixels/weights | `d2_sparse_accumulator.npz` |
| D3 | sparse accumulator | `d3_candidates.json` |
| D4 | image + candidates | `d4_initial_params.json` |
| D5 | image + initialization | `d5_fit.json` |
| D6 | fitted candidates | `d6_final.json` |
| Statistics | truth + D6 | JSON/CSV metrics |

The migration tutorial models the same roles with explicit Pydantic values and
artifact references. It intentionally uses a small NumPy/SciPy implementation;
it is an architectural migration example, not a reproduction of the removed
scientific detector.

## 4. State and intermediate storage

`run_paths()` created fixed domain directories. JSON writes used a temporary
file plus `os.replace`, which was a useful local atomic-write baseline.
Experiment continuation inferred completion from the existence of one expected
stage file per image. `run_signature.json` guarded against mixing a changed
dataset/configuration into a run directory.

This mechanism had no transactional hierarchy, optimistic locking, leases,
artifact manifests or hash verification. A present but truncated stage file
could be treated as complete. Concurrent coordinators had no authoritative
owner. Runweaver replaces these responsibilities with SQL state and committed
content-addressed artifacts.

## 5. Parallelism and resources

`ProcessPoolExecutor` with the spawn context performed per-image work.
Parallelism was configured separately for generation, D1–D6 and visualization.
`max_tasks_per_child=1` bounded retained native-library memory. A D3-specific
RAM estimate capped unsafe worker counts.

The portable mechanisms are per-node parallelism, partition identity, ordered
collection, process isolation, resource declarations and concurrency limits.
The D3 memory heuristic was domain-specific and belongs in a block resource
declaration or adapter.

## 6. Determinism and hidden assumptions

- Generation used `seed + image_index`.
- D2 used configuration seed streams for random triplets.
- Candidate plans were deterministic for a fixed search seed.
- Stages reconstructed a mutable dataclass config from a dictionary.
- Paths, stage names, image identifiers and output filenames were hard-coded.
- Images were assumed to be NumPy arrays and most stages assumed a 2D geometry.
- Evaluation read truth files directly; planning and decision code knew article
  metric names and feasibility thresholds.

There was no hidden singleton coordinator. Mutable filesystem state acted as
the global state. No GPU or distributed framework was used.

## 7. Planning, evaluation and decisions

Article v1 generated one baseline plus independently shuffled stratified
samples (an LHS-like design). It ranked candidates with a robust F1-based
objective, feasibility floors and false-positive penalty, then froze the best
configuration.

Article v2 added Sobol global designs, conditional parameters, adaptive/local
designs, continuous shortfall penalties and Pareto-related ranking data.
Planning, metric aggregation and promotion/freeze decisions lived in the same
experiment module.

Runweaver separates these as `ParameterSpace`, immutable `TrialPlan`,
`ExperimentPlanner`, `MetricRecord`, `DecisionPolicy` and
`RefinementStrategy`.

## 8. Failure and resume behavior

- A stage exception failed the current process/command.
- Completed stage files remained.
- A later experiment command computed `_missing_stage_image_ids` and processed
  only missing images.
- `--restart` deleted the selected run directory.
- Signature mismatch blocked continuation.
- Stage summaries separated pending and pre-existing image counts.

There was no checkpoint inside a fitting block, cooperative signal safe point,
stale worker reconciliation or non-idempotent recovery policy.

## 9. Characterization baseline

Before extraction:

- `python3 -m unittest discover -s tests -v` failed at import because the
  environment lacked `scikit-image`; both collected test modules reported that
  missing dependency.
- Direct generation remained runnable without that optional import.
- A fixed small call with seed `239`, one 32×32 image and one true ring returned
  `{"image_id": "image_0000", "status": "ok", "true_n": 1}` and wrote finite
  NumPy arrays and JSON into a temporary run.
- Existing tests specified deterministic candidate generation, strict JSON
  replacement of non-finite values, stage-file resume and geometric metrics.

These facts are evidence, not a claim that the removed detector was fully
validated in this checkout.

## 10. Compatibility contract retained

The retained contract is architectural:

- ordered generation/preprocess/process/evaluate roles;
- independent per-stage parallelism;
- fixed seeds;
- resumable per-item work;
- immutable experiment designs;
- conditional parameters;
- selection followed by local refinement;
- strict JSON;
- observable artifacts and metrics.

Exact numerical SMBH detector parity is not retained. The user explicitly
authorized replacement of the local prototype and requested a library-only
repository with article workflows represented by an executable tutorial.
