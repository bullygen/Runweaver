# Tutorial execution and expected results

Run from an editable installation. All scripts return exit code zero; optional
extras print a clear `SKIP` line when absent.

| Tutorial | Command | Expected evidence |
|---|---|---|
| 1 | `python tutorials/01_minimal_sequential_pipeline.py` | equal ephemeral/durable MSE |
| 2 | `python tutorials/02_parallel_partitions.py --backend processes` | process-parallel injected failure, ordered fan-in, reused partition IDs |
| 3 | `python tutorials/03_custom_contracts_and_serializers.py` | translated points and a preflight port error |
| 4 | `python tutorials/04_pytorch_training.py` | accuracy/learning curve or optional-extra skip |
| 5 | `python tutorials/05_pause_and_resume.py` | paused state then squared values without recomputing early partitions |
| 6 | `python tutorials/06_latin_hypercube_design.py` | 12-row design CSV |
| 7 | `python tutorials/07_optuna_adaptive_hpo.py` | persistent study restart or optional-extra skip |
| 8 | `python tutorials/08_elite_zoom_round.py` | narrower versioned bounds and 12 next-round points |
| 9 | `python tutorials/09_multi_objective.py` | Pareto trial IDs |
| 10 | `python tutorials/10_custom_plugins.py` | custom block, plans, decision and child bounds |
| 11 | `python tutorials/11_prefect_ray_mlflow.py` | backend-neutral result; select optional backend with `--backend` |
| 12 | `python tutorials/12_migrating_article_workflows.py --protocol both` | four completed trials, one selection and one refined space for each protocol |

Notebooks are paired through Jupytext and contain the same executable cells.
