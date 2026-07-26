# Modular SMBH CV pipeline

This refactor keeps the explicit D1-D6 decomposition and separates data generation and fit statistics.

Typical one-command run:

```bash
python smbh_cv_run.py all --out runs/demo --clean --n-images 71 \
  --workers-generate 2 --workers-d1 4 --workers-d2 4 --workers-d3 1 --workers-d4 4 --workers-d5 2 --workers-d6 4 \
  --d1-edge-method log_mad --d1-threshold-method mad --d1-skeletonize \
  --d2-method sparse_rht_stratified --d2-vote-weight mean \
  --d3-method peak_nms --d4-method profile_harmonic \
  --d5-loss soft_l1 --d5-residual-weighting ring_support \
  --d6-method merge_delta_bic
```

Run a single stage after intermediate files already exist:

```bash
python smbh_cv_run.py d3 --out runs/demo --workers-d3 1 --d3-method ph_safe_proxy
```

The pipeline writes intermediate data under `D1_preprocess`, `D2_hough`, `D3_candidates`, `D4_initialization`, `D5_fit`, and `D6_prune`. Each worker process handles one image and exits when process pools use multiple workers, which limits accumulation from native-library leaks.

## Visualization stage

The refactor includes a separate memory-safe visualization stage. It reads saved artifacts from `images/`, `D1_preprocess/`, `D2_hough/`, `D3_candidates/`, `D4_initialization/`, `D5_fit/`, and `D6_prune/` and writes figures to `visualization/<image_id>/`.

Run it after any completed pipeline:

```bash
python smbh_cv_run.py viz --out runs/demo --workers-viz 1 --viz-dpi 140 --viz-max-accumulator-points 5000
```

Or request it as part of a full run:

```bash
python smbh_cv_run.py all --out runs/demo --clean --n-images 1 --make-plots --workers-viz 1
```

The generated figures are:

- `00_generation.png`: generated image, clean image, pre-degradation clean image and noise map;
- `01_D1_preprocess.png`: original image, edge response, binary/skeleton map and edge overlay;
- `02_D2_hough.png`: sparse RHT accumulator projections and score distributions;
- `03_D3_candidates.png`: D3 candidate rings and scores;
- `04_D4_initialization.png`: initial ring model and residual;
- `05_D5_fit.png`: fitted model, residual and residual weights;
- `06_D6_final.png`: final model, residual and pruning/merge decisions;
- `pipeline_overview.png`: compact figure for stage-by-stage demonstration.
