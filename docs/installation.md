# Installation

## Minimal local environment

```bash
python -m venv .venv
.venv/bin/pip install runweaver
```

Editable development:

```bash
.venv/bin/pip install -e '.[dev,docs]'
```

Extras are `pytorch`, `optuna`, `ray`, `mlflow`, `s3`, `docs` and `dev`.
Python 3.11–3.13 is supported. Local sync/thread/process execution targets
Linux, macOS and Windows. Distributed/GPU operation is supported on Linux.

No Docker or external server is required locally. A production deployment can
configure PostgreSQL, an S3-compatible fsspec URI, Prefect server and Ray
cluster independently.
