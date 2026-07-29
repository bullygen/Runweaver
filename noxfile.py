"""Repeatable development and release sessions."""

from __future__ import annotations

import os

import nox

nox.options.sessions = ["lint", "types", "tests", "docs", "tutorials", "build"]


@nox.session
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".")


@nox.session(python="3.12")
def types(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("mypy")


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "--cov=runweaver", "--cov-report=term", "-q")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="3.12")
def tutorials(session: nox.Session) -> None:
    session.install("-e", ".[docs,optuna]")
    environment = {
        "JUPYTER_CONFIG_DIR": os.path.join(session.create_tmp(), "jupyter-config"),
        "JUPYTER_DATA_DIR": os.path.join(session.create_tmp(), "jupyter-data"),
        "JUPYTER_RUNTIME_DIR": os.path.join(session.create_tmp(), "jupyter-runtime"),
    }
    session.run(
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=180",
        "--output-dir",
        session.create_tmp(),
        *[f"tutorials/{index:02d}_{name}.ipynb" for index, name in TUTORIALS],
        env=environment,
    )


@nox.session(python="3.12")
def build(session: nox.Session) -> None:
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


TUTORIALS = [
    (1, "minimal_sequential_pipeline"),
    (2, "parallel_partitions"),
    (3, "custom_contracts_and_serializers"),
    (4, "pytorch_training"),
    (5, "pause_and_resume"),
    (6, "latin_hypercube_design"),
    (7, "optuna_adaptive_hpo"),
    (8, "elite_zoom_round"),
    (9, "multi_objective"),
    (10, "custom_plugins"),
    (11, "prefect_ray_mlflow"),
    (12, "migrating_article_workflows"),
]
