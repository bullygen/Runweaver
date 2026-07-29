from __future__ import annotations

from pathlib import Path

import pytest
from runweaver import LocalExecutionConfig, MaterializationMode


@pytest.fixture
def durable_config(tmp_path: Path) -> LocalExecutionConfig:
    return LocalExecutionConfig(
        materialization=MaterializationMode.DURABLE,
        work_dir=tmp_path / "work",
        artifact_root=str(tmp_path / "artifacts"),
        state_database_url=f"sqlite:///{tmp_path / 'state.db'}",
        install_signal_handlers=False,
    )
