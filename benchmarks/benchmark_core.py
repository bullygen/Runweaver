"""Small reproducible local benchmark for Runweaver's own overheads."""

from __future__ import annotations

import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel
from runweaver import (
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    Pipeline,
    function_block,
)
from runweaver.artifacts import FsspecArtifactStore, fingerprint
from runweaver.execution import RunContext


class Values(BaseModel):
    values: list[float]


def identity(inputs: Values, context: RunContext) -> Values:
    return inputs


def measure(function: Callable[[], object], repeats: int = 7) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 4),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="runweaver-benchmark-") as directory:
        root = Path(directory)
        ephemeral = LocalExecutor(LocalExecutionConfig(
            materialization=MaterializationMode.EPHEMERAL,
            work_dir=root / "ephemeral-work",
            artifact_root=str(root / "ephemeral-artifacts"),
            install_signal_handlers=False,
        ))
        durable = LocalExecutor(LocalExecutionConfig(
            materialization=MaterializationMode.DURABLE,
            work_dir=root / "durable-work",
            artifact_root=str(root / "durable-artifacts"),
            state_database_url=f"sqlite:///{root / 'state.db'}",
            install_signal_handlers=False,
        ))
        pipeline = Pipeline("benchmark-identity").then(
            function_block(identity, inputs=Values, outputs=Values)
        )
        payload = Values(values=[float(index) for index in range(64)])

        ephemeral.run(pipeline, payload)
        durable.run(pipeline, payload)
        cache_result = durable.run(pipeline, payload)

        one_megabyte = bytes(1024 * 1024)
        artifact_store = FsspecArtifactStore(str(root / "object-store"))

        report = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "single_block_ephemeral": measure(lambda: ephemeral.run(pipeline, payload)),
            "durable_cache_lookup": measure(lambda: durable.run(pipeline, payload)),
            "fingerprint_64_scalars": measure(
                lambda: [fingerprint(payload) for _ in range(1_000)],
                repeats=5,
            ),
            "artifact_commit_1_mib": measure(
                lambda: artifact_store.put_bytes(
                    one_megabyte,
                    media_type="application/octet-stream",
                    serializer_id="bytes",
                ),
                repeats=5,
            ),
            "cache_hit_nodes": list(cache_result.cache_hits),
        }
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
