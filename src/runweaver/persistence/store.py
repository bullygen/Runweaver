"""SQLAlchemy 2.x durable domain store with optimistic locking and leases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

from runweaver.domain.models import ArtifactRef, MetricRecord, RunKind, RunState, utc_now
from runweaver.domain.state import validate_transition
from runweaver.exceptions import StateTransitionError


class Base(DeclarativeBase):
    pass


class RunORM(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    logical_key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class EventORM(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ArtifactORM(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    cache_key: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    uri: Mapped[str] = mapped_column(Text)
    manifest_json: Mapped[str] = mapped_column(Text)
    valid: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "content_hash", name="uq_artifact_run_hash"),)


class MetricORM(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    scalar_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CheckpointORM(Base):
    __tablename__ = "checkpoints"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    block_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    compatible: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    __table_args__ = (UniqueConstraint("block_run_id", "sequence", name="uq_checkpoint_sequence"),)


class DecisionORM(Base):
    __tablename__ = "decisions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    logical_key: str
    kind: RunKind
    parent_id: str | None
    state: RunState
    version: int
    attempt: int
    metadata: Mapping[str, object]
    error: str | None
    worker_owner: str | None
    lease_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _record(row: RunORM) -> RunRecord:
    return RunRecord(
        id=row.id,
        logical_key=row.logical_key,
        kind=RunKind(row.kind),
        parent_id=row.parent_id,
        state=RunState(row.state),
        version=row.version,
        attempt=row.attempt,
        metadata=json.loads(row.metadata_json),
        error=row.error,
        worker_owner=row.worker_owner,
        lease_expires_at=row.lease_expires_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlAlchemyStateStore:
    """Source of truth for experiment/trial/block/partition state."""

    def __init__(self, database_url: str = "sqlite:///runweaver.db", *, echo: bool = False) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, echo=echo, future=True, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def create_run(
        self,
        *,
        logical_key: str,
        kind: RunKind,
        parent_id: str | None = None,
        state: RunState = RunState.PLANNED,
        metadata: Mapping[str, object] | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        with self.sessions.begin() as session:
            existing = session.scalar(select(RunORM).where(RunORM.logical_key == logical_key))
            if existing:
                return _record(existing)
            now = utc_now()
            row = RunORM(
                id=run_id or str(uuid4()),
                logical_key=logical_key,
                kind=kind.value,
                parent_id=parent_id,
                state=state.value,
                metadata_json=_json(metadata or {}),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            session.add(EventORM(
                run_id=row.id,
                event_type="run_created",
                payload_json=_json({"state": state.value, "kind": kind.value}),
            ))
            return _record(row)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.sessions() as session:
            row = session.get(RunORM, run_id)
            return _record(row) if row else None

    def get_by_key(self, logical_key: str) -> RunRecord | None:
        with self.sessions() as session:
            row = session.scalar(select(RunORM).where(RunORM.logical_key == logical_key))
            return _record(row) if row else None

    def transition(
        self,
        run_id: str,
        target: RunState,
        *,
        expected_version: int | None = None,
        error: str | None = None,
        metadata_update: Mapping[str, object] | None = None,
    ) -> RunRecord:
        with self.sessions.begin() as session:
            row = session.get(RunORM, run_id)
            if row is None:
                raise StateTransitionError(f"unknown run id: {run_id}")
            current = RunState(row.state)
            validate_transition(current, target)
            version = row.version if expected_version is None else expected_version
            metadata = json.loads(row.metadata_json)
            metadata.update(metadata_update or {})
            result = session.execute(
                update(RunORM)
                .where(RunORM.id == run_id, RunORM.version == version)
                .values(
                    state=target.value,
                    version=version + 1,
                    error=error,
                    metadata_json=_json(metadata),
                    updated_at=utc_now(),
                )
            )
            if result.rowcount != 1:
                raise StateTransitionError(
                    f"optimistic lock conflict for {run_id}; expected version {version}"
                )
            session.add(EventORM(
                run_id=run_id,
                event_type="state_transition",
                payload_json=_json({"from": current.value, "to": target.value, "error": error}),
            ))
            session.flush()
            return _record(session.get(RunORM, run_id))

    def add_event(self, run_id: str, event_type: str, payload: Mapping[str, object]) -> None:
        with self.sessions.begin() as session:
            session.add(EventORM(run_id=run_id, event_type=event_type, payload_json=_json(payload)))

    def acquire_lease(
        self,
        run_id: str,
        owner: str,
        *,
        duration_s: float = 60.0,
        now: datetime | None = None,
    ) -> bool:
        now = now or utc_now()
        expires = now + timedelta(seconds=duration_s)
        with self.sessions.begin() as session:
            row = session.get(RunORM, run_id)
            if row is None:
                return False
            if row.worker_owner and row.lease_expires_at and row.lease_expires_at > now:
                return row.worker_owner == owner
            row.worker_owner = owner
            row.lease_acquired_at = now
            row.heartbeat_at = now
            row.lease_expires_at = expires
            row.attempt += 1
            row.version += 1
            row.updated_at = now
            session.add(EventORM(
                run_id=run_id,
                event_type="lease_acquired",
                payload_json=_json({"owner": owner, "expires_at": expires.isoformat(), "attempt": row.attempt}),
            ))
            return True

    def heartbeat(self, run_id: str, owner: str, *, duration_s: float = 60.0) -> bool:
        now = utc_now()
        with self.sessions.begin() as session:
            row = session.get(RunORM, run_id)
            if row is None or row.worker_owner != owner:
                return False
            row.heartbeat_at = now
            row.lease_expires_at = now + timedelta(seconds=duration_s)
            row.updated_at = now
            return True

    def reconcile_stale(self, *, now: datetime | None = None) -> list[str]:
        now = now or utc_now()
        stale_ids: list[str] = []
        with self.sessions.begin() as session:
            rows = session.scalars(
                select(RunORM).where(
                    RunORM.state == RunState.RUNNING.value,
                    RunORM.lease_expires_at.is_not(None),
                    RunORM.lease_expires_at < now,
                )
            ).all()
            for row in rows:
                row.state = RunState.STALE.value
                row.version += 1
                row.updated_at = now
                stale_ids.append(row.id)
                session.add(EventORM(
                    run_id=row.id,
                    event_type="lease_stale",
                    payload_json=_json({"owner": row.worker_owner}),
                ))
        return stale_ids

    def report_metric(self, run_id: str, metric: MetricRecord) -> None:
        scalar = float(metric.value) if isinstance(metric.value, (int, float)) else None
        with self.sessions.begin() as session:
            session.add(MetricORM(
                run_id=run_id,
                name=metric.name,
                scalar_value=scalar,
                payload_json=metric.model_dump_json(),
            ))

    def metrics(self, run_id: str) -> tuple[MetricRecord, ...]:
        with self.sessions() as session:
            rows = session.scalars(
                select(MetricORM).where(MetricORM.run_id == run_id).order_by(MetricORM.id)
            ).all()
            return tuple(MetricRecord.model_validate_json(row.payload_json) for row in rows)

    def register_artifact(
        self,
        run_id: str,
        artifact: ArtifactRef,
        cache_key: str | None = None,
    ) -> None:
        with self.sessions.begin() as session:
            existing = session.scalar(select(ArtifactORM).where(ArtifactORM.id == str(artifact.id)))
            if existing:
                return
            session.add(ArtifactORM(
                id=str(artifact.id),
                run_id=run_id,
                cache_key=cache_key,
                content_hash=artifact.content_hash,
                uri=artifact.uri,
                manifest_json=artifact.model_dump_json(),
            ))

    def artifacts_for_run(self, run_id: str) -> tuple[ArtifactRef, ...]:
        with self.sessions() as session:
            rows = session.scalars(
                select(ArtifactORM)
                .where(ArtifactORM.run_id == run_id, ArtifactORM.valid == 1)
                .order_by(ArtifactORM.created_at)
            ).all()
            return tuple(ArtifactRef.model_validate_json(row.manifest_json) for row in rows)

    def find_cache(self, cache_key: str) -> ArtifactRef | None:
        with self.sessions() as session:
            row = session.scalar(
                select(ArtifactORM)
                .where(ArtifactORM.cache_key == cache_key, ArtifactORM.valid == 1)
                .order_by(ArtifactORM.created_at.desc())
            )
            return ArtifactRef.model_validate_json(row.manifest_json) if row else None

    def invalidate_cache(self, cache_key: str) -> int:
        with self.sessions.begin() as session:
            result = session.execute(
                update(ArtifactORM).where(ArtifactORM.cache_key == cache_key).values(valid=0)
            )
            return int(result.rowcount or 0)

    def events(self, run_id: str) -> list[dict[str, object]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(EventORM).where(EventORM.run_id == run_id).order_by(EventORM.id)
            ).all()
            return [
                {
                    "id": row.id,
                    "event_type": row.event_type,
                    "payload": json.loads(row.payload_json),
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def list_runs(self, *, parent_id: str | None = None) -> tuple[RunRecord, ...]:
        with self.sessions() as session:
            query = select(RunORM).order_by(RunORM.created_at)
            if parent_id is not None:
                query = query.where(RunORM.parent_id == parent_id)
            return tuple(_record(row) for row in session.scalars(query).all())

    def descendant_ids(self, run_id: str) -> tuple[str, ...]:
        descendants = [run_id]
        frontier = [run_id]
        while frontier:
            next_frontier: list[str] = []
            for parent in frontier:
                next_frontier.extend(record.id for record in self.list_runs(parent_id=parent))
            descendants.extend(next_frontier)
            frontier = next_frontier
        return tuple(descendants)

    def artifacts_under(self, run_id: str) -> tuple[ArtifactRef, ...]:
        run_ids = self.descendant_ids(run_id)
        with self.sessions() as session:
            rows = session.scalars(
                select(ArtifactORM)
                .where(ArtifactORM.run_id.in_(run_ids), ArtifactORM.valid == 1)
                .order_by(ArtifactORM.created_at)
            ).all()
            return tuple(ArtifactRef.model_validate_json(row.manifest_json) for row in rows)

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
        with self.sessions() as session:
            row = session.get(ArtifactORM, artifact_id)
            return ArtifactRef.model_validate_json(row.manifest_json) if row else None

    def save_checkpoint(self, block_run_id: str, payload_json: str) -> str:
        with self.sessions.begin() as session:
            count = len(session.scalars(
                select(CheckpointORM).where(CheckpointORM.block_run_id == block_run_id)
            ).all())
            checkpoint_id = str(uuid4())
            session.add(CheckpointORM(
                id=checkpoint_id,
                block_run_id=block_run_id,
                sequence=count + 1,
                payload_json=payload_json,
            ))
            return checkpoint_id

    def latest_checkpoint(self, block_run_id: str) -> str | None:
        with self.sessions() as session:
            row = session.scalar(
                select(CheckpointORM)
                .where(
                    CheckpointORM.block_run_id == block_run_id,
                    CheckpointORM.compatible == 1,
                )
                .order_by(CheckpointORM.sequence.desc())
            )
            return row.payload_json if row else None
