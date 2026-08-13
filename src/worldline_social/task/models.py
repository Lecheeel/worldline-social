"""Data models for the Worldline Social task workflow.

A task turns a user goal into a simulated worldline:

    decompose  -> research  -> architect  -> cast  -> configure  -> run

Every stage produces an artifact that is stored as JSON and can be edited
before the next stage consumes it. All models serialize to plain dicts so
they round-trip through SQLite and the studio API untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Stage order; a task always executes the next pending stage.
STAGE_ORDER = ("decompose", "research", "architect", "cast", "configure", "run")

#: Task lifecycle statuses.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_PAUSED = "paused"


@dataclass(frozen=True)
class ResearchQuestion:
    """One search query the research stage will run."""

    id: str
    question: str
    rationale: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "rationale": self.rationale}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResearchQuestion":
        return cls(
            id=str(value.get("id", "")),
            question=str(value.get("question", "")).strip(),
            rationale=str(value.get("rationale", "")).strip(),
        )


@dataclass(frozen=True)
class Decomposition:
    """Stage 1 artifact: the agent's breakdown of the user goal."""

    research_questions: tuple[ResearchQuestion, ...] = ()
    architecture_requirements: str = ""
    cast_requirements: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "research_questions": [item.to_mapping() for item in self.research_questions],
            "architecture_requirements": self.architecture_requirements,
            "cast_requirements": self.cast_requirements,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Decomposition":
        questions = []
        for index, item in enumerate(value.get("research_questions", ()) or ()):
            if isinstance(item, dict) and str(item.get("question", "")).strip():
                questions.append(ResearchQuestion.from_mapping(item))
        return cls(
            research_questions=tuple(questions),
            architecture_requirements=str(value.get("architecture_requirements", "")),
            cast_requirements=str(value.get("cast_requirements", "")),
        )


@dataclass(frozen=True)
class ResearchNote:
    """One web-search result saved as research material."""

    id: str
    query: str
    summary: str = ""
    sources: tuple[str, ...] = ()
    error: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "summary": self.summary,
            "sources": list(self.sources),
            "error": self.error,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResearchNote":
        return cls(
            id=str(value.get("id", "")),
            query=str(value.get("query", "")).strip(),
            summary=str(value.get("summary", "")).strip(),
            sources=tuple(str(item) for item in value.get("sources", ()) or ()),
            error=value.get("error"),
        )


@dataclass(frozen=True)
class WorldDesign:
    """Stage 3 artifact: the world architecture built from research."""

    title: str = ""
    background: str = ""
    event_script: str = ""
    world_rules: str = ""
    required_actors: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "background": self.background,
            "event_script": self.event_script,
            "world_rules": self.world_rules,
            "required_actors": list(self.required_actors),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorldDesign":
        return cls(
            title=str(value.get("title", "")).strip(),
            background=str(value.get("background", "")).strip(),
            event_script=str(value.get("event_script", "")).strip(),
            world_rules=str(value.get("world_rules", "")).strip(),
            required_actors=tuple(
                str(item).strip() for item in value.get("required_actors", ()) or () if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class SearchConfig:
    """User-tunable knobs for the research stage."""

    max_queries: int = 8
    max_output_tokens: int = 1500
    auto_run: bool = True
    concurrency: int = 3

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_queries": self.max_queries,
            "max_output_tokens": self.max_output_tokens,
            "auto_run": self.auto_run,
            "concurrency": self.concurrency,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SearchConfig":
        value = value or {}
        return cls(
            max_queries=max(1, int(value.get("max_queries", 8))),
            max_output_tokens=max(256, int(value.get("max_output_tokens", 1500))),
            auto_run=bool(value.get("auto_run", True)),
            concurrency=max(1, int(value.get("concurrency", 3))),
        )


def _json_or_dict(value: Any) -> Any:
    """DB rows store JSON strings; API payloads pass dicts directly."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return value or {}


@dataclass(frozen=True)
class TaskRecord:
    """Persisted task row plus the latest observed cost snapshots."""

    task_id: str
    goal: str
    seed_material: str
    model: str
    budget: float
    search_config: SearchConfig
    status: str = STATUS_PENDING
    stage: str = STAGE_ORDER[0]
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_balance: str | None = None
    finished_balance: str | None = None
    spent: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "seed_material": self.seed_material,
            "model": self.model,
            "budget": self.budget,
            "search_config": self.search_config.to_mapping(),
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_balance": self.started_balance,
            "finished_balance": self.finished_balance,
            "spent": self.spent,
            "usage": dict(self.usage),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskRecord":
        search_config = _json_or_dict(value.get("search_config"))
        return cls(
            task_id=str(value.get("task_id", "")),
            goal=str(value.get("goal", "")),
            seed_material=str(value.get("seed_material", "")),
            model=str(value.get("model", "")),
            budget=float(value.get("budget", 0.0)),
            search_config=SearchConfig.from_mapping(search_config),
            status=str(value.get("status", STATUS_PENDING)),
            stage=str(value.get("stage", STAGE_ORDER[0])),
            error=value.get("error"),
            created_at=float(value.get("created_at", 0.0)),
            updated_at=float(value.get("updated_at", 0.0)),
            started_balance=value.get("started_balance"),
            finished_balance=value.get("finished_balance"),
            spent=value.get("spent"),
            usage=dict(_json_or_dict(value.get("usage"))),
        )
