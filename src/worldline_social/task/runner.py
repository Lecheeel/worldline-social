"""Task orchestrator: executes stages in order, persisting artifacts.

Each stage is a unit of work that can be run, edited, and re-run:

    decompose  research  architect  cast  configure  run

The runner is async so the studio backend can stream progress over SSE
while a task runs; the CLI wraps it with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import replace
from typing import Any, Awaitable, Callable, Mapping

from ..experiment import ExperimentConfig
from ..generation import generate_population_from_text
from ..providers import BalanceInfo, DeepSeekProvider
from ..runner import run_experiment
from ..stats import UsageRecord, aggregate_usage
from .agents import (
    DecomposeAgent,
    WorldDesignAgent,
    render_event_text,
)
from .models import (
    STAGE_ORDER,
    Decomposition,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PENDING,
    STATUS_RUNNING,
    SearchConfig,
    TaskRecord,
    WorldDesign,
)
from .search import SearchError, SearchResult, web_search
from .store import TaskError, TaskStore

ProgressCallback = Callable[[str, str, str], None]

_STAGE_LABELS = {
    "decompose": "目标拆解",
    "research": "资料搜索",
    "architect": "世界架构",
    "cast": "主体与性格生成",
    "configure": "参数配置",
    "run": "运行模拟",
}


class TaskBudgetExceeded(TaskError):
    """Raised when the task's spend reaches the configured budget."""


class TaskRunner:
    """Execute the stage pipeline for tasks in a :class:`TaskStore`.

    ``provider`` may be injected (tests); by default it is created from the
    environment via ``DEEPSEEK_API_KEY``. ``search_fn`` may be injected for
    offline tests; the default is the real web search.
    """

    def __init__(
        self,
        store: TaskStore,
        *,
        provider: DeepSeekProvider | None = None,
        search_fn: Callable[..., Awaitable[SearchResult]] | None = None,
        on_progress: ProgressCallback | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
    ) -> None:
        self._store = store
        self._provider_instance = provider
        self._search_fn = search_fn or web_search
        self._on_progress = on_progress
        self._api_key_env = api_key_env
        self._usage_records: list[UsageRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_task(
        self,
        *,
        goal: str,
        seed_material: str = "",
        model: str = "deepseek-v4-flash",
        budget: float = 0.0,
        search_config: SearchConfig | None = None,
    ) -> TaskRecord:
        return self._store.create_task(
            goal=goal,
            seed_material=seed_material,
            model=model,
            budget=budget,
            search_config=search_config,
        )

    async def advance(self, task_id: str) -> TaskRecord:
        """Execute the task's current stage, then advance to the next one."""
        task = self._store.get_task(task_id)
        if task.status == STATUS_CANCELLED:
            raise TaskError("task is cancelled")
        if task.status == STATUS_DONE:
            raise TaskError("task is already done")
        stage = task.stage
        self._emit(task_id, stage, f"开始阶段：{_STAGE_LABELS[stage]}")
        task = self._store.set_status(task_id, STATUS_RUNNING, error=None)
        try:
            await self._run_stage(task, stage)
            self._usage_records.clear()
        except TaskBudgetExceeded as error:
            self._store.set_status(task_id, STATUS_PAUSED, error=str(error))
            self._emit(task_id, stage, f"预算耗尽，任务暂停：{error}")
            raise
        except Exception as error:
            self._store.set_status(task_id, STATUS_FAILED, error=str(error))
            self._emit(task_id, stage, f"阶段失败：{error}")
            raise
        task = self._store.get_task(task_id)
        if stage == "run":
            return self._store.set_status(task_id, STATUS_DONE)
        return self._store.set_stage(task_id, STAGE_ORDER[STAGE_ORDER.index(stage) + 1])

    async def run_until(self, task_id: str, stage: str | None = None) -> TaskRecord:
        """Execute stages until ``stage`` (inclusive); default: the whole task."""
        if stage is not None and stage not in STAGE_ORDER:
            raise TaskError(f"invalid stage: {stage}")
        while True:
            task = self._store.get_task(task_id)
            if task.status == STATUS_DONE or task.status == STATUS_CANCELLED:
                return task
            if stage is not None and STAGE_ORDER.index(task.stage) > STAGE_ORDER.index(stage):
                return task
            try:
                task = await self.advance(task_id)
            except TaskBudgetExceeded:
                return self._store.get_task(task_id)
        # unreachable

    def cancel(self, task_id: str) -> TaskRecord:
        return self._store.set_status(task_id, STATUS_CANCELLED)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _run_stage(self, task: TaskRecord, stage: str) -> None:
        if stage == "decompose":
            await self._stage_decompose(task)
        elif stage == "research":
            await self._stage_research(task)
        elif stage == "architect":
            await self._stage_architect(task)
        elif stage == "cast":
            await self._stage_cast(task)
        elif stage == "configure":
            await self._stage_configure(task)
        elif stage == "run":
            await self._stage_run(task)
        else:
            raise TaskError(f"unknown stage: {stage}")
        # Merge LLM usage recorded by this stage into the task totals.
        usage = aggregate_usage(self._usage_records)
        if usage:
            task = self._store.get_task(task.task_id)
            merged = dict(task.usage)
            for metric, value in usage.items():
                merged[metric] = merged.get(metric, 0) + value
            self._store.update_task(replace(task, usage=merged))
            self._usage_records.clear()
        # A finished simulation should not be rolled back by budget checks.
        await self._refresh_cost(task.task_id, enforce_budget=stage != "run")

    async def _stage_decompose(self, task: TaskRecord) -> None:
        agent = DecomposeAgent(self._provider(), task.model, usage_recorder=self._usage_records.append)
        decomposition = await agent.decompose(task.goal, task.seed_material)
        self._store.set_artifact(task.task_id, "decompose", decomposition.to_mapping())
        self._emit(task.task_id, "decompose",
                   f"拆解完成：{len(decomposition.research_questions)} 个研究问题")

    async def _stage_research(self, task: TaskRecord) -> None:
        artifact = self._store.get_artifact(task.task_id, "decompose")
        if artifact is None:
            raise TaskError("decompose stage must run before research")
        decomposition = Decomposition.from_mapping(artifact)
        questions = decomposition.research_questions[: task.search_config.max_queries]
        if not questions:
            raise TaskError("no research questions; edit the decompose artifact first")
        concurrency = max(1, task.search_config.concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        notes: list[dict[str, Any]] = []

        async def search_one(item: Any) -> None:
            async with semaphore:
                note = {"id": str(item.id), "query": item.question, "summary": "", "sources": []}
                self._emit(task.task_id, "research", f"搜索：{item.question}")
                try:
                    result = await self._search_fn(
                        item.question,
                        max_output_tokens=task.search_config.max_output_tokens,
                        model=task.model,
                    )
                    note["summary"] = result.text
                    note["sources"] = _extract_sources(result.text)
                except SearchError as error:
                    note["error"] = str(error)
                    self._emit(task.task_id, "research", f"搜索失败：{item.question}（{error}）")
                notes.append(note)

        await asyncio.gather(*(search_one(item) for item in questions))
        self._store.set_artifact(
            task.task_id, "research", {"notes": notes, "count": len(notes)}
        )
        ok = sum(1 for note in notes if not note.get("error"))
        self._emit(task.task_id, "research", f"搜索完成：{ok}/{len(notes)} 条成功")

    async def _stage_architect(self, task: TaskRecord) -> None:
        decompose = self._store.get_artifact(task.task_id, "decompose")
        research = self._store.get_artifact(task.task_id, "research")
        if decompose is None or research is None:
            raise TaskError("research stage must run before architect")
        agent = WorldDesignAgent(self._provider(), task.model, usage_recorder=self._usage_records.append)
        design = await agent.design(
            task.goal, task.seed_material, research, Decomposition.from_mapping(decompose)
        )
        self._store.set_artifact(task.task_id, "architect", design.to_mapping())
        self._emit(task.task_id, "architect", f"架构完成：{design.title}（{len(design.required_actors)} 个必需主体）")

    async def _stage_cast(self, task: TaskRecord) -> None:
        architect = self._store.get_artifact(task.task_id, "architect")
        research = self._store.get_artifact(task.task_id, "research")
        if architect is None:
            raise TaskError("architect stage must run before cast")
        design = WorldDesign.from_mapping(architect)
        event_text = render_event_text(design, research or {}, task.seed_material)
        result = await generate_population_from_text(
            event_text,
            self._provider(),
            task.model,
            source=f"task:{task.task_id}",
            thinking="disabled",
            usage_recorder=self._usage_records.append,
        )
        payload = result.manifest.to_mapping()
        payload["diagnostics"] = dict(result.diagnostics)
        self._store.set_artifact(task.task_id, "cast", payload)
        self._store.export_artifact(task.task_id, "cast")
        self._emit(task.task_id, "cast",
                   f"主体生成完成：{len(result.manifest.people)} 人，"
                   f"{len(result.manifest.relationships)} 条关系")

    async def _stage_configure(self, task: TaskRecord) -> None:
        cast = self._store.get_artifact(task.task_id, "cast")
        if cast is None:
            raise TaskError("cast stage must run before configure")
        manifest_path = self._store.task_dir(task.task_id) / "manifest.json"
        config = _default_experiment_config(
            task=task,
            manifest_path=manifest_path,
            simulation_db=self._store.simulation_db(task.task_id),
        )
        payload = {**config, "_generated": True}
        self._store.set_artifact(task.task_id, "configure", payload)
        self._store.export_artifact(task.task_id, "configure")
        self._emit(task.task_id, "configure", "实验参数已生成（可编辑）")

    async def _stage_run(self, task: TaskRecord) -> None:
        experiment_path = self._store.experiment_path(task.task_id)
        if not experiment_path.is_file():
            raise TaskError("configure stage must run before the simulation")
        config = ExperimentConfig.from_json(experiment_path)
        # Re-running the simulation stage means a fresh worldline: the engine
        # appends events with monotonically increasing sequences and cannot
        # resume from a stale database (UNIQUE on simulation_id+sequence).
        simulation_db = self._store.simulation_db(task.task_id)
        for suffix in ("", "-wal", "-shm"):
            stale = Path(f"{simulation_db}{suffix}")
            if stale.exists():
                stale.unlink()
        self._emit(task.task_id, "run", "开始运行模拟")
        result = await run_experiment(config)
        task = self._store.get_task(task.task_id)
        usage = dict(task.usage)
        for metric, value in result.usage.items():
            usage[metric] = usage.get(metric, 0) + value
        self._store.update_task(replace(task, usage=usage))
        self._emit(task.task_id, "run", f"模拟完成：{result.completed_ticks} ticks")
        self._emit(task.task_id, "run", f"模拟用量：{result.usage}")
        if result.cost is not None:
            self._emit(task.task_id, "run", f"本次运行花费：{result.cost.get('spent')}")

    # ------------------------------------------------------------------
    # Balance & budget
    # ------------------------------------------------------------------

    async def _refresh_cost(self, task_id: str, *, enforce_budget: bool = True) -> None:
        provider = self._provider()
        try:
            balance = await provider.get_balance()
        except Exception:
            return
        task = self._store.get_task(task_id)
        if not balance.is_available:
            return
        current = balance.total_balance
        started = task.started_balance if task.started_balance else current
        try:
            spent = f"{float(started) - float(current):.4f}"
        except (TypeError, ValueError):
            spent = None
        task = replace(task, started_balance=started, spent=spent)
        self._store.update_task(task)
        if enforce_budget and task.budget > 0 and spent is not None and float(spent) >= task.budget:
            raise TaskBudgetExceeded(
                f"spent {spent} >= budget {task.budget:.2f}"
            )

    # ------------------------------------------------------------------

    def _provider(self) -> DeepSeekProvider:
        if self._provider_instance is None:
            self._provider_instance = DeepSeekProvider.from_environment(self._api_key_env)
        return self._provider_instance

    def _emit(self, task_id: str, stage: str, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(task_id, stage, message)


def _default_experiment_config(*, task: TaskRecord, manifest_path: Any, simulation_db: Any) -> dict[str, Any]:
    """Sensible experiment defaults; users edit these before the run stage."""
    return {
        "config_version": "1",
        "simulation_id": f"task-{task.task_id}",
        "population_manifest": str(manifest_path),
        "output_database": str(simulation_db),
        "seed": 0,
        "max_ticks": 5,
        "activation_probability": 1.0,
        "max_concurrent_turns": 8,
        "max_actions_per_turn": 8,
        "max_controller_calls_per_turn": 8,
        "checkpoint_every_ticks": 1,
        "distribution_policy": "recent",
        "dynamics_policy": "affective",
        "feed_limit": 100,
        "llm": {
            "provider": "deepseek",
            "model": task.model,
            "thinking": "disabled",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    }


def _extract_sources(text: str) -> list[str]:
    """Best-effort source extraction from a search summary (URL-like tokens)."""
    import re

    urls = re.findall(r"https?://[^\s)\]}>\"']+", text)
    seen: list[str] = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    return seen[:5]
