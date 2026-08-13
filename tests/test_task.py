from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from worldline_social.providers import BalanceInfo, CompletionResponse
from worldline_social.task import (
    STAGE_ORDER,
    Decomposition,
    ResearchQuestion,
    SearchConfig,
    TaskRunner,
    TaskStore,
    WorldDesign,
)
from worldline_social.task.agents import render_event_text
from worldline_social.task.runner import TaskBudgetExceeded
from worldline_social.task.search import SearchError, SearchResult, web_search_sync

DECOMPOSE_JSON = json.dumps(
    {
        "research_questions": [
            {"id": "q1", "question": "某公司 2025 年事件时间线", "rationale": "了解事件经过"},
            {"id": "q2", "question": "某公司 财报 争议", "rationale": "了解争议焦点"},
        ],
        "architecture_requirements": "需要一家公司、监管机构与媒体互动的世界",
        "cast_requirements": "包括公司发言人、记者、监管官员",
    },
    ensure_ascii=False,
)

DESIGN_JSON = json.dumps(
    {
        "title": "某公司舆情风波",
        "background": "一家上市公司陷入财务争议。",
        "event_script": "公司被曝财务造假，引发舆论热议。",
        "world_rules": "媒体可以发帖报道；公司可以回应。",
        "required_actors": ["某公司：涉事企业", "财经媒体：报道方"],
    },
    ensure_ascii=False,
)


class FakeProvider:
    """Scripted provider: returns queued responses, records requests."""

    provider_id = "fake"

    def __init__(self, *responses: CompletionResponse) -> None:
        self._responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not self._responses:
            return CompletionResponse("{}")
        return self._responses.pop(0)

    async def get_balance(self):
        return BalanceInfo(True, "CNY", "100.00", "0.00", "100.00")


class FakeSearcher:
    def __init__(self, *results: SearchResult) -> None:
        self._results = list(results)
        self.queries = []

    async def __call__(self, query, **kwargs):
        self.queries.append(query)
        if not self._results:
            return SearchResult("无结果")
        return self._results.pop(0)


EXTRACTION_JSON = json.dumps(
    {
        "participants": [
            {"name": "某公司", "entity_type": "company", "role": "涉事企业",
             "summary": "被曝财务造假的公司", "stance": "neutral"},
            {"name": "财经媒体", "entity_type": "media", "role": "报道方",
             "summary": "报道事件的媒体", "stance": "observer"},
        ],
        "relationships": [
            {"source": "财经媒体", "target": "某公司", "relationship_type": "media_covers"}
        ],
        "initial_spark": "某公司被曝财务造假，引发舆论热议。",
    },
    ensure_ascii=False,
)

PROFILE_JSON = json.dumps(
    {
        "display_name": "某公司", "bio": "涉事企业", "persona": "公司官方",
        "traits": {"openness": 0.5}, "stance": "neutral",
        "interested_topics": ["财务"],
    },
    ensure_ascii=False,
)


class RoutingProvider:
    """Fake provider that routes responses by the request's system prompt.

    The task pipeline mixes several agents (decompose/architect/extractor/
    profiles), so a FIFO queue is not enough: route on the system message.
    """

    provider_id = "fake"

    def __init__(self) -> None:
        self.requests = []
        self._routes = {
            "你是一个社会模拟实验设计师": CompletionResponse(
                DECOMPOSE_JSON, usage={"prompt_tokens": 120, "completion_tokens": 30}
            ),
            "你是一个社会模拟世界架构师": CompletionResponse(
                DESIGN_JSON, usage={"prompt_tokens": 200, "completion_tokens": 50}
            ),
            "你是一个社会舆论事件分析专家": CompletionResponse(
                EXTRACTION_JSON, usage={"prompt_tokens": 300, "completion_tokens": 60}
            ),
            "你是社交媒体用户画像生成专家": CompletionResponse(
                PROFILE_JSON, usage={"prompt_tokens": 80, "completion_tokens": 20}
            ),
        }

    async def complete(self, request):
        self.requests.append(request)
        system = request.messages[0].content if request.messages else ""
        for marker, response in self._routes.items():
            if marker in system:
                return response
        return CompletionResponse("{}")

    async def get_balance(self):
        return BalanceInfo(True, "CNY", "100.00", "0.00", "100.00")


class TaskStoreTests(unittest.TestCase):
    def test_create_get_list_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                record = store.create_task(
                    goal="分析某公司舆情",
                    seed_material="种子材料",
                    model="deepseek-v4-flash",
                    budget=5.0,
                    search_config=SearchConfig(max_queries=3, auto_run=False),
                )
                fetched = store.get_task(record.task_id)
                self.assertEqual(record.task_id, fetched.task_id)
                self.assertEqual("分析某公司舆情", fetched.goal)
                self.assertEqual(5.0, fetched.budget)
                self.assertFalse(fetched.search_config.auto_run)
                self.assertEqual(3, fetched.search_config.max_queries)
                self.assertEqual(["decompose"], [t.stage for t in store.list_tasks()])

    def test_artifacts_edit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                task = store.create_task(goal="g")
                store.set_artifact(task.task_id, "decompose", {"research_questions": []})
                artifact = store.get_artifact(task.task_id, "decompose")
                self.assertIsNotNone(artifact)
                assert artifact is not None
                self.assertFalse(artifact["_edited"])
                store.mark_edited(task.task_id, "decompose")
                self.assertTrue(store.get_artifact(task.task_id, "decompose")["_edited"])

    def test_task_dirs_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                task = store.create_task(goal="g")
                store.set_artifact(
                    task.task_id, "cast",
                    {"people": [], "manifest_version": "1", "source": "s"},
                )
                path = store.export_artifact(task.task_id, "cast")
                self.assertIsNotNone(path)
                assert path is not None
                self.assertTrue(path.is_file())
                self.assertEqual("s", json.loads(path.read_text(encoding="utf-8"))["source"])

    def test_unknown_task_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                with self.assertRaises(Exception):
                    store.get_task("nope")


class TaskRunnerTests(unittest.TestCase):
    def test_full_pipeline_to_configure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                runner = TaskRunner(
                    store,
                    provider=RoutingProvider(),
                    search_fn=FakeSearcher(
                        SearchResult("公司 2025 年财报争议汇总 https://example.com/a"),
                        SearchResult("监管机构回应报道 https://example.com/b"),
                    ),
                )
                task = store.create_task(
                    goal="分析某公司财务争议",
                    seed_material="公司被曝财务造假",
                    model="deepseek-v4-flash",
                )
                task = asyncio.run(runner.run_until(task.task_id, stage="configure"))

                # configure completed: the next pending stage is "run"
                self.assertEqual("run", task.stage)
                self.assertEqual("running", task.status)
                for stage in ("decompose", "research", "architect", "cast", "configure"):
                    self.assertIsNotNone(store.get_artifact(task.task_id, stage), stage)
                research = store.get_artifact(task.task_id, "research")
                self.assertEqual(2, research["count"])
                self.assertIn("https://example.com/a", research["notes"][0]["sources"])
                self.assertTrue((store.task_dir(task.task_id) / "manifest.json").is_file())
                self.assertTrue(store.experiment_path(task.task_id).is_file())
                self.assertGreater(task.usage.get("prompt_tokens", 0), 0)

    def test_configure_exports_parseable_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                runner = TaskRunner(
                    store,
                    provider=RoutingProvider(),
                    search_fn=FakeSearcher(SearchResult("r1"), SearchResult("r2")),
                )
                task = store.create_task(goal="g", model="deepseek-v4-flash")
                asyncio.run(runner.run_until(task.task_id, stage="configure"))

                from worldline_social.experiment import ExperimentConfig

                config = ExperimentConfig.from_json(store.experiment_path(task.task_id))
                self.assertEqual(f"task-{task.task_id}", config.simulation_id)
                self.assertTrue(config.population_manifest.is_file())

    def test_decompose_edits_flow_into_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                runner = TaskRunner(
                    store,
                    provider=RoutingProvider(),
                    search_fn=FakeSearcher(SearchResult("r1"), SearchResult("r2")),
                )
                task = store.create_task(goal="g")
                asyncio.run(runner.advance(task.task_id))
                # user edits the decomposition: only one question
                store.set_artifact(
                    task.task_id, "decompose",
                    Decomposition(
                        research_questions=(ResearchQuestion("q1", "只剩这一个问题"),)
                    ).to_mapping(),
                    edited=True,
                )
                asyncio.run(runner.advance(task.task_id))
                research = store.get_artifact(task.task_id, "research")
                self.assertEqual(1, research["count"])
                self.assertEqual("只剩这一个问题", research["notes"][0]["query"])

    def test_search_failure_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                runner = TaskRunner(
                    store,
                    provider=FakeProvider(CompletionResponse(DECOMPOSE_JSON)),
                    search_fn=FakeSearcher(SearchResult("ok")),
                )
                task = store.create_task(goal="g")
                asyncio.run(runner.advance(task.task_id))

                async def failing(query, **kwargs):
                    raise SearchError("network down")

                runner._search_fn = failing
                asyncio.run(runner.advance(task.task_id))
                research = store.get_artifact(task.task_id, "research")
                self.assertEqual(2, research["count"])
                self.assertIsNotNone(research["notes"][0]["error"])

    def test_budget_exceeded_pauses_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:

                class DrainingProvider(RoutingProvider):
                    def __init__(self) -> None:
                        super().__init__()
                        self._queries = 0

                    async def get_balance(self):
                        # First snapshot sets the baseline; the second shows
                        # the task has already spent 1.00 CNY.
                        self._queries += 1
                        total = "99.00" if self._queries == 1 else "98.00"
                        return BalanceInfo(True, "CNY", total, "0.00", total)

                runner = TaskRunner(
                    store, provider=DrainingProvider(), search_fn=FakeSearcher()
                )
                task = store.create_task(goal="g", budget=0.5)
                asyncio.run(runner.advance(task.task_id))
                with self.assertRaises(TaskBudgetExceeded):
                    asyncio.run(runner.advance(task.task_id))
                task = store.get_task(task.task_id)
                self.assertEqual("paused", task.status)
                self.assertIn("budget", task.error or "")

    def test_cancel_blocks_further_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with TaskStore(Path(tmp) / "tasks") as store:
                runner = TaskRunner(
                    store, provider=FakeProvider(), search_fn=FakeSearcher()
                )
                task = store.create_task(goal="g")
                runner.cancel(task.task_id)
                with self.assertRaises(Exception):
                    asyncio.run(runner.advance(task.task_id))


class SearchAndRenderTests(unittest.TestCase):
    def test_render_event_text_assembles_material(self) -> None:
        design = WorldDesign(
            title="t", background="b", event_script="s", world_rules="w",
            required_actors=("公司：企业",),
        )
        text = render_event_text(
            design,
            {"notes": [{"query": "q", "summary": "摘要内容"}]},
            seed_material="种子",
        )
        self.assertIn("【世界标题】t", text)
        self.assertIn("摘要内容", text)
        self.assertIn("种子", text)

    def test_web_search_sync_requires_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("DEEPSEEK_API_KEY", None)
            with self.assertRaises(SearchError):
                web_search_sync("query")

    def test_stage_order(self) -> None:
        self.assertEqual(
            ("decompose", "research", "architect", "cast", "configure", "run"),
            STAGE_ORDER,
        )


if __name__ == "__main__":
    unittest.main()
