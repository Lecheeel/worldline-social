from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from worldline_engine import ReplayController

from worldline_social.controllers import LLMToolController
from worldline_social.experiment import ExperimentConfig
from worldline_social.runner import build_simulation


def write_config(tmp: Path, *, llm: dict | None = None, scripted: dict | None = None) -> Path:
    population = {
        "manifest_version": "1",
        "source": "runner-test",
        "people": [
            {
                "external_id": "alice",
                "handle": "alice",
                "display_name": "Alice",
                "bio": "Test user",
                "private_traits": {"openness": 0.7},
                "model_policy": {"persona": "Alice likes data."},
            },
            {
                "external_id": "bob",
                "handle": "bob",
                "display_name": "Bob",
                "bio": "Test user",
                "model_policy": {"persona": "Bob is skeptical."},
            },
        ],
        "relationships": [],
    }
    (tmp / "population.json").write_text(
        json.dumps(population, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "config_version": "1",
        "simulation_id": "runner-test",
        "population_manifest": "population.json",
        "output_database": "../runs/runner-test.sqlite",
        "seed": 1,
        "max_ticks": 1,
    }
    if llm is not None:
        config["llm"] = llm
    if scripted is not None:
        config["scripted_actions"] = scripted
    config_path = tmp / "experiment.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config_path


class RunnerAssemblyTests(unittest.TestCase):
    def test_without_llm_config_all_controllers_are_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(write_config(Path(tmp)))
            simulation, world, store, sink = build_simulation(config)
            try:
                for ref, controller in simulation.controllers.items():
                    self.assertIsInstance(controller, ReplayController)
                self.assertEqual(2, len(world.people))
            finally:
                sink.close()
                store.close()

    def test_llm_config_requires_api_key_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(
                write_config(Path(tmp), llm={"provider": "deepseek", "model": "deepseek-chat"})
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DEEPSEEK_API_KEY", None)
                with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                    build_simulation(config)

    def test_llm_config_assembles_llm_controllers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(
                write_config(Path(tmp), llm={"provider": "deepseek", "model": "deepseek-chat"})
            )
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                simulation, world, store, sink = build_simulation(config)
            try:
                for ref, controller in simulation.controllers.items():
                    self.assertIsInstance(controller, LLMToolController)
            finally:
                sink.close()
                store.close()

    def test_scripted_actor_stays_replay_in_llm_experiment(self) -> None:
        scripted = {
            "bob": [
                {"action_type": "do_nothing", "parameters": {}}
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(
                write_config(
                    Path(tmp),
                    llm={"provider": "deepseek", "model": "deepseek-chat"},
                    scripted=scripted,
                )
            )
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                simulation, world, store, sink = build_simulation(config)
            try:
                controllers = list(simulation.controllers.values())
                self.assertEqual(1, len([c for c in controllers if isinstance(c, ReplayController)]))
                self.assertEqual(1, len([c for c in controllers if isinstance(c, LLMToolController)]))
            finally:
                sink.close()
                store.close()

    def test_llm_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(Path(tmp), llm={"provider": "openai"})
            with self.assertRaisesRegex(ValueError, "provider"):
                ExperimentConfig.from_json(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(Path(tmp), llm={"provider": "deepseek"})
            with self.assertRaisesRegex(ValueError, "model"):
                ExperimentConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
