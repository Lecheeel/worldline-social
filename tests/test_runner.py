from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from unittest import mock

from worldline_engine import ReplayController

from worldline_social.controllers import LLMToolController
from worldline_social.experiment import ExperimentConfig
from worldline_social.population import PopulationManifest
from worldline_social.runner import build_simulation, build_worldline_manifest


def write_config(
    tmp: Path,
    *,
    llm: dict | None = None,
    scripted: dict | None = None,
    output_database: str = "../runs/runner-test.sqlite",
) -> Path:
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
        "output_database": output_database,
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


class WorldlineManifestTests(unittest.TestCase):
    def test_manifest_pins_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(
                write_config(
                    Path(tmp),
                    llm={"provider": "deepseek", "model": "deepseek-chat", "temperature": 0.4},
                )
            )
            population = PopulationManifest.from_json(config.population_manifest)
            manifest = build_worldline_manifest(config, population)

        self.assertEqual("1", manifest["worldline_manifest_version"])
        self.assertEqual(metadata.version("worldline-engine"), manifest["engine"])
        self.assertEqual(metadata.version("worldline-social"), manifest["social"])
        self.assertEqual(2, manifest["population"]["size"])
        self.assertEqual(64, len(manifest["population"]["sha256"]))
        self.assertEqual("deepseek", manifest["llm"]["provider"])
        self.assertEqual(0.4, manifest["llm"]["temperature"])
        self.assertIsNone(manifest["scripted_actions_sha256"])
        self.assertEqual(1, manifest["experiment"]["seed"])

    def test_seed_change_only_touches_experiment_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(write_config(Path(tmp)))
            population = PopulationManifest.from_json(config.population_manifest)
            base = build_worldline_manifest(config, population)
            other = build_worldline_manifest(replace(config, seed=2), population)

        self.assertEqual(base["population"], other["population"])
        self.assertEqual(base["llm"], other["llm"])
        self.assertEqual(base["prompt_builder_sha256"], other["prompt_builder_sha256"])
        self.assertEqual(1, base["experiment"]["seed"])
        self.assertEqual(2, other["experiment"]["seed"])

    def test_scripted_actions_are_fingerprinted(self) -> None:
        scripted = {"bob": [{"action_type": "do_nothing", "parameters": {}}]}
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(write_config(Path(tmp), scripted=scripted))
            population = PopulationManifest.from_json(config.population_manifest)
            manifest = build_worldline_manifest(config, population)

        self.assertEqual(64, len(manifest["scripted_actions_sha256"]))

    def test_run_emits_worldline_manifest_as_first_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig.from_json(
                write_config(Path(tmp), output_database="run.sqlite")
            )
            simulation, world, store, sink = build_simulation(config)
            try:
                asyncio.run(simulation.run())
            finally:
                sink.close()
                store.close()

            connection = sqlite3.connect(config.output_database)
            rows = connection.execute(
                "SELECT event_type, payload_json FROM simulation_events"
                " ORDER BY sequence LIMIT 2"
            ).fetchall()
            connection.close()

        self.assertEqual("worldline_manifest", rows[0][0])
        payload = json.loads(rows[0][1])
        self.assertEqual("1", payload["manifest"]["worldline_manifest_version"])
        self.assertEqual(64, len(payload["manifest"]["population"]["sha256"]))
        self.assertEqual("simulation_started", rows[1][0])


if __name__ == "__main__":
    unittest.main()
