from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldline_social.experiment import ExperimentConfig
from worldline_social.runner import run_experiment_sync


class ExperimentRunnerTests(unittest.TestCase):
    def test_json_experiment_runs_and_persists_events_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "population.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "1",
                        "source": "runner-test",
                        "people": [
                            {"external_id": "alice-source", "handle": "alice"},
                            {"external_id": "bob-source", "handle": "bob"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "config_version": "1",
                        "simulation_id": "runner-test",
                        "population_manifest": "population.json",
                        "output_database": "run.sqlite",
                        "max_ticks": 1,
                        "max_actions_per_turn": 1,
                        "scripted_actions": {
                            "alice-source": [
                                {
                                    "action_type": "create_post",
                                    "parameters": {"content": "Configured post"},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ExperimentConfig.from_json(config_path)
            result = run_experiment_sync(config)

            self.assertEqual(2, result.population_size)
            self.assertEqual(1, len(result.world_state["posts"]))
            connection = sqlite3.connect(result.database_path)
            try:
                event_count = connection.execute(
                    "SELECT COUNT(*) FROM simulation_events"
                ).fetchone()[0]
                checkpoint_count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(event_count, 0)
            self.assertEqual(1, checkpoint_count)

    def test_resume_requires_an_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "population.json").write_text(
                json.dumps(
                    {
                        "manifest_version": "1",
                        "source": "resume-test",
                        "people": [{"external_id": "alice", "handle": "alice"}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "experiment.json").write_text(
                json.dumps(
                    {
                        "config_version": "1",
                        "simulation_id": "missing-checkpoint",
                        "population_manifest": "population.json",
                        "output_database": "run.sqlite",
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_json(root / "experiment.json")
            with self.assertRaises(ValueError):
                run_experiment_sync(config, resume=True)


if __name__ == "__main__":
    unittest.main()
