"""Versioned experiment configuration for reproducible social runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldline_engine import ActionIntent, SimulationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    config_version: str
    simulation_id: str
    population_manifest: Path
    output_database: Path
    seed: int = 0
    max_ticks: int = 1
    activation_probability: float = 1.0
    max_concurrent_turns: int = 1
    max_actions_per_turn: int = 8
    max_controller_calls_per_turn: int = 8
    max_cost_per_turn: int | None = None
    turn_timeout_seconds: float | None = None
    checkpoint_every_ticks: int = 1
    distribution_policy: str = "all"
    dynamics_policy: str = "affective"
    feed_limit: int = 100
    scripted_actions: Mapping[str, Sequence[ActionIntent]] = field(default_factory=dict)
    llm: Mapping[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("experiment config must be a JSON object")
        base = config_path.parent
        scripted_actions = {
            external_id: tuple(
                ActionIntent(
                    action_type=action["action_type"],
                    parameters=dict(action.get("parameters", {})),
                    target_ref=action.get("target_ref"),
                    client_ref=action.get("client_ref"),
                )
                for action in actions
            )
            for external_id, actions in value.get("scripted_actions", {}).items()
        }
        config = cls(
            config_version=str(value.get("config_version", "")),
            simulation_id=str(value.get("simulation_id", "")),
            population_manifest=(base / value["population_manifest"]).resolve(),
            output_database=(base / value["output_database"]).resolve(),
            seed=int(value.get("seed", 0)),
            max_ticks=int(value.get("max_ticks", 1)),
            activation_probability=float(value.get("activation_probability", 1.0)),
            max_concurrent_turns=int(value.get("max_concurrent_turns", 1)),
            max_actions_per_turn=int(value.get("max_actions_per_turn", 8)),
            max_controller_calls_per_turn=int(
                value.get("max_controller_calls_per_turn", 8)
            ),
            max_cost_per_turn=value.get("max_cost_per_turn"),
            turn_timeout_seconds=value.get("turn_timeout_seconds"),
            checkpoint_every_ticks=int(value.get("checkpoint_every_ticks", 1)),
            distribution_policy=str(value.get("distribution_policy", "all")),
            dynamics_policy=str(value.get("dynamics_policy", "affective")),
            feed_limit=int(value.get("feed_limit", 100)),
            scripted_actions=scripted_actions,
            llm=dict(value["llm"]) if value.get("llm") is not None else None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.config_version != "1":
            raise ValueError("unsupported config_version")
        if not self.simulation_id.strip():
            raise ValueError("simulation_id must not be empty")
        if not self.population_manifest.is_file():
            raise ValueError("population_manifest does not exist")
        if self.output_database.suffix not in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("output_database must be a SQLite database path")
        if not 0.0 <= self.activation_probability <= 1.0:
            raise ValueError("activation_probability must be in [0, 1]")
        if self.distribution_policy not in {"all", "recent"}:
            raise ValueError("distribution_policy must be 'all' or 'recent'")
        if self.dynamics_policy not in {"recovery", "affective"}:
            raise ValueError("dynamics_policy must be 'recovery' or 'affective'")
        if self.feed_limit < 1:
            raise ValueError("feed_limit must be positive")
        if self.llm is not None:
            provider = str(self.llm.get("provider", "")).strip()
            model = str(self.llm.get("model", "")).strip()
            if not provider or not model:
                raise ValueError("llm requires both 'provider' and 'model'")
            if provider not in {"deepseek"}:
                raise ValueError(f"unsupported llm provider: {provider}")

    def engine_config(self) -> SimulationConfig:
        return SimulationConfig(
            simulation_id=self.simulation_id,
            seed=self.seed,
            max_ticks=self.max_ticks,
            max_actions_per_turn=self.max_actions_per_turn,
            max_controller_calls_per_turn=self.max_controller_calls_per_turn,
            max_concurrent_turns=self.max_concurrent_turns,
            checkpoint_every_ticks=self.checkpoint_every_ticks,
            max_cost_per_turn=self.max_cost_per_turn,
            turn_timeout_seconds=self.turn_timeout_seconds,
        )
