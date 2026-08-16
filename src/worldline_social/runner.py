"""Composition root for configured Worldline Social experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from worldline_engine import (
    AllEntitiesScheduler,
    EntitySpec,
    RandomActivationScheduler,
    ReplayController,
    SQLiteEventSink,
    SQLiteStateStore,
    Simulation,
)

from . import prompting
from .controllers import LLMToolController
from .distribution import AllPostsDistribution, RecentPostsDistribution
from .dynamics import AffectiveDynamics, RecoveryDynamics
from .experiment import ExperimentConfig
from .population import PopulationManifest
from .prompting import SocialPromptBuilder
from .providers import BalanceInfo, DeepSeekProvider
from .stats import SQLiteUsageStore, UsageRecorder
from .world import SocialWorld


@dataclass(frozen=True)
class ExperimentResult:
    simulation_id: str
    completed_ticks: int
    population_size: int
    world_state: dict[str, Any]
    database_path: str
    usage: dict[str, int] = field(default_factory=dict)
    cost: dict[str, Any] | None = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_worldline_manifest(
    config: ExperimentConfig, population: PopulationManifest
) -> dict[str, Any]:
    """Assemble the content-addressed identity of this worldline.

    A worldline is the pair (manifest, recorded stream); the seed is only
    the random-stream parameter. Everything that can change the run is
    pinned here, so two worldlines are comparable exactly when their
    manifests differ in a single field.
    """
    llm = None
    if config.llm is not None:
        llm = {
            "provider": config.llm.get("provider"),
            "model": config.llm.get("model"),
            "temperature": config.llm.get("temperature"),
            "max_tokens": config.llm.get("max_tokens"),
            "thinking": config.llm.get("thinking"),
        }
    scripted_actions_sha256 = None
    if config.scripted_actions:
        scripted_actions_sha256 = _canonical_sha256(
            {
                external_id: [
                    {
                        "action_type": action.action_type,
                        "parameters": dict(action.parameters),
                        "target_ref": action.target_ref,
                        "client_ref": action.client_ref,
                    }
                    for action in actions
                ]
                for external_id, actions in sorted(config.scripted_actions.items())
            }
        )
    return {
        "worldline_manifest_version": "1",
        "engine": metadata.version("worldline-engine"),
        "social": metadata.version("worldline-social"),
        "population": {
            "sha256": _canonical_sha256(population.to_mapping()),
            "manifest_version": population.manifest_version,
            "source": population.source,
            "size": len(population.people),
        },
        "llm": llm,
        "prompt_builder_sha256": hashlib.sha256(
            Path(prompting.__file__).read_bytes()
        ).hexdigest(),
        "scripted_actions_sha256": scripted_actions_sha256,
        "experiment": {
            "config_version": config.config_version,
            "seed": config.seed,
            "max_ticks": config.max_ticks,
            "activation_probability": config.activation_probability,
            "distribution_policy": config.distribution_policy,
            "dynamics_policy": config.dynamics_policy,
            "feed_limit": config.feed_limit,
            "max_actions_per_turn": config.max_actions_per_turn,
            "max_controller_calls_per_turn": config.max_controller_calls_per_turn,
            "max_cost_per_turn": config.max_cost_per_turn,
            "turn_timeout_seconds": config.turn_timeout_seconds,
            "checkpoint_every_ticks": config.checkpoint_every_ticks,
        },
    }


def build_simulation(
    config: ExperimentConfig,
    *,
    provider: DeepSeekProvider | None = None,
    usage_recorder: UsageRecorder | None = None,
):
    manifest = PopulationManifest.from_json(config.population_manifest)
    imported = manifest.import_population()
    unknown_scripts = set(config.scripted_actions) - set(imported.external_id_mapping)
    if unknown_scripts:
        raise ValueError(
            "scripted_actions reference unknown external IDs: "
            + ", ".join(sorted(unknown_scripts))
        )
    distribution = (
        AllPostsDistribution()
        if config.distribution_policy == "all"
        else RecentPostsDistribution()
    )
    dynamics = (
        AffectiveDynamics()
        if config.dynamics_policy == "affective"
        else RecoveryDynamics()
    )
    world = SocialWorld.from_manifest(
        manifest,
        distribution_policy=distribution,
        dynamics_policy=dynamics,
        feed_limit=config.feed_limit,
    )
    if provider is None:
        provider = _create_llm_provider(config.llm) if config.llm is not None else None
    entities = []
    controllers = {}
    for external_id, person_id in sorted(imported.external_id_mapping.items()):
        controller_ref = f"controller:{person_id}"
        actions = tuple(config.scripted_actions.get(external_id, ()))
        entities.append(
            EntitySpec(person_id, controller_ref, metadata={"external_id": external_id})
        )
        if actions or provider is None:
            # Scripted actors win; without an llm config everything is replay.
            controllers[controller_ref] = ReplayController({person_id: actions})
        else:
            controllers[controller_ref] = _llm_controller(
                config, world, provider, person_id, usage_recorder=usage_recorder
            )
    scheduler = (
        AllEntitiesScheduler()
        if config.activation_probability == 1.0
        else RandomActivationScheduler(config.activation_probability)
    )
    state_store = SQLiteStateStore(config.output_database)
    event_sink = SQLiteEventSink(config.output_database)
    simulation = Simulation(
        config=config.engine_config(),
        entities=tuple(entities),
        controllers=controllers,
        scheduler=scheduler,
        world=world,
        state_store=state_store,
        event_sink=event_sink,
        manifest=build_worldline_manifest(config, manifest),
    )
    return simulation, world, state_store, event_sink


def _create_llm_provider(llm_config: Mapping[str, Any]) -> DeepSeekProvider:
    """Create a provider from config; the API key always comes from the
    environment, never from the experiment file."""
    provider_id = str(llm_config["provider"]).strip()
    if provider_id != "deepseek":
        raise ValueError(f"unsupported llm provider: {provider_id}")
    env_var = str(llm_config.get("api_key_env", "DEEPSEEK_API_KEY"))
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise ValueError(f"{env_var} is not set; required for llm provider 'deepseek'")
    return DeepSeekProvider(api_key)


def _llm_controller(
    config: ExperimentConfig,
    world: SocialWorld,
    provider: DeepSeekProvider,
    person_id: str,
    usage_recorder: UsageRecorder | None = None,
) -> LLMToolController:
    """Assemble an LLM controller whose prompt carries the live persona.

    A per-person ``model_policy.model`` overrides the experiment-wide model.
    """
    assert config.llm is not None
    person = world.state["people"][person_id]
    model = str(person.get("model_policy", {}).get("model") or config.llm["model"])
    return LLMToolController(
        provider,
        model,
        prompt_builder=SocialPromptBuilder(world),
        temperature=config.llm.get("temperature"),
        max_tokens=config.llm.get("max_tokens"),
        thinking=config.llm.get("thinking"),
        usage_recorder=usage_recorder,
    )


async def run_experiment(config: ExperimentConfig, resume: bool = False) -> ExperimentResult:
    """Run one experiment, recording per-request usage and account cost.

    When an LLM is configured, the account balance is sampled before and
    after the run (``/user/balance``) and the difference is persisted as the
    run's cost. Balance failures degrade gracefully: the run still proceeds
    and ``cost`` simply reports ``None`` for the missing sample.
    """
    started_at = time.time()
    provider = _create_llm_provider(config.llm) if config.llm is not None else None
    balance_before: BalanceInfo | None = None
    if provider is not None:
        balance_before = await _fetch_balance(provider)
    usage_store = SQLiteUsageStore(config.output_database) if provider is not None else None
    recorder = usage_store.record if usage_store is not None else None
    simulation, world, state_store, event_sink = build_simulation(
        config, provider=provider, usage_recorder=recorder
    )
    try:
        if resume:
            if not simulation.restore_latest_checkpoint():
                raise ValueError("no checkpoint exists for this simulation_id")
        await simulation.run()
        finished_at = time.time()
        balance_after: BalanceInfo | None = None
        if provider is not None:
            balance_after = await _fetch_balance(provider)
        if usage_store is not None:
            usage_store.record_run_cost(
                simulation_id=config.simulation_id,
                balance_before=_balance_total(balance_before),
                balance_after=_balance_total(balance_after),
                currency=balance_after.currency if balance_after is not None else "CNY",
                started_at=started_at,
                finished_at=finished_at,
            )
        return ExperimentResult(
            simulation_id=config.simulation_id,
            completed_ticks=simulation.current_tick,
            population_size=len(world.people),
            world_state=world.state,
            database_path=str(config.output_database),
            usage=usage_store.totals(config.simulation_id) if usage_store else {},
            cost=usage_store.latest_run_cost(config.simulation_id) if usage_store else None,
        )
    finally:
        if usage_store is not None:
            usage_store.close()
        event_sink.close()
        state_store.close()


async def _fetch_balance(provider: DeepSeekProvider) -> BalanceInfo | None:
    """Best-effort balance sample; never raises into the run."""
    try:
        return await provider.get_balance()
    except Exception:
        return None


def _balance_total(balance: BalanceInfo | None) -> str | None:
    if balance is None or not balance.is_available:
        return None
    return balance.total_balance


def run_experiment_sync(config: ExperimentConfig, resume: bool = False) -> ExperimentResult:
    return asyncio.run(run_experiment(config, resume))
