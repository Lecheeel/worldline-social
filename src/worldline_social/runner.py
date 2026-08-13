"""Composition root for configured Worldline Social experiments."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
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

from .controllers import LLMToolController
from .distribution import AllPostsDistribution, RecentPostsDistribution
from .dynamics import AffectiveDynamics, RecoveryDynamics
from .experiment import ExperimentConfig
from .population import PopulationManifest
from .prompting import SocialPromptBuilder
from .providers import DeepSeekProvider
from .world import SocialWorld


@dataclass(frozen=True)
class ExperimentResult:
    simulation_id: str
    completed_ticks: int
    population_size: int
    world_state: dict[str, Any]
    database_path: str


def build_simulation(config: ExperimentConfig):
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
            controllers[controller_ref] = _llm_controller(config, world, provider, person_id)
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
    )


async def run_experiment(config: ExperimentConfig, resume: bool = False) -> ExperimentResult:
    simulation, world, state_store, event_sink = build_simulation(config)
    try:
        if resume:
            if not simulation.restore_latest_checkpoint():
                raise ValueError("no checkpoint exists for this simulation_id")
        await simulation.run()
        return ExperimentResult(
            simulation_id=config.simulation_id,
            completed_ticks=simulation.current_tick,
            population_size=len(world.people),
            world_state=world.state,
            database_path=str(config.output_database),
        )
    finally:
        event_sink.close()
        state_store.close()


def run_experiment_sync(config: ExperimentConfig, resume: bool = False) -> ExperimentResult:
    return asyncio.run(run_experiment(config, resume))
