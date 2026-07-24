"""Composition root for configured Worldline Social experiments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from worldline_engine import (
    AllEntitiesScheduler,
    EntitySpec,
    RandomActivationScheduler,
    ReplayController,
    SQLiteEventSink,
    SQLiteStateStore,
    Simulation,
)

from .distribution import AllPostsDistribution, RecentPostsDistribution
from .experiment import ExperimentConfig
from .population import PopulationManifest
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
    world = SocialWorld.from_manifest(
        manifest,
        distribution_policy=distribution,
        feed_limit=config.feed_limit,
    )
    entities = []
    controllers = {}
    for external_id, person_id in sorted(imported.external_id_mapping.items()):
        controller_ref = f"controller:{person_id}"
        actions = tuple(config.scripted_actions.get(external_id, ()))
        entities.append(EntitySpec(person_id, controller_ref, metadata={"external_id": external_id}))
        controllers[controller_ref] = ReplayController({person_id: actions})
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
