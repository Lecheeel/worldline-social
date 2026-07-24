from __future__ import annotations

import asyncio
import unittest

from worldline_engine import (
    AllEntitiesScheduler,
    EntitySpec,
    InMemoryStateStore,
    MemoryEventSink,
    ReplayController,
    Simulation,
    SimulationConfig,
)

from worldline_social.dynamics import DynamicState, TraitProfile
from worldline_social.population import PopulationManifest
from worldline_social.world import SocialWorld


class SocialDynamicsTests(unittest.TestCase):
    def test_traits_and_dynamic_state_have_bounded_ranges(self) -> None:
        TraitProfile(openness=1.0, psychopathy=0.2)
        DynamicState(mood=-1.0, anger=1.0)
        with self.assertRaises(ValueError):
            TraitProfile(openness=1.1)
        with self.assertRaises(ValueError):
            DynamicState(stress=-0.1)

    def test_world_advances_dynamic_state_once_per_tick(self) -> None:
        manifest = PopulationManifest.from_mapping(
            {
                "manifest_version": "1",
                "source": "dynamics-test",
                "people": [
                    {
                        "external_id": "alice-source",
                        "handle": "alice",
                        "private_traits": TraitProfile(openness=0.8).to_mapping(),
                        "initial_state": DynamicState(
                            mood=-0.5,
                            anger=0.4,
                            stress=0.3,
                        ).to_mapping(),
                    }
                ],
            }
        )
        world = SocialWorld.from_manifest(manifest)
        person_id = manifest.import_population().external_id_mapping["alice-source"]
        simulation = Simulation(
            config=SimulationConfig("dynamics", max_ticks=2),
            entities=(EntitySpec(person_id, "controller"),),
            controllers={"controller": ReplayController({person_id: []})},
            scheduler=AllEntitiesScheduler(),
            world=world,
            state_store=InMemoryStateStore(),
            event_sink=MemoryEventSink(),
        )
        asyncio.run(simulation.run())

        state = world.state["people"][person_id]["dynamic_state"]
        self.assertAlmostEqual(-0.34, state["mood"])
        self.assertAlmostEqual(0.3, state["anger"])
        self.assertAlmostEqual(0.22, state["stress"])


if __name__ == "__main__":
    unittest.main()
