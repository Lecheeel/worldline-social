"""Run the first Worldline Social vertical slice."""

from __future__ import annotations

import asyncio

from worldline_engine import (
    ActionIntent,
    AllEntitiesScheduler,
    EntitySpec,
    MemoryEventSink,
    ReplayController,
    Simulation,
    SimulationConfig,
    InMemoryStateStore,
)

from worldline_social.world import SocialWorld


async def main() -> None:
    world = SocialWorld(("alice", "bob"))
    simulation = Simulation(
        config=SimulationConfig("social-example", max_actions_per_turn=2),
        entities=(EntitySpec("alice", "alice"), EntitySpec("bob", "bob")),
        controllers={
            "alice": ReplayController({"alice": [ActionIntent("create_post", {"content": "Hello Worldline Social"})]}),
            "bob": ReplayController({"bob": [ActionIntent("view_feed")]}),
        },
        scheduler=AllEntitiesScheduler(),
        world=world,
        state_store=InMemoryStateStore(),
        event_sink=MemoryEventSink(),
    )
    await simulation.run()
    print(world.state)


if __name__ == "__main__":
    asyncio.run(main())
