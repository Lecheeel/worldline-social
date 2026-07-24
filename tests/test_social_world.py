from __future__ import annotations

import asyncio
import unittest

from worldline_engine import (
    ActionIntent,
    AllEntitiesScheduler,
    EntitySpec,
    InMemoryStateStore,
    MemoryEventSink,
    ReplayController,
    Simulation,
    SimulationConfig,
)

from worldline_social.world import SocialWorld


POST_ID = "post-social:0:0:0"


def make_simulation(world: SocialWorld, store: InMemoryStateStore, events: MemoryEventSink) -> Simulation:
    return Simulation(
        config=SimulationConfig("social", max_ticks=2, max_actions_per_turn=1),
        entities=(EntitySpec("alice", "alice"), EntitySpec("bob", "bob")),
        controllers={
            "alice": ReplayController({"alice": [
                ActionIntent("create_post", {"content": "First post"}),
                ActionIntent("create_comment", {"post_id": POST_ID, "content": "Follow-up"}),
            ]}),
            "bob": ReplayController({"bob": [
                ActionIntent("do_nothing"),
                ActionIntent("like_post", {"post_id": POST_ID}),
            ]}),
        },
        scheduler=AllEntitiesScheduler(),
        world=world,
        state_store=store,
        event_sink=events,
    )


class SocialWorldTests(unittest.TestCase):
    def test_checkpoint_restore_matches_continuous_social_run(self) -> None:
        continuous_world = SocialWorld(("alice", "bob"))
        asyncio.run(make_simulation(continuous_world, InMemoryStateStore(), MemoryEventSink()).run())

        store = InMemoryStateStore()
        first_world = SocialWorld(("alice", "bob"))
        asyncio.run(make_simulation(first_world, store, MemoryEventSink()).run(ticks=1))

        resumed_world = SocialWorld(("alice", "bob"))
        resumed = make_simulation(resumed_world, store, MemoryEventSink())
        self.assertTrue(resumed.restore_latest_checkpoint())
        asyncio.run(resumed.run())

        self.assertEqual(continuous_world.state, resumed_world.state)
        self.assertEqual(1, resumed_world.state["posts"][POST_ID]["like_count"])
        self.assertEqual(1, len(resumed_world.state["comments"]))

    def test_feed_read_does_not_mutate_world(self) -> None:
        world = SocialWorld(("alice",))
        simulation = Simulation(
            config=SimulationConfig("feed", max_actions_per_turn=1),
            entities=(EntitySpec("alice", "alice"),),
            controllers={"alice": ReplayController({"alice": [ActionIntent("view_feed")]})},
            scheduler=AllEntitiesScheduler(),
            world=world,
            state_store=InMemoryStateStore(),
            event_sink=MemoryEventSink(),
        )
        asyncio.run(simulation.run())
        self.assertEqual({"posts": {}, "comments": {}, "likes": []}, world.state)


if __name__ == "__main__":
    unittest.main()
