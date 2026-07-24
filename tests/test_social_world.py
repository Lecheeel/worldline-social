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
    RuleController,
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
    def test_concurrency_does_not_change_social_state_or_events(self) -> None:
        def run(concurrency):
            world = SocialWorld(("alice", "bob"))
            sink = MemoryEventSink()
            simulation = Simulation(
                config=SimulationConfig(
                    "social-concurrency",
                    max_concurrent_turns=concurrency,
                    max_actions_per_turn=1,
                ),
                entities=(EntitySpec("alice", "alice"), EntitySpec("bob", "bob")),
                controllers={
                    "alice": ReplayController(
                        {"alice": [ActionIntent("create_post", {"content": "Alice"})]}
                    ),
                    "bob": ReplayController(
                        {"bob": [ActionIntent("create_post", {"content": "Bob"})]}
                    ),
                },
                scheduler=AllEntitiesScheduler(),
                world=world,
                state_store=InMemoryStateStore(),
                event_sink=sink,
            )
            asyncio.run(simulation.run())
            return world.state, [
                (event.event_type, event.payload) for event in sink.events
            ]

        sequential = run(1)
        concurrent = run(2)
        self.assertEqual(sequential, concurrent)

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
        self.assertEqual({}, world.state["posts"])
        self.assertEqual({}, world.state["comments"])
        self.assertEqual([], world.state["post_likes"])

    def test_turn_can_read_and_reference_its_local_post(self) -> None:
        def rule(context):
            if context.previous_result is None:
                return ActionIntent("create_post", {"content": "Local post"})
            return ActionIntent(
                "create_comment",
                {
                    "post_id": context.previous_result.local_ref,
                    "content": "Local comment",
                },
            )

        world = SocialWorld(("alice",))
        simulation = Simulation(
            config=SimulationConfig("local-social", max_actions_per_turn=2),
            entities=(EntitySpec("alice", "alice"),),
            controllers={"alice": RuleController(rule)},
            scheduler=AllEntitiesScheduler(),
            world=world,
            state_store=InMemoryStateStore(),
            event_sink=MemoryEventSink(),
        )
        asyncio.run(simulation.run())

        self.assertEqual(1, len(world.state["posts"]))
        comment = next(iter(world.state["comments"].values()))
        self.assertEqual(next(iter(world.state["posts"])), comment["post_id"])

    def test_search_and_thread_reads_are_structured(self) -> None:
        world = SocialWorld(("alice",))
        first = Simulation(
            config=SimulationConfig("search-seed"),
            entities=(EntitySpec("alice", "alice"),),
            controllers={
                "alice": ReplayController(
                    {"alice": [ActionIntent("create_post", {"content": "Solar evidence"})]}
                )
            },
            scheduler=AllEntitiesScheduler(),
            world=world,
            state_store=InMemoryStateStore(),
            event_sink=MemoryEventSink(),
        )
        asyncio.run(first.run())
        post_id = next(iter(world.state["posts"]))
        sink = MemoryEventSink()
        second = Simulation(
            config=SimulationConfig("search-read", max_actions_per_turn=2),
            entities=(EntitySpec("alice", "alice"),),
            controllers={
                "alice": ReplayController(
                    {
                        "alice": [
                            ActionIntent("search_square", {"query": "solar"}),
                            ActionIntent("view_thread", {"post_id": post_id}),
                        ]
                    }
                )
            },
            scheduler=AllEntitiesScheduler(),
            world=world,
            state_store=InMemoryStateStore(),
            event_sink=sink,
        )
        asyncio.run(second.run())

        reads = [event for event in sink.events if event.event_type == "action_read"]
        self.assertEqual("post", reads[0].payload["result"]["data"]["results"][0]["result_type"])
        self.assertEqual("alice", reads[0].payload["result"]["data"]["results"][0]["author_handle"])
        self.assertNotIn("author_person_id", reads[0].payload["result"]["data"]["results"][0])
        self.assertEqual(post_id, reads[1].payload["result"]["data"]["post"]["post_id"])


if __name__ == "__main__":
    unittest.main()
