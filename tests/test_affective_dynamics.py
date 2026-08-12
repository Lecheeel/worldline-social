from __future__ import annotations

import unittest

from worldline_engine import ActionIntent
from worldline_engine.protocols import BoundAction

from worldline_social.dynamics import (
    AffectiveDynamics,
    DynamicState,
    RecoveryDynamics,
    TraitProfile,
)
from worldline_social.world import SocialWorld

PERSON_IDS = ("alice", "bob")


def bound(entity: str, action_id: str, action_type: str, **params) -> BoundAction:
    return BoundAction(
        action_id=action_id,
        simulation_id="test",
        tick_id=0,
        turn_id="t0",
        entity_id=entity,
        turn_index=0,
        action_index=0,
        intent=ActionIntent(action_type, params),
    )


def alice_person_id(world: SocialWorld) -> str:
    return world.state["people"]["alice"]["person_id"]


class AffectiveDynamicsUnitTests(unittest.TestCase):
    def test_received_like_raises_mood(self) -> None:
        dynamics = AffectiveDynamics()
        result = dynamics.apply_feedback(
            "alice",
            TraitProfile().to_mapping(),
            DynamicState().to_mapping(),
            ({"kind": "received_like", "count": 1},),
        )
        self.assertGreater(result["mood"], 0.0)

    def test_received_unlike_lowers_mood_and_raises_anger(self) -> None:
        dynamics = AffectiveDynamics()
        result = dynamics.apply_feedback(
            "bob",
            TraitProfile().to_mapping(),
            DynamicState().to_mapping(),
            ({"kind": "received_unlike", "count": 2},),
        )
        self.assertLess(result["mood"], 0.0)
        self.assertGreater(result["anger"], 0.0)

    def test_neurotic_person_reacts_stronger_to_negative_feedback(self) -> None:
        dynamics = AffectiveDynamics()
        calm = dynamics.apply_feedback(
            "a", TraitProfile(neuroticism=0.1).to_mapping(),
            DynamicState().to_mapping(), ({"kind": "read_negative"},),
        )
        anxious = dynamics.apply_feedback(
            "b", TraitProfile(neuroticism=0.9).to_mapping(),
            DynamicState().to_mapping(), ({"kind": "read_negative"},),
        )
        self.assertGreater(anxious["stress"], calm["stress"])

    def test_psychopathy_insulates_from_negative_feedback(self) -> None:
        dynamics = AffectiveDynamics()
        sensitive = dynamics.apply_feedback(
            "a", TraitProfile(psychopathy=0.0).to_mapping(),
            DynamicState().to_mapping(), ({"kind": "received_unlike"},),
        )
        callous = dynamics.apply_feedback(
            "b", TraitProfile(psychopathy=1.0).to_mapping(),
            DynamicState().to_mapping(), ({"kind": "received_unlike"},),
        )
        self.assertGreater(sensitive["anger"], callous["anger"])
        self.assertLess(sensitive["mood"], callous["mood"])

    def test_values_are_clamped(self) -> None:
        dynamics = AffectiveDynamics()
        result = dynamics.apply_feedback(
            "a",
            TraitProfile(neuroticism=1.0).to_mapping(),
            DynamicState(mood=0.9, anger=0.9).to_mapping(),
            ({"kind": "received_unlike", "count": 100},),
        )
        self.assertGreaterEqual(result["mood"], -1.0)
        self.assertLessEqual(result["anger"], 1.0)

    def test_recovery_dynamics_ignores_feedback(self) -> None:
        dynamics = RecoveryDynamics()
        result = dynamics.apply_feedback(
            "a", {}, DynamicState(mood=0.5).to_mapping(),
            ({"kind": "received_like"},),
        )
        self.assertEqual(0.5, result["mood"])


class WorldFeedbackIntegrationTests(unittest.TestCase):
    def test_posting_and_liking_move_author_mood(self) -> None:
        world = SocialWorld(PERSON_IDS, dynamics_policy=AffectiveDynamics())
        snapshot = world.snapshot()
        world.resolve_and_apply(
            snapshot,
            (
                bound("alice", "a1", "create_post", content="hello world"),
                bound("bob", "a2", "like_post", post_id="post-a1"),
            ),
        )
        state = world.state
        alice = state["people"]["alice"]
        bob = state["people"]["bob"]
        self.assertGreater(alice["dynamic_state"]["mood"], 0.0)
        self.assertEqual(0.0, bob["dynamic_state"]["mood"])

    def test_comment_author_receives_feedback_from_reply(self) -> None:
        world = SocialWorld(PERSON_IDS, dynamics_policy=AffectiveDynamics())
        snapshot = world.snapshot()
        world.resolve_and_apply(
            snapshot,
            (
                bound("alice", "a1", "create_post", content="hello"),
                bound("bob", "a2", "create_comment", post_id="post-a1", content="nice"),
                bound("alice", "a3", "reply_comment", comment_id="comment-a2", content="thanks"),
            ),
        )
        bob = world.state["people"]["bob"]
        self.assertGreater(bob["dynamic_state"]["stress"], 0.0)

    def test_tick_advance_still_recovers_states(self) -> None:
        world = SocialWorld(PERSON_IDS, dynamics_policy=AffectiveDynamics())
        world.resolve_and_apply(
            world.snapshot(),
            (bound("alice", "a1", "create_post", content="hello"),),
        )
        before = world.state["people"]["alice"]["dynamic_state"]["mood"]
        world.advance_tick(1)
        after = world.state["people"]["alice"]["dynamic_state"]["mood"]
        self.assertLess(after, before)

    def test_engine_level_run_is_deterministic_and_moves_mood(self) -> None:
        import asyncio

        from worldline_engine import (
            AllEntitiesScheduler,
            EntitySpec,
            InMemoryStateStore,
            MemoryEventSink,
            ReplayController,
            Simulation,
            SimulationConfig,
        )
        from worldline_social.population import PopulationManifest

        manifest = PopulationManifest.from_mapping(
            {
                "manifest_version": "1",
                "source": "engine-feedback-test",
                "people": [
                    {
                        "external_id": "alice",
                        "handle": "alice",
                        "private_traits": TraitProfile(extraversion=0.8).to_mapping(),
                        "initial_state": DynamicState().to_mapping(),
                    },
                    {
                        "external_id": "bob",
                        "handle": "bob",
                        "private_traits": TraitProfile().to_mapping(),
                        "initial_state": DynamicState().to_mapping(),
                    },
                ],
            }
        )
        imported = manifest.import_population()
        alice = imported.external_id_mapping["alice"]
        bob = imported.external_id_mapping["bob"]

        def run_once() -> dict:
            world = SocialWorld.from_manifest(manifest, dynamics_policy=AffectiveDynamics())
            simulation = Simulation(
                config=SimulationConfig("feedback-run", max_ticks=1, seed=7),
                entities=(EntitySpec(alice, "c1"), EntitySpec(bob, "c2")),
                controllers={
                    "c1": ReplayController(
                        {alice: [ActionIntent("create_post", {"content": "hello"})]}
                    ),
                    "c2": ReplayController({bob: [ActionIntent("view_feed")]}),
                },
                scheduler=AllEntitiesScheduler(),
                world=world,
                state_store=InMemoryStateStore(),
                event_sink=MemoryEventSink(),
            )
            asyncio.run(simulation.run())
            return dict(world.state["people"][alice]["dynamic_state"])

        first = run_once()
        second = run_once()
        self.assertGreater(first["mood"], 0.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
