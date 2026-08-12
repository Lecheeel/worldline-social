from __future__ import annotations

import unittest

from worldline_engine import ActionKind, ActionSpec
from worldline_engine.protocols import TurnContext

from worldline_social.prompting import (
    SocialPromptBuilder,
    render_observation_message,
    render_person_system_message,
)

PERSON = {
    "person_id": "person-alice",
    "handle": "alice",
    "display_name": "Alice",
    "bio": "Interested in local energy policy",
    "private_traits": {"openness": 0.8, "neuroticism": 0.7},
    "dynamic_state": {"mood": -0.3, "anger": 0.2, "stress": 0.5, "fatigue": 0.0, "threat": 0.1},
    "model_policy": {
        "persona": "Alice is a cautious researcher who values evidence.",
        "stance": "opposing",
        "interested_topics": ["solar energy", "local policy"],
        "personal_memory": "She attended last week's hearing and was disappointed.",
        "age": 35,
        "gender": "female",
    },
}


def context() -> TurnContext:
    return TurnContext(
        simulation_id="test",
        tick_id=0,
        turn_id="0:0:alice",
        entity_id="person-alice",
        observation={"feed": [{"post_id": "p1", "content": "hello"}]},
        available_actions=(ActionSpec("view_feed", ActionKind.READ),),
        previous_result=None,
        remaining_actions=3,
        remaining_controller_calls=3,
    )


class PersonaRenderingTests(unittest.TestCase):
    def test_system_message_carries_identity_persona_and_stance(self) -> None:
        message = render_person_system_message(PERSON)

        self.assertIn("你是 Alice", message)
        self.assertIn("@alice", message)
        self.assertIn("cautious researcher", message)
        self.assertIn("反对", message)
        self.assertIn("solar energy", message)
        self.assertIn("last week's hearing", message)

    def test_system_message_carries_live_state_and_traits(self) -> None:
        message = render_person_system_message(PERSON)

        self.assertIn("情绪（-1 负面 ~ +1 正面）: -0.3", message)
        self.assertIn("神经质: 0.7", message)
        self.assertIn("开放性: 0.8", message)

    def test_missing_fields_are_tolerated(self) -> None:
        message = render_person_system_message({"handle": "bob"})
        self.assertIn("@bob", message)


class ObservationRenderingTests(unittest.TestCase):
    def test_observation_includes_previous_result_when_present(self) -> None:
        ctx = context()
        payload = render_observation_message(ctx)
        self.assertIn("hello", payload)
        self.assertIn("remaining_actions", payload)


class SocialPromptBuilderTests(unittest.TestCase):
    def test_builder_reads_live_person_from_world_state(self) -> None:
        class FakeWorld:
            def __init__(self) -> None:
                self.state = {"people": {"person-alice": PERSON}}

        builder = SocialPromptBuilder(FakeWorld())
        messages = builder(context())

        self.assertEqual(2, len(messages))
        self.assertEqual("system", messages[0].role)
        self.assertIn("Alice", messages[0].content)
        self.assertEqual("user", messages[1].role)
        self.assertIn("hello", messages[1].content)

    def test_builder_falls_back_for_unknown_entity(self) -> None:
        class FakeWorld:
            state = {"people": {}}

        messages = SocialPromptBuilder(FakeWorld())(context())
        self.assertIn("@person-alice", messages[0].content)


if __name__ == "__main__":
    unittest.main()
