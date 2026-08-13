from __future__ import annotations

import asyncio
import unittest

from worldline_engine import ActionKind, ActionSpec
from worldline_social import LLMToolController
from worldline_engine.protocols import FinishTurn, TurnContext
from worldline_social.providers import (
    CompletionResponse,
    DeepSeekProvider,
    ModelToolCall,
    builtin_provider_registry,
)


class FakeProvider:
    provider_id = "fake"

    def __init__(self, response: CompletionResponse) -> None:
        self.response = response
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.response


def context() -> TurnContext:
    return TurnContext(
        simulation_id="test",
        tick_id=0,
        turn_id="0:0:alice",
        entity_id="alice",
        observation={"value": 0},
        available_actions=(
            ActionSpec(
                "claim",
                ActionKind.WRITE,
                "Claim the one available resource.",
                {"type": "object", "properties": {}},
            ),
        ),
        previous_result=None,
        remaining_actions=2,
        remaining_controller_calls=2,
    )


class ProviderTests(unittest.TestCase):
    def test_llm_controller_normalizes_provider_tool_call(self) -> None:
        provider = FakeProvider(
            CompletionResponse(
                content=None,
                tool_calls=(ModelToolCall("provider-call-1", "claim", {}),),
                usage={"prompt_tokens": 12, "completion_tokens": 3},
            )
        )
        controller = LLMToolController(provider, "fake-model")

        decision = asyncio.run(controller.next_action(context()))

        self.assertEqual("claim", decision.action_type)
        self.assertEqual("provider-call-1", decision.client_ref)
        self.assertEqual("claim", provider.requests[0].tools[0].name)
        self.assertEqual(
            {"request_count": 1, "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            controller.dump_state(),
        )

    def test_llm_controller_finishes_without_a_tool_call(self) -> None:
        controller = LLMToolController(FakeProvider(CompletionResponse("No action.")), "fake-model")

        decision = asyncio.run(controller.next_action(context()))

        self.assertIsInstance(decision, FinishTurn)
        self.assertEqual("model_finished_without_tool_call", decision.reason)

    def test_llm_controller_supports_async_prompt_builder(self) -> None:
        provider = FakeProvider(
            CompletionResponse(None, (ModelToolCall("call", "claim", {}),))
        )

        async def prompt_builder(_context):
            return ()

        controller = LLMToolController(provider, "fake-model", prompt_builder)
        decision = asyncio.run(controller.next_action(context()))

        self.assertEqual("claim", decision.action_type)
        self.assertEqual((), provider.requests[0].messages)

    def test_llm_controller_forwards_thinking_mode(self) -> None:
        provider = FakeProvider(
            CompletionResponse(None, (ModelToolCall("call", "claim", {}),))
        )
        controller = LLMToolController(provider, "fake-model", thinking="disabled")

        asyncio.run(controller.next_action(context()))

        self.assertEqual("disabled", provider.requests[0].thinking)

    def test_builtin_registry_creates_deepseek_provider_without_exposing_key(self) -> None:
        registry = builtin_provider_registry()
        provider = registry.create("deepseek", {"api_key": "test-key"})

        self.assertEqual(("deepseek",), registry.provider_ids)
        self.assertEqual("deepseek", provider.provider_id)
        self.assertFalse(hasattr(provider, "api_key"))
        self.assertNotIn("test-key", repr(provider))

    def test_deepseek_normalizes_compact_action_schema(self) -> None:
        tool = ActionSpec("add", ActionKind.WRITE, parameters_schema={"required": ["amount"]})

        schema = DeepSeekProvider._tool_schema(tool)

        self.assertEqual("object", schema["function"]["parameters"]["type"])
        self.assertEqual({}, schema["function"]["parameters"]["properties"])
        self.assertEqual(["amount"], schema["function"]["parameters"]["required"])
