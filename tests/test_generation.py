from __future__ import annotations

import asyncio
import unittest

from worldline_social.generation import (
    EventExtractor,
    generate_population_from_text,
    generate_population_from_text_sync,
)
from worldline_social.generation.extract import Participant, Relationship
from worldline_social.generation.json_utils import extract_json_object
from worldline_social.generation.profiles import ProfileGenerator
from worldline_social.providers.base import CompletionResponse


class ScriptedProvider:
    """Fake provider returning queued responses, no network involved."""

    provider_id = "fake"

    def __init__(self, *responses: CompletionResponse) -> None:
        self._responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not self._responses:
            return CompletionResponse("{}")
        return self._responses.pop(0)


EXTRACTION_JSON = """{
  "participants": [
    {"name": "张教授", "entity_type": "person", "role": "涉事学者",
     "summary": "被指控学术不端的大学教授", "stance": "opposing"},
    {"name": "大学", "entity_type": "organization", "role": "涉事机构",
     "summary": "事件发生的大学", "stance": "neutral"},
    {"name": "媒体", "entity_type": "media", "role": "报道方",
     "summary": "报道事件的媒体", "stance": "observer"}
  ],
  "relationships": [
    {"source": "张教授", "target": "大学", "relationship_type": "employer_of",
     "description": "张教授受雇于大学"}
  ],
  "initial_spark": "知名高校教授被举报学术不端，校方表示将展开调查。"
}"""

PROFILE_JSON = """{
  "display_name": "张教授",
  "bio": "大学教师，专注科研二十余年。",
  "persona": "张教授是资深学者，性格严谨内向，面对质疑反应强烈，重视学术声誉。",
  "traits": {
    "openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.3,
    "agreeableness": 0.4, "neuroticism": 0.7, "honesty_humility": 0.6,
    "machiavellianism": 0.1, "narcissism": 0.3, "psychopathy": 0.0
  },
  "stance": "opposing",
  "interested_topics": ["学术诚信", "教育改革"],
  "personal_memory": "已经公开发声否认指控，准备法律途径维权。",
  "age": 52, "gender": "male", "country": "中国", "profession": "大学教授"
}"""

EVENT_TEXT = (
    "某知名大学张教授被学生举报学术不端，校方宣布成立调查组。"
    "张教授公开否认指控，称这是恶意诽谤。多家媒体跟进报道。"
)


class JsonExtractionTests(unittest.TestCase):
    def test_extracts_object_from_fenced_content(self) -> None:
        value = extract_json_object('```json\n{"a": 1}\n```')
        self.assertEqual({"a": 1}, value)

    def test_extracts_object_from_prose(self) -> None:
        value = extract_json_object('结果如下：{"a": {"b": 2}} 完')
        self.assertEqual({"a": {"b": 2}}, value)

    def test_recovers_truncated_object(self) -> None:
        value = extract_json_object('{"a": 1, "b": [1, 2')
        self.assertIsNotNone(value)
        self.assertEqual(1, value.get("a"))

    def test_returns_none_for_garbage(self) -> None:
        self.assertIsNone(extract_json_object("没有 JSON"))


class EventExtractorTests(unittest.TestCase):
    def test_extracts_participants_and_relationships(self) -> None:
        provider = ScriptedProvider(CompletionResponse(EXTRACTION_JSON))
        extractor = EventExtractor(provider, "fake-model")

        result = asyncio.run(extractor.extract(EVENT_TEXT))

        self.assertEqual(3, len(result.participants))
        self.assertEqual("张教授", result.participants[0].name)
        self.assertEqual("opposing", result.participants[0].stance)
        self.assertEqual("organization", result.participants[1].entity_type)
        self.assertEqual(1, len(result.relationships))
        self.assertEqual("employer_of", result.relationships[0].relationship_type)
        self.assertEqual("知名高校教授被举报学术不端，校方表示将展开调查。", result.initial_spark)
        self.assertTrue(result.diagnostics["raw_parseable"])

    def test_drops_relationships_referencing_unknown_actors(self) -> None:
        provider = ScriptedProvider(
            CompletionResponse(
                '{"participants": [{"name": "A", "entity_type": "person"}],'
                ' "relationships": [{"source": "A", "target": "幽灵",'
                ' "relationship_type": "follow"}], "initial_spark": ""}'
            )
        )
        result = asyncio.run(EventExtractor(provider, "m").extract("事件"))
        self.assertEqual(0, len(result.relationships))

    def test_returns_empty_result_when_llm_output_is_unparseable(self) -> None:
        provider = ScriptedProvider(
            CompletionResponse("抱歉，我无法理解。"),
            CompletionResponse("抱歉，我无法理解。"),
            CompletionResponse("抱歉，我无法理解。"),
        )
        result = asyncio.run(EventExtractor(provider, "m", max_attempts=3).extract("事件"))
        self.assertEqual(0, len(result.participants))
        self.assertFalse(result.diagnostics["raw_parseable"])
        self.assertEqual(3, len(provider.requests))


class ProfileGeneratorTests(unittest.TestCase):
    def test_generates_profile_with_normalized_traits(self) -> None:
        participant = Participant(
            name="张教授", entity_type="person", role="涉事学者",
            summary="被指控学术不端的大学教授", stance="opposing",
        )
        provider = ScriptedProvider(CompletionResponse(PROFILE_JSON))

        profile = asyncio.run(
            ProfileGenerator(provider, "fake-model").generate(
                participant, "zhang-00", EVENT_TEXT
            )
        )

        self.assertEqual("张教授", profile.display_name)
        self.assertEqual(0.9, profile.private_traits["conscientiousness"])
        self.assertEqual(0.0, profile.private_traits["psychopathy"])
        self.assertEqual("opposing", profile.model_policy["stance"])
        self.assertEqual(-0.2, profile.initial_state["mood"])
        self.assertEqual(0.3, profile.initial_state["anger"])
        self.assertRegex(profile.handle, r"^[a-z0-9_]+$")

    def test_falls_back_to_rule_based_profile(self) -> None:
        participant = Participant(
            name="大学", entity_type="organization", role="涉事机构",
            summary="事件发生的大学", stance="neutral",
        )
        provider = ScriptedProvider(CompletionResponse("失败"))

        profile = asyncio.run(
            ProfileGenerator(provider, "m", max_attempts=1).generate(
                participant, "uni-00", ""
            )
        )

        self.assertEqual("大学", profile.display_name)
        self.assertIn("官方账号", profile.bio)
        self.assertEqual(0.0, profile.initial_state["mood"])

    def test_handles_percent_scale_traits(self) -> None:
        participant = Participant(name="A", entity_type="person")
        provider = ScriptedProvider(
            CompletionResponse(
                '{"display_name": "A", "traits": {"openness": 80, "neuroticism": "高"},'
                ' "stance": "observer", "interested_topics": ["x"]}'
            )
        )
        profile = asyncio.run(
            ProfileGenerator(provider, "m", max_attempts=1).generate(
                participant, "a-00", ""
            )
        )
        self.assertEqual(1.0, profile.private_traits["openness"])
        self.assertEqual(0.8, profile.private_traits["neuroticism"])


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_builds_validated_manifest(self) -> None:
        provider = ScriptedProvider(
            CompletionResponse(EXTRACTION_JSON, usage={"prompt_tokens": 100, "completion_tokens": 40, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100}),
            CompletionResponse(PROFILE_JSON, usage={"prompt_tokens": 50, "completion_tokens": 20, "prompt_cache_hit_tokens": 50, "prompt_cache_miss_tokens": 0}),
            CompletionResponse(
                '{"display_name": "大学", "bio": "官方账号。", "traits": {},'
                ' "stance": "neutral", "interested_topics": []}',
                usage={"prompt_tokens": 30, "completion_tokens": 10},
            ),
            CompletionResponse(
                '{"display_name": "媒体", "bio": "媒体账号。", "traits": {},'
                ' "stance": "observer", "interested_topics": []}',
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            ),
        )

        result = asyncio.run(
            generate_population_from_text(EVENT_TEXT, provider, "fake-model")
        )

        manifest = result.manifest
        self.assertEqual(3, len(manifest.people))
        self.assertEqual(1, len(manifest.relationships))
        self.assertEqual(1, len(manifest.initial_content))
        self.assertEqual(
            manifest.initial_content[0]["external_id"],
            manifest.relationships[0].source_external_id,
        )
        self.assertEqual("worldline-social.generation.pipeline",
                         manifest.generation_metadata["generator"])
        # usage totals across extraction + all profile calls
        usage = manifest.generation_metadata["usage"]
        self.assertEqual(200, usage["prompt_tokens"])
        self.assertEqual(75, usage["completion_tokens"])
        self.assertEqual(50, usage["prompt_cache_hit_tokens"])
        self.assertEqual(100, usage["prompt_cache_miss_tokens"])
        self.assertEqual(usage, result.diagnostics["usage"])
        # handles unique and valid
        handles = [person.handle for person in manifest.people]
        self.assertEqual(len(handles), len(set(handles)))
        for handle in handles:
            self.assertRegex(handle, r"^[a-z0-9_]+$")

    def test_sync_wrapper_and_json_export(self) -> None:
        provider = ScriptedProvider(
            CompletionResponse(
                '{"participants": [{"name": "A", "entity_type": "person"}],'
                ' "relationships": [], "initial_spark": "start"}'
            ),
            CompletionResponse(
                '{"display_name": "A", "bio": "b", "traits": {}, "stance": "neutral",'
                ' "interested_topics": []}'
            ),
        )
        from worldline_social.generation import result_to_json

        result = generate_population_from_text_sync("事件", provider, "m")
        payload = result_to_json(result)
        self.assertEqual(1, len(payload["people"]))
        self.assertEqual("start", payload["initial_content"][0]["content"])

    def test_generated_manifest_seeds_live_world_posts(self) -> None:
        from worldline_social.world import SocialWorld

        provider = ScriptedProvider(
            CompletionResponse(EXTRACTION_JSON),
            CompletionResponse(PROFILE_JSON),
            CompletionResponse(
                '{"display_name": "大学", "bio": "官方账号。", "traits": {},'
                ' "stance": "neutral", "interested_topics": []}'
            ),
            CompletionResponse(
                '{"display_name": "媒体", "bio": "媒体账号。", "traits": {},'
                ' "stance": "observer", "interested_topics": []}'
            ),
        )
        result = generate_population_from_text_sync(EVENT_TEXT, provider, "fake-model")

        world = SocialWorld.from_manifest(result.manifest)
        posts = world.state["posts"]
        self.assertEqual(1, len(posts))
        post = next(iter(posts.values()))
        self.assertIn("学术不端", post["content"])
        self.assertEqual(0, post["created_tick"])


if __name__ == "__main__":
    unittest.main()
