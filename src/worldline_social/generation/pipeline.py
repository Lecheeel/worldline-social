"""Composition root: turn event text into a validated PopulationManifest.

The pipeline only produces the population contract; it never touches the
engine or world internals. Downstream consumers import the manifest through
the same path as hand-written populations.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..population import PopulationManifest, RelationshipSpec
from ..providers.base import ModelProvider
from .extract import EventExtractor, Participant, Relationship
from .profiles import ProfileGenerator

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class GenerationResult:
    """A validated manifest plus the diagnostics behind it."""

    manifest: PopulationManifest
    participants: tuple[Participant, ...]
    relationships: tuple[Relationship, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


async def generate_population_from_text(
    text: str,
    provider: ModelProvider,
    model: str,
    *,
    source: str = "event-generated",
    max_participants: int = 30,
    include_initial_content: bool = True,
    extraction_temperature: float | None = 0.2,
    profile_temperature: float | None = 0.4,
    max_attempts: int = 3,
) -> GenerationResult:
    """Extract actors, build profiles and return a validated manifest.

    The first participant is treated as the central actor and, when the
    extractor produced an ``initial_spark``, that spark becomes the first
    post of the simulated world (see ``initial_content`` consumption in
    ``SocialWorld``).
    """
    if not text.strip():
        raise ValueError("event text must not be empty")
    if max_participants < 1:
        raise ValueError("max_participants must be positive")

    extractor = EventExtractor(provider, model, extraction_temperature, max_attempts)
    extraction = await extractor.extract(text)
    participants = tuple(extraction.participants[:max_participants])

    profile_generator = ProfileGenerator(provider, model, profile_temperature, max_attempts)
    people: list[PersonProfile] = []
    name_to_external: dict[str, str] = {}
    for index, participant in enumerate(participants):
        external_id = _external_id(participant.name, index)
        name_to_external[participant.name] = external_id
        profile = await profile_generator.generate(participant, external_id, text)
        people.append(profile)

    relationships = tuple(
        RelationshipSpec(
            source_external_id=name_to_external[item.source],
            target_external_id=name_to_external[item.target],
            relationship_type=item.relationship_type,
            strength=1.0,
        )
        for item in extraction.relationships
        if item.source in name_to_external and item.target in name_to_external
    )

    initial_content: list[dict[str, Any]] = []
    if include_initial_content and extraction.initial_spark and people:
        initial_content.append(
            {
                "external_id": name_to_external[participants[0].name],
                "content": extraction.initial_spark,
            }
        )

    manifest = PopulationManifest(
        manifest_version="1",
        source=source,
        people=tuple(people),
        relationships=relationships,
        generation_metadata={
            "description": "Generated from event text by worldline-social.generation",
            "generator": "worldline-social.generation.pipeline",
            "model": model,
            "source_text_chars": len(text),
            "participant_count": len(people),
            "relationship_count": len(relationships),
            "extraction_diagnostics": dict(extraction.diagnostics),
        },
        initial_content=tuple(initial_content),
    )
    manifest.validate()
    return GenerationResult(
        manifest=manifest,
        participants=participants,
        relationships=extraction.relationships,
        diagnostics={
            "participants_requested": len(extraction.participants),
            "participants_kept": len(people),
            "relationships_kept": len(relationships),
            "initial_content": len(initial_content),
            "extraction": dict(extraction.diagnostics),
        },
    )


def generate_population_from_text_sync(
    text: str,
    provider: ModelProvider,
    model: str,
    **kwargs: Any,
) -> GenerationResult:
    """Synchronous wrapper for the async generation pipeline."""
    return asyncio.run(generate_population_from_text(text, provider, model, **kwargs))


def _external_id(name: str, index: int) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")[:24]
    return f"{slug or 'actor'}-{index:02d}"


def result_to_json(result: GenerationResult) -> dict[str, Any]:
    """Serialize a GenerationResult for CLI/JSON export."""
    payload = result.manifest.to_mapping()
    payload["diagnostics"] = dict(result.diagnostics)
    return payload
