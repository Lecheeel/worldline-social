"""Versioned population manifests and deterministic person-id assignment."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


HANDLE_PATTERN = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class PersonProfile:
    external_id: str
    handle: str
    display_name: str = ""
    bio: str = ""
    private_traits: Mapping[str, Any] = field(default_factory=dict)
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    controller_ref: str = "default"
    model_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationshipSpec:
    source_external_id: str
    target_external_id: str
    relationship_type: str = "follow"
    strength: float = 1.0


@dataclass(frozen=True)
class PopulationManifest:
    manifest_version: str
    source: str
    people: Sequence[PersonProfile]
    relationships: Sequence[RelationshipSpec] = field(default_factory=tuple)
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)
    groups: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    initial_content: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PopulationManifest":
        people = tuple(PersonProfile(**person) for person in value.get("people", ()))
        relationships = tuple(
            RelationshipSpec(**relationship)
            for relationship in value.get("relationships", ())
        )
        manifest = cls(
            manifest_version=str(value.get("manifest_version", "")),
            source=str(value.get("source", "")),
            people=people,
            relationships=relationships,
            generation_metadata=dict(value.get("generation_metadata", {})),
            groups=tuple(value.get("groups", ())),
            initial_content=tuple(value.get("initial_content", ())),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_json(cls, path: str | Path) -> "PopulationManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("population manifest must be a JSON object")
        return cls.from_mapping(value)

    def validate(self) -> None:
        if self.manifest_version != "1":
            raise ValueError("unsupported manifest_version")
        if not self.source.strip():
            raise ValueError("manifest source must not be empty")
        if not self.people:
            raise ValueError("manifest must contain at least one person")
        external_ids = [person.external_id for person in self.people]
        handles = [person.handle for person in self.people]
        if any(not external_id.strip() for external_id in external_ids):
            raise ValueError("external_id must not be empty")
        if len(set(external_ids)) != len(external_ids):
            raise ValueError("external_id values must be unique")
        if len(set(handles)) != len(handles):
            raise ValueError("handle values must be unique")
        if any(HANDLE_PATTERN.fullmatch(handle) is None for handle in handles):
            raise ValueError("handles must match ^[a-z0-9_]+$")
        known = set(external_ids)
        for relationship in self.relationships:
            if relationship.source_external_id not in known or relationship.target_external_id not in known:
                raise ValueError("relationship references an unknown external_id")
            if not 0.0 <= relationship.strength <= 1.0:
                raise ValueError("relationship strength must be in [0, 1]")

    def import_population(self) -> "ImportedPopulation":
        self.validate()
        mapping = {
            person.external_id: _person_id(self.source, person.external_id)
            for person in sorted(self.people, key=lambda item: item.external_id)
        }
        people = {
            mapping[person.external_id]: person
            for person in sorted(self.people, key=lambda item: item.external_id)
        }
        relationships = tuple(
            ImportedRelationship(
                mapping[item.source_external_id],
                mapping[item.target_external_id],
                item.relationship_type,
                item.strength,
            )
            for item in sorted(
                self.relationships,
                key=lambda value: (
                    value.source_external_id,
                    value.target_external_id,
                    value.relationship_type,
                ),
            )
        )
        return ImportedPopulation(mapping, people, relationships)


@dataclass(frozen=True)
class ImportedRelationship:
    source_person_id: str
    target_person_id: str
    relationship_type: str
    strength: float


@dataclass(frozen=True)
class ImportedPopulation:
    external_id_mapping: Mapping[str, str]
    people: Mapping[str, PersonProfile]
    relationships: Sequence[ImportedRelationship]


def _person_id(source: str, external_id: str) -> str:
    digest = hashlib.sha256(f"{source}\x00{external_id}".encode("utf-8")).hexdigest()
    return f"person-{digest[:24]}"
