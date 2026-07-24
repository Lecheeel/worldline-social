"""Versioned JSON-compatible state owned by Worldline Social."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SocialState:
    schema_version: int = 1
    people: dict[str, dict[str, Any]] = field(default_factory=dict)
    posts: dict[str, dict[str, Any]] = field(default_factory=dict)
    comments: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    likes: list[list[str]] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return deepcopy(
            {
                "schema_version": self.schema_version,
                "people": self.people,
                "posts": self.posts,
                "comments": self.comments,
                "relationships": self.relationships,
                "likes": self.likes,
            }
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SocialState":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported SocialState schema_version")
        required = ("people", "posts", "comments", "relationships", "likes")
        if any(key not in value for key in required):
            raise ValueError("SocialState is missing required fields")
        return cls(
            schema_version=1,
            people=deepcopy(dict(value["people"])),
            posts=deepcopy(dict(value["posts"])),
            comments=deepcopy(dict(value["comments"])),
            relationships=deepcopy(list(value["relationships"])),
            likes=deepcopy(list(value["likes"])),
        )
