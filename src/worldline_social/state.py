"""Versioned JSON-compatible state owned by Worldline Social."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


CURRENT_SOCIAL_STATE_VERSION = 2


@dataclass
class SocialState:
    schema_version: int = CURRENT_SOCIAL_STATE_VERSION
    people: dict[str, dict[str, Any]] = field(default_factory=dict)
    posts: dict[str, dict[str, Any]] = field(default_factory=dict)
    comments: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    post_likes: list[list[str]] = field(default_factory=list)
    comment_likes: list[list[str]] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return deepcopy(
            {
                "schema_version": self.schema_version,
                "people": self.people,
                "posts": self.posts,
                "comments": self.comments,
                "relationships": self.relationships,
                "post_likes": self.post_likes,
                "comment_likes": self.comment_likes,
            }
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SocialState":
        version = value.get("schema_version")
        if version == 1:
            value = {
                **value,
                "schema_version": CURRENT_SOCIAL_STATE_VERSION,
                "post_likes": value.get("likes", ()),
                "comment_likes": (),
            }
            version = CURRENT_SOCIAL_STATE_VERSION
        if version != CURRENT_SOCIAL_STATE_VERSION:
            raise ValueError("unsupported SocialState schema_version")
        required = (
            "people",
            "posts",
            "comments",
            "relationships",
            "post_likes",
            "comment_likes",
        )
        if any(key not in value for key in required):
            raise ValueError("SocialState is missing required fields")
        return cls(
            people=deepcopy(dict(value["people"])),
            posts=deepcopy(dict(value["posts"])),
            comments=deepcopy(dict(value["comments"])),
            relationships=deepcopy(list(value["relationships"])),
            post_likes=deepcopy(list(value["post_likes"])),
            comment_likes=deepcopy(list(value["comment_likes"])),
        )
