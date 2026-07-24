"""Deterministic social world implemented through Worldline Engine protocols."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from worldline_engine.protocols import (
    ActionKind,
    ActionResult,
    ActionSpec,
    ActionStatus,
    BoundAction,
    CommitDecision,
)

from .population import PopulationManifest
from .state import SocialState


class SocialWorld:
    """Posts, comments, feed reads, likes, and deterministic commit semantics."""

    _actions = (
        ActionSpec("view_feed", ActionKind.READ, "Read the current public feed."),
        ActionSpec(
            "create_post",
            ActionKind.WRITE,
            "Publish a post.",
            {"required": ("content",)},
        ),
        ActionSpec(
            "create_comment",
            ActionKind.WRITE,
            "Comment on a post.",
            {"required": ("post_id", "content")},
        ),
        ActionSpec(
            "like_post",
            ActionKind.WRITE,
            "Like a post.",
            {"required": ("post_id",)},
        ),
        ActionSpec("do_nothing", ActionKind.WRITE, "Record no social action.", cost=0),
    )

    def __init__(self, people: Sequence[str]) -> None:
        person_ids = tuple(sorted(set(people)))
        if len(person_ids) != len(tuple(people)):
            raise ValueError("person ids must be unique")
        self.people = person_ids
        self._state = SocialState(
            people={
                person_id: {"person_id": person_id, "handle": person_id}
                for person_id in person_ids
            }
        )

    @classmethod
    def from_manifest(cls, manifest: PopulationManifest) -> "SocialWorld":
        imported = manifest.import_population()
        world = cls(tuple(imported.people))
        world._state.people = {
            person_id: {
                "person_id": person_id,
                "external_id": profile.external_id,
                "handle": profile.handle,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "private_traits": deepcopy(dict(profile.private_traits)),
                "dynamic_state": deepcopy(dict(profile.initial_state)),
                "controller_ref": profile.controller_ref,
                "model_policy": deepcopy(dict(profile.model_policy)),
            }
            for person_id, profile in imported.people.items()
        }
        world._state.relationships = [
            {
                "source_person_id": item.source_person_id,
                "target_person_id": item.target_person_id,
                "relationship_type": item.relationship_type,
                "strength": item.strength,
            }
            for item in imported.relationships
        ]
        return world

    @property
    def state(self) -> dict[str, Any]:
        return self._state.to_mapping()

    def snapshot(self) -> dict[str, Any]:
        return self.state

    def restore(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("invalid SocialWorld state")
        restored = SocialState.from_mapping(state)
        if set(restored.people) != set(self.people):
            raise ValueError("checkpoint population does not match this SocialWorld")
        self._state = restored

    def available_actions(self, entity_id: str, snapshot: Any) -> Sequence[ActionSpec]:
        del snapshot
        return self._actions if entity_id in self.people else ()

    def observe(
        self,
        entity_id: str,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> dict[str, Any]:
        del local_overlay
        posts = list(snapshot.get("posts", {}).values())
        posts.sort(key=lambda post: post["post_id"])
        person = snapshot.get("people", {}).get(entity_id, {})
        return {
            "self": {
                "handle": person.get("handle"),
                "display_name": person.get("display_name", ""),
            },
            "feed": deepcopy(posts),
        }

    def execute_read(
        self,
        action: BoundAction,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> ActionResult:
        if action.intent.action_type != "view_feed":
            return ActionResult(
                action.action_id,
                ActionStatus.REJECTED,
                error_code="read_not_supported",
            )
        observation = self.observe(action.entity_id, snapshot, local_overlay)
        return ActionResult(
            action.action_id,
            ActionStatus.ACCEPTED,
            data={"feed": observation["feed"]},
        )

    def validate_write(
        self,
        action: BoundAction,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> ActionResult:
        del local_overlay
        name = action.intent.action_type
        params = action.intent.parameters
        if name == "do_nothing":
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=0)
        if name == "create_post" and _valid_content(params.get("content")):
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        if name == "create_comment":
            if params.get("post_id") not in snapshot.get("posts", {}):
                return ActionResult(
                    action.action_id,
                    ActionStatus.REJECTED,
                    error_code="post_not_found",
                )
            if _valid_content(params.get("content")):
                return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        if name == "like_post":
            if params.get("post_id") not in snapshot.get("posts", {}):
                return ActionResult(
                    action.action_id,
                    ActionStatus.REJECTED,
                    error_code="post_not_found",
                )
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        return ActionResult(
            action.action_id,
            ActionStatus.REJECTED,
            error_code="invalid_social_action",
        )

    def resolve_and_apply(
        self,
        snapshot: Any,
        actions: Sequence[BoundAction],
    ) -> Sequence[CommitDecision]:
        next_state = SocialState.from_mapping(snapshot)
        decisions: list[CommitDecision] = []
        for action in actions:
            name = action.intent.action_type
            params = action.intent.parameters
            if name == "create_post":
                post_id = f"post-{action.action_id}"
                next_state.posts[post_id] = {
                    "post_id": post_id,
                    "author_person_id": action.entity_id,
                    "content": params["content"],
                    "like_count": 0,
                }
            elif name == "create_comment":
                comment_id = f"comment-{action.action_id}"
                next_state.comments[comment_id] = {
                    "comment_id": comment_id,
                    "post_id": params["post_id"],
                    "author_person_id": action.entity_id,
                    "content": params["content"],
                }
            elif name == "like_post":
                key = [action.entity_id, params["post_id"]]
                if key not in next_state.likes:
                    next_state.likes.append(key)
                    next_state.posts[params["post_id"]]["like_count"] += 1
            decisions.append(
                CommitDecision(
                    action,
                    ActionResult(
                        action.action_id,
                        ActionStatus.ACCEPTED,
                        data={"action": name},
                    ),
                )
            )
        self._state = next_state
        return tuple(decisions)


def _valid_content(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 10_000
