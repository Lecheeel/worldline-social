"""Deterministic social world implemented through Worldline Engine protocols."""

from __future__ import annotations

import hashlib
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

from .distribution import AllPostsDistribution, DistributionPolicy
from .dynamics import DynamicsPolicy, RecoveryDynamics
from .population import PopulationManifest
from .state import SocialState


class SocialWorld:
    """Public conversation with replaceable distribution and stable state."""

    _actions = (
        ActionSpec("view_feed", ActionKind.READ, "Read the current public feed."),
        ActionSpec(
            "view_thread",
            ActionKind.READ,
            "Read a post and its comments.",
            {"required": ("post_id",)},
        ),
        ActionSpec(
            "search_square",
            ActionKind.READ,
            "Search public posts and comments.",
            {"required": ("query",)},
            cost=2,
        ),
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
            "reply_comment",
            ActionKind.WRITE,
            "Reply to a comment.",
            {"required": ("comment_id", "content")},
        ),
        ActionSpec(
            "like_post",
            ActionKind.WRITE,
            "Like a post.",
            {"required": ("post_id",)},
        ),
        ActionSpec(
            "unlike_post",
            ActionKind.WRITE,
            "Remove a post like.",
            {"required": ("post_id",)},
        ),
        ActionSpec(
            "like_comment",
            ActionKind.WRITE,
            "Like a comment.",
            {"required": ("comment_id",)},
        ),
        ActionSpec(
            "unlike_comment",
            ActionKind.WRITE,
            "Remove a comment like.",
            {"required": ("comment_id",)},
        ),
        ActionSpec("do_nothing", ActionKind.WRITE, "Record no social action.", cost=0),
    )

    def __init__(
        self,
        people: Sequence[str],
        distribution_policy: DistributionPolicy | None = None,
        dynamics_policy: DynamicsPolicy | None = None,
        feed_limit: int = 100,
    ) -> None:
        person_ids = tuple(sorted(set(people)))
        if len(person_ids) != len(tuple(people)):
            raise ValueError("person ids must be unique")
        if feed_limit < 1:
            raise ValueError("feed_limit must be positive")
        self.people = person_ids
        self.distribution_policy = distribution_policy or AllPostsDistribution()
        self.dynamics_policy = dynamics_policy or RecoveryDynamics()
        self.feed_limit = feed_limit
        self._state = SocialState(
            people={
                person_id: {"person_id": person_id, "handle": person_id}
                for person_id in person_ids
            }
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: PopulationManifest,
        distribution_policy: DistributionPolicy | None = None,
        dynamics_policy: DynamicsPolicy | None = None,
        feed_limit: int = 100,
    ) -> "SocialWorld":
        imported = manifest.import_population()
        world = cls(
            tuple(imported.people),
            distribution_policy,
            dynamics_policy,
            feed_limit,
        )
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
        visible = self._visible_state(snapshot, local_overlay)
        person = visible.people.get(entity_id, {})
        feed = self.distribution_policy.select(
            entity_id,
            visible.to_mapping(),
            self.feed_limit,
            _stable_seed(entity_id),
        )
        return {
            "self": {
                "handle": person.get("handle"),
                "display_name": person.get("display_name", ""),
            },
            "feed": deepcopy(list(feed)),
        }

    def execute_read(
        self,
        action: BoundAction,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> ActionResult:
        visible = self._visible_state(snapshot, local_overlay)
        name = action.intent.action_type
        params = action.intent.parameters
        if name == "view_feed":
            observation = self.observe(action.entity_id, snapshot, local_overlay)
            return ActionResult(
                action.action_id,
                ActionStatus.ACCEPTED,
                data={"feed": observation["feed"]},
            )
        if name == "view_thread":
            post_id = params.get("post_id")
            post = visible.posts.get(post_id)
            if post is None:
                return _rejected(action, "post_not_found")
            comments = sorted(
                (
                    comment
                    for comment in visible.comments.values()
                    if comment["post_id"] == post_id
                ),
                key=lambda comment: comment["comment_id"],
            )
            return ActionResult(
                action.action_id,
                ActionStatus.ACCEPTED,
                data={"post": deepcopy(post), "comments": deepcopy(comments)},
            )
        if name == "search_square":
            query = params.get("query")
            if not _valid_query(query):
                return _rejected(action, "invalid_query")
            needle = query.casefold()
            matches = [
                {"result_type": "post", **deepcopy(post)}
                for post in visible.posts.values()
                if needle in post["content"].casefold()
            ]
            matches.extend(
                {"result_type": "comment", **deepcopy(comment)}
                for comment in visible.comments.values()
                if needle in comment["content"].casefold()
            )
            matches.sort(
                key=lambda item: (
                    item["result_type"],
                    item.get("post_id", ""),
                    item.get("comment_id", ""),
                )
            )
            return ActionResult(
                action.action_id,
                ActionStatus.ACCEPTED,
                data={"query": query, "results": matches[:100]},
            )
        return _rejected(action, "read_not_supported")

    def validate_write(
        self,
        action: BoundAction,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> ActionResult:
        visible = self._visible_state(snapshot, local_overlay)
        name = action.intent.action_type
        params = action.intent.parameters
        if name == "do_nothing":
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=0)
        if name == "create_post" and _valid_content(params.get("content")):
            local_ref = f"post-{action.action_id}"
            return ActionResult(
                action.action_id,
                ActionStatus.ACCEPTED,
                data={"post_id": local_ref},
                local_ref=local_ref,
            )
        if name == "create_comment":
            if params.get("post_id") not in visible.posts:
                return _rejected(action, "post_not_found")
            if _valid_content(params.get("content")):
                local_ref = f"comment-{action.action_id}"
                return ActionResult(
                    action.action_id,
                    ActionStatus.ACCEPTED,
                    data={"comment_id": local_ref},
                    local_ref=local_ref,
                )
        if name == "reply_comment":
            parent = visible.comments.get(params.get("comment_id"))
            if parent is None:
                return _rejected(action, "comment_not_found")
            if _valid_content(params.get("content")):
                local_ref = f"comment-{action.action_id}"
                return ActionResult(
                    action.action_id,
                    ActionStatus.ACCEPTED,
                    data={"comment_id": local_ref, "post_id": parent["post_id"]},
                    local_ref=local_ref,
                )
        if name in {"like_post", "unlike_post"}:
            if params.get("post_id") not in visible.posts:
                return _rejected(action, "post_not_found")
            return ActionResult(action.action_id, ActionStatus.ACCEPTED)
        if name in {"like_comment", "unlike_comment"}:
            if params.get("comment_id") not in visible.comments:
                return _rejected(action, "comment_not_found")
            return ActionResult(action.action_id, ActionStatus.ACCEPTED)
        return _rejected(action, "invalid_social_action")

    def resolve_and_apply(
        self,
        snapshot: Any,
        actions: Sequence[BoundAction],
    ) -> Sequence[CommitDecision]:
        next_state = SocialState.from_mapping(snapshot)
        decisions: list[CommitDecision] = []
        for action in actions:
            data = self._apply_action(next_state, action)
            decisions.append(
                CommitDecision(
                    action,
                    ActionResult(
                        action.action_id,
                        ActionStatus.ACCEPTED,
                        data=data,
                    ),
                )
            )
        self._state = next_state
        return tuple(decisions)

    def advance_tick(self, tick_id: int) -> None:
        next_state = SocialState.from_mapping(self.state)
        for person_id in sorted(next_state.people):
            person = next_state.people[person_id]
            person["dynamic_state"] = dict(
                self.dynamics_policy.advance(
                    person_id,
                    person.get("private_traits", {}),
                    person.get("dynamic_state", {}),
                    tick_id,
                )
            )
        self._state = next_state

    def _visible_state(
        self,
        snapshot: Any,
        local_overlay: Sequence[BoundAction],
    ) -> SocialState:
        visible = SocialState.from_mapping(snapshot)
        for action in local_overlay:
            self._apply_action(visible, action)
        return visible

    @staticmethod
    def _apply_action(state: SocialState, action: BoundAction) -> dict[str, Any]:
        name = action.intent.action_type
        params = action.intent.parameters
        if name == "create_post":
            post_id = f"post-{action.action_id}"
            state.posts[post_id] = {
                "post_id": post_id,
                "author_person_id": action.entity_id,
                "content": params["content"],
                "created_tick": action.tick_id,
                "like_count": 0,
            }
            return {"post_id": post_id}
        if name in {"create_comment", "reply_comment"}:
            comment_id = f"comment-{action.action_id}"
            parent_comment_id = params.get("comment_id") if name == "reply_comment" else None
            post_id = (
                state.comments[parent_comment_id]["post_id"]
                if parent_comment_id is not None
                else params["post_id"]
            )
            state.comments[comment_id] = {
                "comment_id": comment_id,
                "post_id": post_id,
                "parent_comment_id": parent_comment_id,
                "author_person_id": action.entity_id,
                "content": params["content"],
                "created_tick": action.tick_id,
                "like_count": 0,
            }
            return {"comment_id": comment_id, "post_id": post_id}
        if name in {"like_post", "unlike_post"}:
            key = [action.entity_id, params["post_id"]]
            is_like = name == "like_post"
            changed = _set_reaction(state.post_likes, key, is_like)
            if changed:
                state.posts[params["post_id"]]["like_count"] += 1 if is_like else -1
            return {"post_id": params["post_id"], "liked": is_like}
        if name in {"like_comment", "unlike_comment"}:
            key = [action.entity_id, params["comment_id"]]
            is_like = name == "like_comment"
            changed = _set_reaction(state.comment_likes, key, is_like)
            if changed:
                state.comments[params["comment_id"]]["like_count"] += 1 if is_like else -1
            return {"comment_id": params["comment_id"], "liked": is_like}
        return {"action": name}


def _set_reaction(collection: list[list[str]], key: list[str], enabled: bool) -> bool:
    if enabled and key not in collection:
        collection.append(key)
        return True
    if not enabled and key in collection:
        collection.remove(key)
        return True
    return False


def _rejected(action: BoundAction, error_code: str) -> ActionResult:
    return ActionResult(action.action_id, ActionStatus.REJECTED, error_code=error_code)


def _valid_content(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 10_000


def _valid_query(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 500


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
