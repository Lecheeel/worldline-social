"""Small deterministic social world used as the first Worldline Social slice."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from worldline_engine.protocols import (
    ActionKind,
    ActionResult,
    ActionSpec,
    ActionStatus,
    BoundAction,
    CommitDecision,
)


class SocialWorld:
    """Posts, comments, feed reads, likes, and deterministic commit semantics."""

    _actions = (
        ActionSpec("view_feed", ActionKind.READ, "Read the current public feed."),
        ActionSpec("create_post", ActionKind.WRITE, "Publish a post.", {"required": ("content",)}),
        ActionSpec("create_comment", ActionKind.WRITE, "Comment on a post.", {"required": ("post_id", "content")}),
        ActionSpec("like_post", ActionKind.WRITE, "Like a post.", {"required": ("post_id",)}),
        ActionSpec("do_nothing", ActionKind.WRITE, "Record no social action."),
    )

    def __init__(self, people: Sequence[str]) -> None:
        self.people = tuple(sorted(set(people)))
        self.state: dict[str, Any] = {"posts": {}, "comments": {}, "likes": []}

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def restore(self, state: Any) -> None:
        if not isinstance(state, dict) or not all(key in state for key in ("posts", "comments", "likes")):
            raise ValueError("invalid SocialWorld state")
        self.state = deepcopy(state)

    def available_actions(self, entity_id: str, snapshot: Any) -> Sequence[ActionSpec]:
        if entity_id not in self.people:
            return ()
        return self._actions

    def observe(self, entity_id: str, snapshot: Any, local_overlay: Sequence[BoundAction]) -> dict[str, Any]:
        posts = list(snapshot.get("posts", {}).values())
        posts.sort(key=lambda post: post["post_id"])
        return {"person_id": entity_id, "feed": deepcopy(posts)}

    def execute_read(self, action: BoundAction, snapshot: Any, local_overlay: Sequence[BoundAction]) -> ActionResult:
        if action.intent.action_type != "view_feed":
            return ActionResult(action.action_id, ActionStatus.REJECTED, error_code="read_not_supported")
        return ActionResult(action.action_id, ActionStatus.ACCEPTED, data={"feed": self.observe(action.entity_id, snapshot, local_overlay)["feed"]})

    def validate_write(self, action: BoundAction, snapshot: Any, local_overlay: Sequence[BoundAction]) -> ActionResult:
        name = action.intent.action_type
        params = action.intent.parameters
        if name == "do_nothing":
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=0)
        if name == "create_post" and isinstance(params.get("content"), str) and params["content"].strip():
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        if name == "create_comment":
            post_id = params.get("post_id")
            if post_id not in snapshot.get("posts", {}):
                return ActionResult(action.action_id, ActionStatus.REJECTED, error_code="post_not_found")
            if isinstance(params.get("content"), str) and params["content"].strip():
                return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        if name == "like_post":
            if params.get("post_id") not in snapshot.get("posts", {}):
                return ActionResult(action.action_id, ActionStatus.REJECTED, error_code="post_not_found")
            return ActionResult(action.action_id, ActionStatus.ACCEPTED, cost=1)
        return ActionResult(action.action_id, ActionStatus.REJECTED, error_code="invalid_social_action")

    def resolve_and_apply(self, snapshot: Any, actions: Sequence[BoundAction]) -> Sequence[CommitDecision]:
        decisions: list[CommitDecision] = []
        for action in actions:
            name = action.intent.action_type
            params = action.intent.parameters
            if name == "create_post":
                post_id = f"post-{action.action_id}"
                self.state["posts"][post_id] = {"post_id": post_id, "author": action.entity_id, "content": params["content"], "like_count": 0}
            elif name == "create_comment":
                comment_id = f"comment-{action.action_id}"
                self.state["comments"][comment_id] = {"comment_id": comment_id, "post_id": params["post_id"], "author": action.entity_id, "content": params["content"]}
            elif name == "like_post":
                key = (action.entity_id, params["post_id"])
                if key not in [tuple(item) for item in self.state["likes"]]:
                    self.state["likes"].append(list(key))
                    self.state["posts"][params["post_id"]]["like_count"] += 1
            decisions.append(CommitDecision(action, ActionResult(action.action_id, ActionStatus.ACCEPTED, data={"action": name})))
        return decisions
