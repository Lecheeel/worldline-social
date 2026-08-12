"""Map committed social actions to affective feedback for their actors.

The mapping from ``(action type, parameters)`` to ``{kind, count}`` feedback
is domain knowledge owned by Worldline Social. Keeping it in its own module
lets the world class stay lean and lets dynamics policies be tested against
feedback streams directly.
"""

from __future__ import annotations

from typing import Any, Mapping

from worldline_engine.protocols import BoundAction


def collect_action_feedback(
    state: Any,
    action: BoundAction,
    feedback_by_person: dict[str, list[dict[str, Any]]],
) -> None:
    """Accumulate affective feedback caused by one committed action.

    ``state`` is the (already applied) next state, used to resolve target
    authors. Read actions (view_feed and friends) intentionally produce no
    feedback yet: they run outside the commit path and would require a
    separate staging mechanism.

    Feedback kinds understood by ``AffectiveDynamics``:
    ``post_created``, ``received_like``, ``received_unlike``,
    ``received_comment``. Future kinds are free-form; unknown kinds are
    ignored by dynamics policies that do not understand them.
    """
    name = action.intent.action_type
    params = action.intent.parameters
    if name == "create_post":
        _append(feedback_by_person, action.entity_id, {"kind": "post_created"})
    elif name in {"create_comment", "reply_comment"}:
        _append(feedback_by_person, action.entity_id, {"kind": "post_created"})
        target = _comment_target_author(state, name, params)
        if target is not None:
            _append(feedback_by_person, target, {"kind": "received_comment"})
    elif name == "like_post":
        target = _post_author(state, params.get("post_id"))
        if target is not None:
            _append(feedback_by_person, target, {"kind": "received_like"})
    elif name == "unlike_post":
        target = _post_author(state, params.get("post_id"))
        if target is not None:
            _append(feedback_by_person, target, {"kind": "received_unlike"})


def _append(
    feedback_by_person: dict[str, list[dict[str, Any]]],
    person_id: str,
    feedback: dict[str, Any],
) -> None:
    feedback_by_person.setdefault(person_id, []).append(feedback)


def _post_author(state: Any, post_id: Any) -> str | None:
    post = state.posts.get(post_id)
    return post["author_person_id"] if post is not None else None


def _comment_target_author(
    state: Any, name: str, params: Mapping[str, Any]
) -> str | None:
    if name == "create_comment":
        post = state.posts.get(params.get("post_id"))
        return post["author_person_id"] if post is not None else None
    parent = state.comments.get(params.get("comment_id"))
    return parent["author_person_id"] if parent is not None else None
