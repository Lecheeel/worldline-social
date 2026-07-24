"""Replaceable, deterministic feed distribution policies."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class DistributionPolicy(Protocol):
    def select(
        self,
        person_id: str,
        state: Mapping[str, Any],
        limit: int,
        random_seed: int,
    ) -> Sequence[Mapping[str, Any]]: ...


class AllPostsDistribution:
    """Return every post in stable creation order, bounded by `limit`."""

    def select(self, person_id: str, state: Mapping[str, Any], limit: int, random_seed: int):
        del person_id, random_seed
        posts = sorted(
            state.get("posts", {}).values(),
            key=lambda post: (post.get("created_tick", 0), post["post_id"]),
        )
        return tuple(posts[:limit])


class RecentPostsDistribution:
    """Return newest posts first without engagement-based amplification."""

    def select(self, person_id: str, state: Mapping[str, Any], limit: int, random_seed: int):
        del person_id, random_seed
        posts = sorted(
            state.get("posts", {}).values(),
            key=lambda post: (post.get("created_tick", 0), post["post_id"]),
            reverse=True,
        )
        return tuple(posts[:limit])
