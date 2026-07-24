from __future__ import annotations

import unittest

from worldline_social.state import SocialState


class SocialStateTests(unittest.TestCase):
    def test_version_one_checkpoint_migrates_to_current_schema(self) -> None:
        state = SocialState.from_mapping(
            {
                "schema_version": 1,
                "people": {"alice": {"handle": "alice"}},
                "posts": {},
                "comments": {},
                "relationships": [],
                "likes": [["alice", "post-1"]],
            }
        )

        self.assertEqual(2, state.schema_version)
        self.assertEqual([["alice", "post-1"]], state.post_likes)
        self.assertEqual([], state.comment_likes)


if __name__ == "__main__":
    unittest.main()
