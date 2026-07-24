from __future__ import annotations

import unittest

from worldline_social.population import PopulationManifest
from worldline_social.world import SocialWorld


def manifest(people):
    return PopulationManifest.from_mapping(
        {
            "manifest_version": "1",
            "source": "test-dataset",
            "generation_metadata": {"seed": 42},
            "people": people,
            "relationships": [
                {
                    "source_external_id": "source-a",
                    "target_external_id": "source-b",
                    "relationship_type": "follow",
                    "strength": 0.75,
                }
            ],
        }
    )


class PopulationManifestTests(unittest.TestCase):
    def test_import_is_deterministic_across_input_order(self) -> None:
        alice = {"external_id": "source-a", "handle": "alice", "display_name": "Alice"}
        bob = {"external_id": "source-b", "handle": "bob", "display_name": "Bob"}

        first = manifest([alice, bob]).import_population()
        second = manifest([bob, alice]).import_population()

        self.assertEqual(first.external_id_mapping, second.external_id_mapping)
        self.assertEqual(tuple(first.people), tuple(second.people))
        self.assertEqual(first.relationships, second.relationships)

    def test_manifest_rejects_invalid_or_duplicate_handles(self) -> None:
        with self.assertRaises(ValueError):
            manifest(
                [
                    {"external_id": "source-a", "handle": "Alice"},
                    {"external_id": "source-b", "handle": "Alice"},
                ]
            )

    def test_social_world_owns_private_and_public_identity_state(self) -> None:
        population = manifest(
            [
                {
                    "external_id": "source-a",
                    "handle": "alice",
                    "private_traits": {"openness": 0.8},
                    "initial_state": {"energy": 0.6},
                },
                {"external_id": "source-b", "handle": "bob"},
            ]
        )
        imported = population.import_population()
        world = SocialWorld.from_manifest(population)
        alice_id = imported.external_id_mapping["source-a"]

        self.assertEqual("alice", world.state["people"][alice_id]["handle"])
        self.assertEqual(0.8, world.state["people"][alice_id]["private_traits"]["openness"])
        self.assertEqual(1, len(world.state["relationships"]))


if __name__ == "__main__":
    unittest.main()
