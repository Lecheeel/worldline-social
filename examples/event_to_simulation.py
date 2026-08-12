"""End-to-end: event text -> generated population -> LLM-driven simulation.

This is the full automated loop Worldline Social is built for:

    1. Read a real-world public-opinion event (text or .txt/.md file).
    2. Extract the actors who can speak on social media, generate personas,
       traits, stances and relationships (LLM).
    3. Run a deterministic simulation where agents act through the LLM,
       with affective dynamics (moods react to likes, dislikes, comments).
    4. Print a summary of the final world state.

Environment: set DEEPSEEK_API_KEY (or another OpenAI-compatible key via
--api-key-env) before running.

Example:
    python examples/event_to_simulation.py event.txt --ticks 3 --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldline_social.experiment import ExperimentConfig  # noqa: E402
from worldline_social.generation import generate_population_from_text_sync  # noqa: E402
from worldline_social.providers import DeepSeekProvider  # noqa: E402
from worldline_social.runner import run_experiment_sync  # noqa: E402


def read_input(location: str) -> str:
    if location == "-":
        return sys.stdin.read()
    return Path(location).read_text(encoding="utf-8")


def write_experiment_config(tmp: Path, manifest_path: Path, args) -> Path:
    config = {
        "config_version": "1",
        "simulation_id": "event-simulation",
        "population_manifest": manifest_path.name,
        "output_database": "../runs/event-simulation.sqlite",
        "seed": args.seed,
        "max_ticks": args.ticks,
        "activation_probability": 1.0,
        "max_actions_per_turn": args.max_actions,
        "max_concurrent_turns": 1,
        "checkpoint_every_ticks": 1,
        "distribution_policy": "all",
        "dynamics_policy": "affective",
        "feed_limit": 100,
        "llm": {
            "provider": "deepseek",
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "api_key_env": args.api_key_env,
        },
    }
    path = tmp / "experiment.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Event text file (.txt/.md) or '-' for stdin")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--seed", type=int, default=0, help="Worldline seed (one worldline)")
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=4, help="Actions per turn (read-then-write loop)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-participants", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate the manifest only; do not run the simulation")
    args = parser.parse_args()

    text = read_input(args.input)
    if not text.strip():
        parser.error("input is empty")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"{args.api_key_env} is not set; required for the LLM pipeline")

    provider = DeepSeekProvider(api_key)

    print(f"[1/3] Extracting actors and generating personas from {len(text)} chars ...")
    result = generate_population_from_text_sync(
        text,
        provider,
        args.model,
        source="event-to-simulation",
        max_participants=args.max_participants,
    )
    manifest = result.manifest
    print(f"  -> {len(manifest.people)} actors, {len(manifest.relationships)} relationships, "
          f"{len(manifest.initial_content)} initial post(s)")
    for person in manifest.people:
        stance = person.model_policy.get("stance", "?")
        print(f"     - {person.display_name} (@{person.handle}) stance={stance}")

    if args.dry_run:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest_path = tmp / "population.json"
        manifest_path.write_text(
            json.dumps(manifest.to_mapping(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        config_path = write_experiment_config(tmp, manifest_path, args)
        config = ExperimentConfig.from_json(config_path)

        print(f"[2/3] Running deterministic simulation: {args.ticks} ticks, seed={args.seed} ...")
        result_run = run_experiment_sync(config, resume=False)
        print(f"  -> completed {result_run.completed_ticks} ticks; database: {result_run.database_path}")

        print("[3/3] Final emotional states:")
        for person_id, person in sorted(result_run.world_state["people"].items()):
            state = person.get("dynamic_state", {})
            print(
                f"     - {person.get('handle')}: mood={state.get('mood', 0):+.2f} "
                f"anger={state.get('anger', 0):.2f} stress={state.get('stress', 0):.2f} "
                f"threat={state.get('threat', 0):.2f}"
            )


if __name__ == "__main__":
    main()
