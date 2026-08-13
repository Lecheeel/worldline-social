"""Command-line interface for Worldline Social experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .experiment import ExperimentConfig
from .generation import generate_population_from_text_sync, result_to_json
from .population import PopulationManifest
from .providers import DeepSeekProvider
from .runner import run_experiment_sync
from .task.cli import add_task_parser, handle_task_command


def main() -> None:
    parser = argparse.ArgumentParser(prog="worldline-social")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a versioned experiment config")
    run.add_argument("config")
    run.add_argument("--resume", action="store_true")
    validate = commands.add_parser("validate-population", help="Validate a population manifest")
    validate.add_argument("manifest")
    generate = commands.add_parser(
        "generate-manifest",
        help="Generate a population manifest from an event text or file",
    )
    generate.add_argument("input", help="Path to a .txt/.md file, or '-' for stdin")
    generate.add_argument("output", help="Output manifest JSON path")
    generate.add_argument("--model", default="deepseek-chat", help="LLM model name")
    generate.add_argument("--source", default="event-generated", help="Manifest source label")
    generate.add_argument("--max-participants", type=int, default=30)
    generate.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled",
                          help="LLM thinking mode (default: disabled)")
    add_task_parser(commands)
    args = parser.parse_args()

    if args.command == "task":
        handle_task_command(args)
        return
    if args.command == "validate-population":
        manifest = PopulationManifest.from_json(args.manifest)
        imported = manifest.import_population()
        print(json.dumps({"valid": True, "population_size": len(imported.people)}))
        return

    if args.command == "generate-manifest":
        text = _read_input(args.input)
        provider = DeepSeekProvider.from_environment()
        result = generate_population_from_text_sync(
            text,
            provider,
            args.model,
            source=args.source,
            max_participants=args.max_participants,
            thinking=args.thinking,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result_to_json(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "written": str(output),
                    "population_size": len(result.manifest.people),
                    "relationships": len(result.manifest.relationships),
                    "initial_content": len(result.manifest.initial_content),
                    "usage": result.diagnostics.get("usage", {}),
                },
                sort_keys=True,
            )
        )
        return

    config = ExperimentConfig.from_json(args.config)
    result = run_experiment_sync(config, resume=args.resume)
    summary = asdict(result)
    summary.pop("world_state")
    print(json.dumps(summary, sort_keys=True))


def _read_input(location: str) -> str:
    if location == "-":
        return sys.stdin.read()
    return Path(location).read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
