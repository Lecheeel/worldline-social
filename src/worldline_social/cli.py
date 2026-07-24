"""Command-line interface for Worldline Social experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .experiment import ExperimentConfig
from .population import PopulationManifest
from .runner import run_experiment_sync


def main() -> None:
    parser = argparse.ArgumentParser(prog="worldline-social")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a versioned experiment config")
    run.add_argument("config")
    run.add_argument("--resume", action="store_true")
    validate = commands.add_parser("validate-population", help="Validate a population manifest")
    validate.add_argument("manifest")
    args = parser.parse_args()

    if args.command == "validate-population":
        manifest = PopulationManifest.from_json(args.manifest)
        imported = manifest.import_population()
        print(json.dumps({"valid": True, "population_size": len(imported.people)}))
        return

    config = ExperimentConfig.from_json(args.config)
    result = run_experiment_sync(config, resume=args.resume)
    summary = asdict(result)
    summary.pop("world_state")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
