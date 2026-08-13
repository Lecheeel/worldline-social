"""CLI subcommands for the task workflow (registered by the main CLI)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ..providers import DeepSeekProvider
from .models import STAGE_ORDER, SearchConfig
from .runner import TaskRunner
from .store import TaskStore

DEFAULT_TASKS_DIR = Path("runs/tasks")


def add_task_parser(commands: Any) -> None:
    task = commands.add_parser(
        "task", help="Task workflow: goal -> research -> design -> cast -> run"
    )
    sub = task.add_subparsers(dest="task_command", required=True)

    create = sub.add_parser("create", help="Create a task from a goal (+ optional material)")
    create.add_argument("--goal", required=True, help="Research/simulation goal")
    create.add_argument("--material", help="Seed material: file path or '-' for stdin")
    create.add_argument("--model", default="deepseek-v4-flash")
    create.add_argument("--budget", type=float, default=0.0,
                        help="Task budget in CNY (0 = unlimited)")
    create.add_argument("--max-queries", type=int, default=8)
    create.add_argument("--no-auto-run", action="store_true",
                        help="Plan research questions but do not execute searches yet")
    create.add_argument("--tasks-dir", default=None)

    status = sub.add_parser("status", help="Show a task's status and artifacts")
    status.add_argument("task_id")
    status.add_argument("--tasks-dir", default=None)

    show = sub.add_parser("show", help="Print one stage artifact as JSON")
    show.add_argument("task_id")
    show.add_argument("stage", choices=STAGE_ORDER)
    show.add_argument("--tasks-dir", default=None)

    edit = sub.add_parser("edit", help="Replace one stage artifact from a JSON file")
    edit.add_argument("task_id")
    edit.add_argument("stage", choices=STAGE_ORDER)
    edit.add_argument("file", help="JSON file with the new artifact payload")
    edit.add_argument("--tasks-dir", default=None)

    advance = sub.add_parser("advance", help="Execute the current stage, then stop")
    advance.add_argument("task_id")
    advance.add_argument("--tasks-dir", default=None)

    run = sub.add_parser("run", help="Execute stages until --stage (default: whole task)")
    run.add_argument("task_id")
    run.add_argument("--stage", choices=STAGE_ORDER, default=None)
    run.add_argument("--tasks-dir", default=None)

    export = sub.add_parser("export", help="Export task artifacts (manifest.json, experiment.json)")
    export.add_argument("task_id")
    export.add_argument("--tasks-dir", default=None)

    cancel = sub.add_parser("cancel", help="Cancel a task")
    cancel.add_argument("task_id")
    cancel.add_argument("--tasks-dir", default=None)


def _tasks_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "tasks_dir", None):
        return Path(args.tasks_dir).expanduser().resolve()
    return DEFAULT_TASKS_DIR.resolve()


def _read_material(location: str | None) -> str:
    if location is None:
        return ""
    if location == "-":
        return sys.stdin.read()
    return Path(location).read_text(encoding="utf-8")


def _runner(store: TaskStore) -> TaskRunner:
    def progress(task_id: str, stage: str, message: str) -> None:
        print(f"[{stage}] {message}", flush=True)

    return TaskRunner(store, on_progress=progress)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def handle_task_command(args: argparse.Namespace) -> None:
    store = TaskStore(_tasks_dir(args))
    try:
        if args.task_command == "create":
            record = store.create_task(
                goal=args.goal,
                seed_material=_read_material(args.material),
                model=args.model,
                budget=args.budget,
                search_config=SearchConfig(
                    max_queries=args.max_queries,
                    auto_run=not args.no_auto_run,
                ),
            )
            _print_json({
                **record.to_mapping(),
                "tasks_dir": str(_tasks_dir(args)),
                "next": "worldline-social task advance " + record.task_id,
            })

        elif args.task_command == "status":
            record = store.get_task(args.task_id)
            _print_json(record.to_mapping())

        elif args.task_command == "show":
            artifact = store.get_artifact(args.task_id, args.stage)
            if artifact is None:
                print(f"no artifact for stage '{args.stage}'", file=sys.stderr)
                raise SystemExit(1)
            _print_json(artifact)

        elif args.task_command == "edit":
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            store.set_artifact(args.task_id, args.stage, payload, edited=True)
            print(json.dumps({"ok": True, "stage": args.stage}, sort_keys=True))

        elif args.task_command == "advance":
            runner = _runner(store)
            task = asyncio.run(runner.advance(args.task_id))
            _print_json(task.to_mapping())

        elif args.task_command == "run":
            runner = _runner(store)
            task = asyncio.run(runner.run_until(args.task_id, stage=args.stage))
            _print_json(task.to_mapping())

        elif args.task_command == "export":
            for stage in STAGE_ORDER:
                path = store.export_artifact(args.task_id, stage)
                if path is not None:
                    print(f"exported: {path}")

        elif args.task_command == "cancel":
            task = store.set_status(args.task_id, "cancelled")
            _print_json(task.to_mapping())
    finally:
        store.close()
