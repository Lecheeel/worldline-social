# Worldline Social Project Rules

Worldline Social is the domain layer built on the separately published
`worldline-engine` package. It is a new implementation, not an OASIS API or
database compatibility layer.

## Boundaries

- Keep tick, turn, snapshot, commit, checkpoint, and event semantics in
  `worldline-engine`.
- Keep population, posts, comments, relationships, recommendation, memory,
  providers, and social dynamics in this repository.
- Do not import this package from `worldline-engine`.
- Do not let a Controller mutate SocialWorld or its storage directly; it must
  return structured `ActionIntent` values.
- Do not write credentials or private provider requests into events,
  checkpoints, or tests.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests examples scripts
python -m pip wheel . --wheel-dir dist --no-deps
```

The first social vertical slice must preserve deterministic results across
different engine concurrency settings and checkpoint restore/replay.

## License

Apache License 2.0, matching `worldline-engine`.
