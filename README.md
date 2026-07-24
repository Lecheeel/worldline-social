# Worldline Social

Worldline Social is a new, domain-focused social simulation built on [Worldline Engine](../worldline-engine). It is intentionally not an OASIS compatibility layer. OASIS is used only as a reference for experiments and social-platform capabilities; this project owns its own contracts, state model, and execution integration.

## First slice

The initial implementation provides a deterministic `SocialWorld` with:

- people as stable simulation entities;
- public posts and comments;
- feed, thread, and square-search reads;
- comment replies and idempotent post/comment likes;
- replaceable all-posts and recent-posts distribution policies;
- checkpoint and restore through the engine;
- rule and replay controller integration.
- a versioned JSON `PopulationManifest` with deterministic internal person IDs;
- a schema-versioned `SocialState` for checkpoint compatibility.
- bounded trait and dynamic-state models with deterministic per-tick recovery.

Public `handle` values and internal `person_id` values are separate. State,
events and relationships use `person_id`; the manifest preserves external ID
mapping for imports and reports.

Future modules will add population manifests, recommendation policies, social dynamics, memory, embeddings, and model providers.

## Development

Install the engine and this repository in editable mode while both are local:

```powershell
python -m pip install -e ..\worldline-engine
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

Optional memory integrations are kept outside the engine core:

```powershell
python -m pip install -e ".[vector]"
python -m pip install -e ".[embedding]"
```

## License

Apache License 2.0, matching Worldline Engine. See [LICENSE](LICENSE).
