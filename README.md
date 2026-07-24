# Worldline Social

Worldline Social is a new, domain-focused social simulation built on [Worldline Engine](../worldline-engine). It is intentionally not an OASIS compatibility layer. OASIS is used only as a reference for experiments and social-platform capabilities; this project owns its own contracts, state model, and execution integration.

## First slice

The initial implementation provides a deterministic `SocialWorld` with:

- people as stable simulation entities;
- public posts and comments;
- a read-only feed action;
- post likes;
- checkpoint and restore through the engine;
- rule and replay controller integration.

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
