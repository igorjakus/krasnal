# Krasnal

Private monorepo for the Krasnal chess models and experiments.

| Package | Purpose | Documentation |
| --- | --- | --- |
| `packages/krasnal` | Original chess engine, training and UCI tooling | [README](packages/krasnal/README.md) |
| `packages/krasnal-think` | Thinking, human/SF training and audited evaluations | [README](packages/krasnal-think/README.md) |

Both packages retain their own `pyproject.toml`, `uv.lock`, source modules and
configuration. Run each project's commands from its package directory:

```bash
cd packages/krasnal
uv sync --locked
```

```bash
cd packages/krasnal-think
uv sync --locked
```

Use a separate environment per package. This migration preserves implementations;
it does not make the models, tokenizers or checkpoints interchangeable.

Audited experiment reports are in
[packages/krasnal-think/reports](packages/krasnal-think/reports), including the
[human–SF depth comparison](packages/krasnal-think/reports/human-sf-depths-20261023/REPORT.md).

See [migration notes](docs/repository-migration.md) and
[source provenance](docs/migration-sources.json). Large datasets, checkpoints and
worker experiments remain outside Git.
