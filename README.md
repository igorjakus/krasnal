# Krasnal

Chess transformers for human play, Stockfish value prediction and explicit
reasoning. One Python project, one environment and one dependency lockfile.

```text
src/krasnal/
  models/          # History transformer, reasoning transformer, clock conditioning
  tokenization/    # Distinct vocabularies for the two model architectures
  data/            # Human games, SF annotations, interleave and foundation registry
  training/        # Pretraining, SFT, RL, loss masking and optimizers
  inference/       # Cached generation, legal move selection and reasoning
  eval/            # Metrics, replay, probes and game arena
  preprocess/      # Downloaded-game preprocessing and packed datasets
  uci_engine/      # UCI protocol and model provider
configs/           # Training configurations and hash-pinned foundations
scripts/           # Training, data, evaluation and experiment entry points
reports/           # Audited results and SHA-256 manifests
docs/              # Methodology, compute workflow and migration provenance
```

```bash
uv sync --locked
uv run python scripts/training/pretrain.py --help  # history model
uv run python scripts/pretrain.py --help           # reasoning model
uv run python scripts/experiment.py --help         # configured SFT/DAgger pipeline
```

The architectures retain their own tokenization and checkpoint formats. Shared
organization does not imply interchangeable weights. Legacy import names are
thin compatibility aliases; new code uses the functional modules above.

- [History engine and UCI workflow](docs/history-engine.md)
- [Reasoning workflows](docs/reasoning-workflows.md)
- [Thinking format](THINKING_FORMAT.md)
- [Evaluation methodology](EVALS.md)
- [Human–SF depth comparison](reports/human-sf-depths-20261023/REPORT.md)
- [Unaided 2×2 comparison](reports/unaided-2x2-20261018/REPORT.md)
- [Repository migration](docs/repository-migration.md)

Large datasets and checkpoints stay on workers, outside Git. Read [AGENTS.md](AGENTS.md)
before dispatching compute; all new foundation-based training must use the verified
100M-game foundation from [configs/foundations.json](configs/foundations.json).
