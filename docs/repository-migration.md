# Repository migration — 2026-10-25

## Destination and GitHub identity

The requested destination is the original, public `KSI-ML-research/krasnal`
transferred to `igorjakus/krasnal`. Preserve repository ID `1156224823`, its 12
stars, branches, issues, pull requests and history. Do not make it private: GitHub
erases stars on a public-to-private visibility change.

The transfer is currently blocked by the existing `igorjakus/kobold` fork in the
same network. Its removal is authorized by the user, but the CLI token lacks
`delete_repo`. Browser deletion requires confirmation at action time. No transfer
has completed yet.

The initial private repository (ID `1433640657`) has been renamed to
`igorjakus/krasnal-monorepo-backup-20261025` to free the destination name. It is a
backup, not the intended final GitHub identity. Local checkout:
`/Users/igorjakus/Projects/krasnal-monorepo`.

## Modular code

The original `main` histories of both projects are ancestors of the combined main,
with original commit IDs preserved. The initial package-directory merge is part
of that history; the current tree has no `packages/` directory.

Code now lives in functional modules under `src/krasnal`: `models`, `tokenization`,
`data`, `training`, `inference`, `eval`, `preprocess` and `uci_engine`. One root
`pyproject.toml` and `uv.lock` describe a Python 3.13 environment. Historical
dependency files are retained only as provenance in `docs/migration/`.

`krasnal_think` and the old flat `krasnal.model`, `krasnal.tokens`, etc. are thin
aliases to canonical modules, preserving historical imports without duplicate
implementations. The architectures and vocabularies remain distinct. Do not infer
checkpoint interchangeability from the shared project layout.

`migration-sources.json` records hashes of the original imported trees, not hashes
of the refactored implementations. `migration/module-map.json` records module
relocations. Reports and their hash-bearing contents are preserved byte for byte.
Historical tracked tests are relocated, not rewritten; no new unit tests are added.

## Local and worker state

The old local checkouts remain available. Uncommitted tests, collector logs and
the continuously refreshed collector timestamp remain in the original
`/Users/igorjakus/Projects/krasnal-think`; they are not silently committed.
Ignored data, environments and checkpoints are not copied into this checkout.
Before retiring an old checkout, reconfigure collectors and migrate local work.

Worker paths, artifacts, active jobs and pinned snapshots remain unchanged.
Current launcher source paths are updated for the new modules, but historical
protocols still describe their original snapshots. New dispatch must pin the
complete source dependency closure, verify its hashes and smoke imports inside
the worker runtime. Do not overwrite active snapshots or resume a historical run
with refactored code without an explicit code/protocol audit.

## Verification

Both source main commits are ancestors of the merged main. Every imported report
matches its original blob. A wheel builds from the root project and the lockfile
passes `uv lock --check`. Smoke checks against the installed wheel verify both
architectures' initialization, logits, loss and gradients exactly match the
pre-migration implementation. Legacy class aliases resolve to canonical classes;
checkpoint save/reload and short direct/thinking legal games pass on CPU.

These are migration checks, not a new training experiment or Elo measurement.
After successful transfer, push the combined main as a fast-forward from the
original main and confirm repository ID and star count are preserved. Never force
push over newer upstream work.
