# Repository migration — 2026-10-25

## Destination and GitHub identity

On 2026-10-29, the original public `KSI-ML-research/krasnal` was transferred to
`igorjakus/krasnal`. Repository ID `1156224823` and all 12 stars were verified
unchanged after transfer. Branches, issues, pull requests and repository history
remain attached to that original repository. Do not make it private: GitHub
erases stars on a public-to-private visibility change.

The old `igorjakus/kobold` fork was deleted with the user's explicit approval,
after GitHub reauthentication, to unblock the transfer. The old organization URL
redirects to `igorjakus/krasnal`.

The initial private repository (ID `1433640657`) has been renamed to
`igorjakus/krasnal-monorepo-backup-20261025` to free the destination name. It is a
backup, not the canonical GitHub repository. Local checkout:
`/Users/igorjakus/Projects/krasnal` (consolidated on 2026-10-29).

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

Old local source trees, Git object stores and uncommitted work are preserved in
the ignored `.local/migration-20261029/` snapshot. Missing logs and local experiment
artifacts were moved into the canonical checkout; differing historical reports
and uncommitted tests remain preserved in the snapshot rather than silently
overwriting published artifacts. The default `.venv` is reconciled to the unified
lockfile. Five obsolete, repeatedly restarted collector services were removed.

Worker roots are consolidated to `/home/ijakus/projects/krasnal` (Adrian) and
`/Ziob/ijakus/krasnal` (UWR). The former `krasnal-think` paths are compatibility
symlinks, preserving historical paths and container mounts. Artifacts and pinned
experiment snapshots are relocated without changing their contents. Original
worker source trees and Git metadata are archived separately; they are not
overwritten. See the worker migration report for runtime verification.

New dispatch must pin the complete source dependency closure, verify its hashes
and smoke imports inside the worker runtime. Do not resume a historical run with
refactored code without an explicit code/protocol audit.

## Verification

Both source main commits are ancestors of the merged main. Every imported report
matches its original blob. A wheel builds from the root project and the lockfile
passes `uv lock --check`. Smoke checks against the installed wheel verify both
architectures' initialization, logits, loss and gradients exactly match the
pre-migration implementation. Legacy class aliases resolve to canonical classes;
checkpoint save/reload and short direct/thinking legal games pass on CPU.

These are migration checks, not a new training experiment or Elo measurement.
The combined main is published as a fast-forward from the original main. Never
force push over newer upstream work. The canonical remote is
`git@github.com:igorjakus/krasnal.git`; the private migration backup remains under
the `migration-backup` remote for recovery.
