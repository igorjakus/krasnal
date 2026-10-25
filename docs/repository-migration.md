# Repository migration — 2026-10-25

Destination: private `igorjakus/krasnal`.
Local checkout: `/Users/igorjakus/Projects/krasnal-monorepo`.

The original `main` histories from `KSI-ML-research/krasnal` and
`igorjakus/krasnal-think` are joined without rewriting their commit IDs. Original
Krasnal files were moved under `packages/krasnal`; the complete Krasnal Think
tree was merged under `packages/krasnal-think`. The source commit/tree hashes
are recorded in `migration-sources.json`; imported package trees must match them
exactly at migration time. Historical tracked tests are retained unchanged.

The two source repositories and local checkouts are preserved. This is a new
private repository, not an ownership transfer, fork or deletion. GitHub issues,
pull requests, repository settings, collaborators, releases and non-main branches
are not migrated. Consult the source repositories for them. Original Krasnal has
branches differing only in case (`KV-cache` / `kv-cache`), which cannot both be
stored with the default Git ref backend on this Mac's case-insensitive filesystem.

Local uncommitted tests, collector logs and the continuously updated collector
timestamp remain in `/Users/igorjakus/Projects/krasnal-think`. They were not
silently committed or overwritten. Ignored datasets, environments and checkpoints
are not copied into this checkout. Before retiring an original checkout, stop or
reconfigure its collectors and explicitly migrate any remaining local work.

Run commands from the relevant package directory. Lockfiles and Python dependency
sets remain separate; there is no shared uv workspace yet. Existing scripts may
contain historical absolute checkout paths: audit them before dispatching jobs
from the new checkout. Worker paths, artifacts, active jobs and pinned snapshots
are unchanged. Deployment sync must use the package root, not the monorepo root.

The original `.github` workflow is preserved inside the original package for
provenance. It is not a root GitHub Actions workflow. CI has not been redesigned
or enabled by this migration.

Future work should use `igorjakus/krasnal`. Source repositories remain available
for rollback and historical references; archive them only after collaborators,
collectors, automation prompts and worker dispatch paths have been reviewed.

Verification completed on the Mac: both imported Git trees and every file's
SHA-256 match the source commits; both source commits are ancestors of the merged
main; 118 source/script Python files parse; both model modules import from the new
checkout; a tiny random CPU model plays eight legal plies with both direct and
thinking generation. This is a relocation smoke check, not a training run or an
Elo measurement. No new unit tests were added.
