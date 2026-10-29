# Consolidated checkout and workers — 2026-10-29

Canonical remote: `https://github.com/igorjakus/krasnal`.

| Host | Canonical root | Previous root |
| --- | --- | --- |
| Mac | `/Users/igorjakus/Projects/krasnal` | `krasnal-monorepo` and `krasnal-think` |
| Adrian | `/home/ijakus/projects/krasnal` | `/home/ijakus/projects/krasnal-think` |
| UWR | `/Ziob/ijakus/krasnal` | separate `krasnal` and `krasnal-think` trees |

The former `krasnal-think` paths are compatibility symlinks, not second checkouts.
They support existing chat working directories, historical container mounts and
saved protocols. Use the canonical root for all new work. There is one default
environment (`.venv`) per host, reconciled to the shared `uv.lock`.

## Preservation

- Original Mac source trees, Git object stores, dirty tests, report differences
  and collector state are in the ignored `.local/migration-20261029/` snapshot.
  Its manifest verifies 502 preserved files. Missing raw logs and local experiment
  artifacts were moved into the canonical root. No new unit tests were committed.
- Five obsolete launchctl collector services were removed. They were continuously
  restarting already completed collectors, with roughly 45,000–51,000 launches
  each. Do not recreate them automatically.
- Original worker source/config trees and Git metadata are archived under
  `.krasnal-migration-20261029` beside the canonical root. Old Git metadata is named
  `git-history`, preventing accidental use as a second active checkout. Restore
  it explicitly if investigating historical uncommitted worker changes.
- Data, experiments, runs, artifacts, caches and bot assets were relocated by
  same-filesystem rename, not copied through the Mac. No collision was overwritten.
  Before/after inventories compare file paths, inodes, sizes, timestamps and links;
  this is preservation verification, not a fresh full-content rehash of all data.
- Adrian: 28,801 relocated files, 279,117,467,550 logical bytes, plus a preserved
  9.3 GiB legacy recurrent-depth experiment under
  `experiments/legacy/recurrent-depth-20260803`.
- UWR: 73,371 relocated files and 1,240,420,533,057 logical bytes. Original history
  datasets and the reasoning datasets are present together. Counts include caches
  and the old environment as they existed at relocation, before environment sync.
- The foundation SHA-256 on **both** workers was independently verified as
  `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.

Full manifests stay in the ignored local snapshot and on their workers. Small
verification results and manifest checksums are published under
`reports/migration-20261029/`. Historical reports and pinned experiment code are
unchanged. A historical rerun must use its recorded runtime/lockfile, not assume
the new default environment reproduces old package versions.

## Runtime

Adrian uses Docker image name `krasnal:runtime` with the canonical root mounted at
`/app`. UWR setup and smoke checks run through Slurm, with caches in the project
tree and scratch space under `/tmp/$USER`. No training experiment was started.
The CUDA checks only allocate a four-element tensor.

The first UWR setup attempt (job 19382) failed before environment installation
because the Snap launcher could not create `/run/user/377821591`. Use the actual
installed binary `/snap/astral-uv/current/bin/uv` for noninteractive jobs on this
cluster, with a private `XDG_RUNTIME_DIR` under `/tmp/$USER`. Do not change system
permissions to work around this.

The inherited UWR environment also referenced a Python interpreter unavailable on
compute nodes. Rebuilding from the lockfile uses the project-local Python/runtime
cache instead of relying on that home-directory interpreter.

Final checks passed on all three hosts with PyTorch 2.14.1 (CUDA 13.0 wheels on
Linux): model imports, history forward/backward, direct/thinking legal games,
legacy aliases and checkpoint roundtrip. Adrian's RTX 3090 and the allocated UWR
RTX 3080 both passed the CUDA scalar check. UWR setup/smoke job 19383 and GPU smoke
job 19384 completed; no test jobs remain running.

## New work

Edit and commit on the Mac. Worker Git origins point at the canonical public
repository for read-only synchronization. Check jobs before syncing; every new
training run still needs an independent pinned code/config snapshot and hashes.
Never overwrite pinned historical experiment code or restart an old experiment as
part of checkout maintenance.
