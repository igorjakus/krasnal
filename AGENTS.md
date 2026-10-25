# Krasnal monorepo

- This checkout is the canonical repository for future code/config/docs edits.
- Work in `packages/krasnal` or `packages/krasnal-think` as appropriate; each has
  its own dependencies, lockfile and working-directory assumptions. Read nested
  instructions before modifying a package.
- Never write or add unit tests. Prefer smoke and end-to-end verification.
- Before shared compute work, read
  `/Users/igorjakus/.codex/skills/compute-hosts/SKILL.md`.
- Existing worker paths and pinned snapshots remain valid. Do not relocate worker
  data, overwrite active code/config or start competing GPU jobs for this migration.
- Preserve the 100M foundation identity and SHA-256 requirements in
  `packages/krasnal-think/AGENTS.md`. Relative paths there refer to that package
  locally and the existing project root on workers.
- Sync only exact needed files from the relevant package into a unique pinned
  worker snapshot, with SHA-256 provenance. Never sync this entire monorepo over
  an existing worker checkout. Bridge UWR/Adrian transfers through the Mac.
- Keep datasets, checkpoints, environments, credentials and raw collector logs
  outside Git. Preserve original checkouts and their uncommitted work until
  explicitly migrated; do not delete them or archive GitHub sources implicitly.
