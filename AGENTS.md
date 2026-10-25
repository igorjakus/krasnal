# Krasnal workflow

## Code and verification

- This checkout is canonical for code, configuration and documentation.
- Use one root `pyproject.toml` and `uv.lock`; run commands from the repository root.
- New code imports functional modules under `src/krasnal`: `models`, `tokenization`,
  `data`, `training`, `inference`, `eval`, `preprocess`, `uci_engine`.
- `krasnal_think` and the old flat module names are compatibility aliases for old
  callers/checkpoints. Do not duplicate implementations there.
- Never write or add unit tests. Prefer smoke and end-to-end verification.
- Preserve historical reports, manifests and pinned protocols unchanged; their
  hashes refer to the source snapshots used for those experiments.

## Compute

- Before shared compute work, read
  `/Users/igorjakus/.codex/skills/compute-hosts/SKILL.md`.
- Adrian is the primary GPU/artifact worker:
  `/home/ijakus/projects/krasnal-think`. UWR Slurm worker:
  `/Ziob/ijakus/krasnal-think` (case-sensitive). Worker paths remain unchanged.
- Bridge Adrian/UWR transfers through this Mac. Keep large datasets, experiments,
  checkpoints and environments outside Git; transfer exact needed artifacts with
  SHA-256 manifests and atomic/resumable copies. Never use `--delete` for data.
- Inspect active jobs before dispatch/sync. Use existing SSH, Docker, Slurm and
  rsync; no competing GPU jobs or overwriting active pinned code/config.
- Every run gets a unique directory, pinned code/config, source hashes, logs and
  job/container ID; one writer per output/shard. Jobs survive the Mac session.
- With modular code, a pinned snapshot needs the complete import dependency
  closure under `src/krasnal`, including package initializers. Do not copy only
  legacy alias files. Verify imports inside the pinned runtime before launch.
- Adrian has no Slurm; use Docker. UWR uses `student-cpu` for CPU work or
  `student-nvidia` with `--gres=gpu:1` for GPU work, explicit CPU/RAM/time requests
  and `experiments/slurm/` logs. Do not pin nodes without a documented requirement.
- Lab hosts/runtime/shared mounts remain unverified; do not guess them.
- Preserve original local checkouts and their uncommitted work. Reconfigure
  collectors/automations explicitly before retiring any old checkout.

## Mandatory foundation

- Current foundation: `human-pretrain-100m-20260730`, 100,000,000 training games
  (+20,000 validation), 100,756,992 parameters, 29-class eval head.
- Registry: `configs/foundations.json`; canonical worker artifact:
  `runs/foundations/human-pretrain-100m-20260730/model.pt`.
- Required SHA-256:
  `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
- Verify before new training. Record foundation ID/hash, dataset hash, source
  hashes, sampling, objective and GPU count in the run protocol.
- Never use `runs/pretrain/model.pt` or the v1 interleave pretrain copy for new
  training. SHA `52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84`
  is the retired 34M-game model. Archive, never delete; do not interrupt active
  historical jobs merely because they retain an old path.
- Historical 8–9 October interleave results used the retired model and a 1.16M
  subset. Do not mix them with the 100M/full-data experiment.
