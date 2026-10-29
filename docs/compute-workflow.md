# Krasnal workflow

## Source of truth

- This local checkout is the canonical code/config/docs repository.
- `adrian` is the GPU/training worker: `/home/ijakus/projects/krasnal`.
- `uwr` is the Slurm worker: `/Ziob/ijakus/krasnal` (case-sensitive).
- Keep large `data/`, `experiments/`, `runs/`, and checkpoints on workers; sync only the needed shard or artifact with a SHA-256 manifest.
- `adrian` cannot reliably SSH to `uwr`; bridge transfers through this Mac.

## Compute / dispatch

Mac edits and dispatches; Adrian is the primary artifact store and reliable
RTX 3090 worker. UWR = Cymes provides queued Slurm capacity. Lab machines are
lower-reliability extra capacity, reached through Cymes; their exact hostnames,
runtime and shared-disk mount path are **not yet verified**. Do not guess them.
On 2026-10-09, `lab007-03.stud.ii` / `192.168.4.104` answered SSH from
Cymes; login is pending independent host-key verification. GPU/runtime and
shared mount remain unverified. See generic notes for the observed fingerprint.
Shared host skill: `/Users/igorjakus/.codex/skills/compute-hosts/SKILL.md` on the Mac.

- Use existing SSH/Docker/Slurm/rsync, not another dispatcher framework.
- Each run gets a unique experiment directory, pinned code/config, SHA-256
  provenance, logs and container/job ID. Keep jobs independent of the Mac session.
- One writer per output/shard, including across shared UWR/lab storage.
  Labs should only run resumable work after an access/runtime smoke check.
- Prefer Adrian for continuous GPU training, Slurm for parallel CPU annotation
  and extra GPU capacity. Check running jobs before allocating resources.
- Slurm jobs should request GPU/CPU/RAM resources, not a fixed node. Use
  `--nodelist` only for a documented hardware/runtime requirement; do not pin
  jobs merely to spread load. Preserve resource requests when removing a pin.

## Foundation checkpoint — mandatory

- The current foundation is **human-pretrain-100m-20260730**: 100,000,000
  training games (+20,000 validation), 100,756,992 parameters, 29-class eval head.
- Registry: `configs/foundations.json`. Canonical worker path:
  `runs/foundations/human-pretrain-100m-20260730/model.pt`.
- SHA-256 must be `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
  Validate before submitting any new training. Record foundation ID, hash,
  dataset hash, source-code hashes, sampling, objective and GPU count in the run.
- **Never use `runs/pretrain/model.pt` or the v1 interleave pretrain copy for
  new training.** SHA `52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84`
  is the retired **34M-game** model, not the 100M foundation. Archive, never delete.
  An existing Adrian container may temporarily retain its old path; do not
  interrupt it or treat this exception as permission for new runs.
- Historical SF interleave results from 8–9 October used the retired foundation
  and a 1.16M-game subset; do not mix them with the new 100M/full-data experiment.
- Before syncing, inspect other active worker jobs. Do not overwrite their
  pinned code/config. Sync exact needed files and compare SHA-256; use a unique
  experiment directory per protocol. Paths or similar names do not prove identity.

## Adrian

```bash
ssh adrian
cd /home/ijakus/projects/krasnal
docker run --rm --gpus all -v "$PWD:/app" -w /app krasnal-think:audit-runtime <command>
```

Adrian has no Slurm; use Docker for GPU work. Do not start competing GPU jobs.

## UWR / Slurm

```bash
ssh uwr
cd /Ziob/ijakus/krasnal
sinfo
sbatch scripts/slurm/<job>.sh
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <JOBID>
```

Use `student-cpu` for CPU Stockfish jobs and `student-nvidia` with `--gres=gpu:1` for GPU jobs. Set explicit `--cpus-per-task` and `--time`; write logs to `experiments/slurm/` and artifacts to the project tree.

## Sync rules

- Pull code/config from this checkout to workers before a run; never edit two copies in parallel.
- Default: sync exact required files into a new run's pinned code directory;
  compare SHA-256 before launch. Full checkout sync is only safe after checking
  that it cannot overwrite code used by active jobs. Never use `--delete` for data.
- UWR outputs land in `/Ziob/ijakus/krasnal`; copy completed outputs to `adrian` through the Mac, validate/deduplicate there, and record source, parameters, counts, and SHA-256.
- Prefer resumable, atomic files (`*.partial` then rename). Never delete data or checkpoints; archive failed outputs.
