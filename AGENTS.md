# Krasnal workflow

## Source of truth

- This local checkout is the canonical code/config/docs repository.
- `adrian` is the GPU/training worker: `/home/ijakus/projects/krasnal-think`.
- `uwr` is the Slurm worker: `/Ziob/ijakus/krasnal-think` (case-sensitive).
- Keep large `data/`, `experiments/`, `runs/`, and checkpoints on workers; sync only the needed shard or artifact with a SHA-256 manifest.
- `adrian` cannot reliably SSH to `uwr`; bridge transfers through this Mac.

## Adrian

```bash
ssh adrian
cd /home/ijakus/projects/krasnal-think
docker run --rm --gpus all -v "$PWD:/app" -w /app krasnal-think:audit-runtime <command>
```

Adrian has no Slurm; use Docker for GPU work. Do not start competing GPU jobs.

## UWR / Slurm

```bash
ssh uwr
cd /Ziob/ijakus/krasnal-think
sinfo
sbatch scripts/slurm/<job>.sh
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <JOBID>
```

Use `student-cpu` for CPU Stockfish jobs and `student-nvidia` with `--gres=gpu:1` for GPU jobs. Set explicit `--cpus-per-task` and `--time`; write logs to `experiments/slurm/` and artifacts to the project tree.

## Sync rules

- Pull code/config from this checkout to workers before a run; never edit two copies in parallel.
- From this Mac, sync the lightweight checkout with `rsync -a --exclude data/ --exclude experiments/ --exclude runs --exclude .git/ ./ adrian:/home/ijakus/projects/krasnal-think/` (and replace `adrian:` with `uwr:/Ziob/ijakus/krasnal-think/` when needed).
- UWR outputs land in `/Ziob/ijakus/krasnal-think`; copy completed outputs to `adrian` through the Mac, validate/deduplicate there, and record source, parameters, counts, and SHA-256.
- Prefer resumable, atomic files (`*.partial` then rename). Never delete data or checkpoints; archive failed outputs.
