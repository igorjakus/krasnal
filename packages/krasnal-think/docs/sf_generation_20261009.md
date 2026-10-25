# Additional SF training corpus, 2026-10-09

Arrays 19086 (`0-16%1`, opus2, 26 Slurm CPUs / 24 engines) and 19087
(`17-32%1`, opus3, 34 CPUs / 32 engines) resume the 33 missing v2 training
months listed in `scripts/slurm/sf_more_months_20261009.txt`.
Each task has 10 GiB allocated memory and a 48-hour limit. One task per
array runs at a time; Slurm `JobArrayTaskLimit` on the rest is normal.
The arrays reserve sufficient CPUs and declared RAM for current and queued GPU
jobs. Existing training configs, source snapshots and datasets are untouched.

Source limit is 100,000 games per month, before the minimum-ply filter:
at most 3.3 million additional accepted games, likely slightly fewer.
The missing shards previously failed with `No space left on device` in
`/tmp/huggingface` and have empty partials. This is not evidence of bad games.
Current generators use persistent project-disk cache and temporary storage under
`data/source-cache/sf-more-20261009/`. Downloaded sources are retained. There was
approximately 483 GiB free at launch; start guard requires 30 GiB free.

Engine SHA-256 is verified against the existing SF 17.1 corpus:
`38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.
Settings: 30,000 nodes, Threads=1, Hash=16 MiB; minimum rating 1100,
minimum duration 180 seconds, minimum 20 plies. Every hint is computed at
the actual human prefix; only human moves advance the board.
May and June 2024 remain held out and are rejected by the wrapper.

Canonical `scripts/generate_sf_more.py` wraps the annotation generator and
bounds DuckDB memory to 2 GiB. UWR uses an immutable deployment snapshot of
canonical `scripts/sf_interleave.py` named `sf_generate_20261009_base.py`.
Each shard gets a `.manifest.json.launch.json` with source hashes, engine hash,
Slurm identity, cache path and preexisting record count; final manifests contain
accepted record count and output SHA-256. A Slurm array child's numeric job ID
can differ from its array ID. For monitoring use 19086/19087 or `<array>_<task>`.

Do not rerun the original annotator over completed shards: its resume path
can replace a complete output with an empty partial. This wrapper only skips
completed outputs after verifying their manifest SHA-256. Interrupted partials
resume by recorded game ID, without overwriting existing completed artifacts.
If a partial has a corrupt trailing record, preserve/archive it before repair.
No duplicate array may write the same month concurrently.

Outputs remain in `data/sf-interleave-v2/raw/train/`. Never rebuild the pinned
`sf-interleave-distinct-20261008-full-buffered` dataset while its runs are active.
When generation completes, build a new independently versioned dataset,
validate/deduplicate against all existing shards and held-out game IDs, and
retain manifests. Do not count repeated passes as additional original games.
The current 4-hour experiment heartbeat also watches these CPU arrays.
