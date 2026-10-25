# Overnight distinct-Stockfish experiment, 8 October 2026

Worker: UWR `/Ziob/ijakus/krasnal-think`. Existing pretrain:
`experiments/sf-interleave-v1/checkpoints/pretrain-model.pt` (100.7M parameters,
14 layers, width 768, block size 1024). Expansion preserves original weights
and adds separate Stockfish move IDs (2112–4079, padded vocabulary 4096).
Actual available GPUs on opus2 and opus3: RTX 3080 10 GB, not RTX 3090.

Completed v2 raw data: 69 shards, 6,649,318 games; four v1 training shards are
also reusable. Raw annotations are not regenerated. Quick build selects twelve
evenly spaced completed shards; full build consumes all 73 completed shards.
Existing disjoint validation and test months are May and June 2024. Build checks
unique game IDs and disjoint source months and writes SHA-256 manifests.
Overlap with the original pretrain remains unknown, as its game IDs are absent.

Active jobs:

| Job | Purpose | Node |
| --- | --- | --- |
| 19072 | Quick build | opus3 |
| 19073 | Distinct SF interleave, two GPU DDP, then val/test evaluation | opus2 |
| 19074 | Human-only control, one GPU, then control and base val/test evaluation | opus3 |
| 19075 | Full build for subsequent reuse | opus3 |

Data roots: `data/sf-interleave-distinct-20261008-{quick,full}-buffered`.
Run root: `experiments/sf-interleave-distinct-20261008`, subdirectories
`augmented`, `human`, and `metrics`. Treatment uses `S W S B ...` with S
computed at the actual human prefix; evaluation supplies true SF and human
history. Both arms start from the same pretrain and expand to the same vocabulary.
Global batch is 32 sequences; requested steps are 20,000 treatment and 10,000
control, LR 1e-5, human/SF loss weights 2/1, checkpoints every 1000 steps.
Each arm ends after at most eight training hours and evaluates the saved final
model. Time cutoff may leave unequal human-target exposure; report actual step
counts and compare matched intermediate checkpoints when available. Final-model
comparison alone does not establish an equal-compute or equal-human-token result.

Jobs 18924–18927 and first slow-build attempt 19065–19068 were cancelled;
their outputs were preserved. Buffered byte writes replace per-game numpy
`tofile` calls to avoid excessive filesystem round trips. Smoke job 19069
completed two DDP optimization steps from pretrain on both opus2 GPUs.

Quick manifest confirms 1,156,769 train games, 78,044,747 human targets and
160,716,570 augmented tokens; validation has 9,677 games and test 9,666 games.
Run manifests hash the source files, source checkpoint and dataset manifest;
training summaries also record actual human-target exposure.

The previous monitor was absent in the app. Replacement monitor automation:
`kontrola-treningu-stockfish-co-4h`.

## Seminar deadline safeguards

Deadline: 9 October 2026, 09:00 Europe/Warsaw, report present on the Mac.
Independent evaluation job 19076 runs on the fourth GPU (opus3). It evaluates
the baseline immediately and predeclared human/treatment checkpoint pairs
1000/2000, 5000/10000, 10000/20000. Evaluation outputs are atomic, include source
checkpoint/dataset/records SHA-256, and retry with smaller batches on failure.

Mac collector is managed by launchctl label `com.krasnal.sf-seminar-20261009`:
`scripts/collect_sf_seminar.py`, wrapped with `caffeinate -dims`. It polls every
minute until 09:00, preserves fetched artifacts and verifies transfer checksums.
Local report: `reports/stockfish-seminarium-20261009/RAPORT.md`.
This requires the Mac to remain powered on and connected; it does not survive
a shutdown. Monitoring now runs every 15 minutes until the seminar deadline.

Recovery script supports `RESUME_FROM` and `MAX_HOURS` environment variables.
Model and optimizer restore were tested end-to-end on a tiny distinct-SF model.
Original checkpoints omit crop RNG, so recovery uses a new deterministic crop
stream. Summary explicitly reports exposure for the resumed invocation only.
No raw annotation or checkpoint is removed during recovery.
