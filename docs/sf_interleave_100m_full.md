# Correct 100M foundation + full Stockfish interleave

Configuration: `configs/sf-interleave-100m-full-20261009.json`.
Foundation registry: `configs/foundations.json`. Both identify the source by
SHA-256, not by a generic `pretrain` filename. Retired 34M checkpoint hashes
are rejected by the interleave trainer and future curriculum launcher.

All 7,035,726 ready training games are used, with 474,419,960 original human
moves and 976,982,824 interleaved tokens. Two passes through shuffled,
rank-assigned 1024-token blocks give about 14M game-equivalent exposures;
there are still only 7.04M distinct annotated games. The tail is covered by one
overlapping final window; global batch rounding may add a few extra blocks.

Training starts from the untouched 100M-game model, never from the old SF
treatment. Stockfish inputs have separate token IDs, initialized from the
corresponding pretrained human move embeddings. Other weights are preserved,
including the 29-class evaluation head (unused in this experiment). Only human
labels carry training loss; SF labels are inputs, not an auxiliary prediction
objective. This objective differs from the old pilot's human/SF weights 2/1.

Validation human NLL/top-1/top-5 are computed on all 9,677 held-out May-2024
games at each 5,000-step checkpoint. The minimum validation human NLL selects
the final checkpoint. Only that checkpoint is evaluated on the 9,666 June-2024
test games, compared with the original 100M pretrain. True SF labels are
teacher-forced at every human prefix. Learning curves show whether improvement
continues or plateaus; no plateau claim is made before observing the curve.
Pretrain-source overlap remains unknown. This is one run, not a significance
claim or an equal-compute controlled experiment.

Artifacts live at `experiments/sf-interleave-100m-full-20261009-humanloss` on UWR.
Mac collector `scripts/collect_sf_full.py` verifies transfer SHA-256 and saves
`reports/sf-interleave-100m-full-20261009/RAPORT.md`. No raw games or historical
checkpoints are deleted. Sync exact required files from the canonical Mac;
do not overwrite code/config pinned by the existing Adrian curriculum run.

Submitted UWR jobs: smoke 19077, full treatment 19078 (afterok smoke), learning
curve/final selection 19079 (afterok smoke). Mac collector launchctl label:
`com.krasnal.sf-full-20261009`. Planned steps: 59,631. Monitoring is recorded in
the existing Stockfish heartbeat; no new human-only control is submitted.

Retired copies were successfully moved on both UWR and Adrian. The old Adrian
container exited before its archival watcher moved the file. No checkpoint was
deleted; generic retired paths now fail instead of silently selecting a model.
