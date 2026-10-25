# SV data pilot — completed results

Read live from Adrian on 2026-10-14. Container `kt-sv-data-pilot-20261011`
exited 0, no OOM. Started 2026-10-11 14:12 CEST, finished 14:31 CEST.
Total driver wall time: 1148.2 s (19.1 minutes). No new training dispatched.

## Matched setup

10,702,848-parameter scratch SV transformer, 1M unique available training positions
per source, seed 42, 4096 updates x batch 256 = 1,048,576 draws **with replacement**
per arm. Not a full sweep over every training position. Both completed all updates;
identical initial tensor SHA256:
`d0f16d974836b1185688fc0e6174f6070daab6e180ba9d3c7d739bc7117c853c`.
Architecture/input/optimizer/loss/sampling protocol is in the pre-dispatch report.

## Results

| Training data | SF30k top1 / 1000 positions | CB test value MAE / 10k | Mateusz test value MAE / 10k | Train seconds | Mean move latency |
|---|---:|---:|---:|---:|---:|
| ChessBench | 18.0% | 0.08767 | 0.12768 | 191.2 | 18.86 ms |
| Mateusz/Lichess | 17.0% | 0.10128 | 0.11838 | 198.3 | 20.25 ms |

Value MAE is absolute error on the bounded logistic-transformed evaluation in
[0,1], not centipawn error, calibrated win probability, game score or Elo.
Latencies include Python board handling and terminal-rule checks, not just a
neural forward. The same 1000 gold positions were used for both decision evals.

## Interpretation

Both arms trained successfully, using only 389.5 seconds of combined training
wall time. Each does better on its own source's held-out evaluation. That is
consistent with a source-distribution/teacher difference; it does not establish
that either source is universally better. The 1 percentage-point SF top1 gap is
inconclusive: one seed, tiny training budget, and no paired significance analysis.
No games or Elo were measured. This pilot does not demonstrate improvement over
DeepMind's trained models. No recurrence or other architecture innovation yet.

Suggested next experiment (not dispatched): keep the same evaluation and
architecture, increase matched exposure budget and use multiple seeds before
selecting the data source. A bounded game comparison is needed for playing strength.

## Checkpoints retained on Adrian

- ChessBench `chessbench-model.pt` SHA256:
  `a8de6bfba3fa5b7fedf3dd3be189314ff6a1ffab02b16371765ffd3fbe6efe88`
- Mateusz `mateusz-model.pt` SHA256:
  `63b4772fe95875be323f024526035d009c72c53673e7bb1a0f26bc27833f164c`

Root: `/home/ijakus/projects/krasnal-think/experiments/sv-data-pilot-20261011`.
Small original artifacts are copied to the adjacent `sv-data-pilot-20261011-artifacts`
directory, with source/local SHA256 verification. Large data/checkpoints stay on worker.
