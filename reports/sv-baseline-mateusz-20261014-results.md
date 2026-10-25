# SV baseline — completed results (read 2026-10-15)

Container `kt-sv-baseline-mateusz-20261014` exited0, no OOM. Finished
2026-10-14 21:35 CEST. Container wall time2h9min, including preflight e2e smoke.
Training phase (including periodic validation/checkpointing)5109.4s =85.2min.
All100,000 updates completed before either cap; no new GPU job started on this turn.

## Setup and outcomes

Random-initialized10,702,848-parameter SV transformer;20M canonical-unique training
positions from the pinned Mateusz/Lichess corpus.25.6M exposures, shuffled without
replacement within epochs: approximately1.28 epochs. No PGN pretrain, DAgger,
text thinking, recurrence, search or RL. Minimum Mateusz validation MAE selected
checkpoint; the selected step was100,000, not selected by SF test or games.

| Metric | 1M-data Mateusz pilot | 20M-data baseline |
|---|---:|---:|
| SF30k exact move top1, same1000 positions |17.0% |30.7% |
| Mateusz validation value MAE, same10k positions |0.11838 |0.08811 |
| ChessBench held-out value MAE, same10k positions |0.10128 |0.07129 |
| Mean end-to-end SV move latency, RTX3090 |20.25ms |19.79ms |

MAE is error on the bounded logistic-transformed evaluation, not centipawns,
calibrated win probability or Elo. These runs differ in both available training
data and updates; they are not an isolated data-size ablation. Architecture and
evaluation stayed fixed. New dataset manifest reports0 train/eval overlap.

## Games

Eight fixed openings, both colours, greedy SV decision policy. Stockfish opponent
configured UCI_LimitStrength=true/UCI_Elo1600,20k nodes/move, one thread, Hash32MB.
Not a calibrated human/Lichess1600 opponent.10min aggregate screen budget.

| Opponent | W-D-L for baseline | Score | Unfinished |
|---|---:|---:|---:|
| 1M-data Mateusz pilot |13-3-0 |90.625% |0 |
| Stockfish configured1600 |0-5-11 |15.625% |0 |

All32 PGNs parsed locally without errors. Final boards/results verified by
python-chess:24 checkmates,8 claimable threefold repetitions. No ply/time-cap
draws were counted. Small fixed-opening screen, one seed: **no reliable absolute
Elo or DeepMind strength comparison**. Baseline clearly beats pilot in this
screen but remains substantially weaker than this Stockfish configuration.

## Learning curve

SF top1:17.9% at10k updates ->25.7% at50k ->30.7% at100k.
Mateusz validation MAE:0.11122 at10k ->0.09908 at50k ->0.08922 at90k ->0.08811
at100k. Validation was still improving at the end; no demonstrated convergence
plateau or architecture bottleneck. Full curve retained in `curve.json`.

## Provenance / retained artifacts

Worker root `/home/ijakus/projects/krasnal-think/experiments/sv-baseline-mateusz-20261014`.
Best checkpoint SHA256 `5e2aef5361f497b850c8069d846addb6a7465028be189e0948a45ed74312b638`.
Latest optimizer checkpoint SHA256 `66f3e7d0e54ffd44e58799dd8fe77ffd98e3edff878d59a42c95c79eba1d0032`.
Large data/checkpoints remain on Adrian. Original small artifacts copied to
`sv-baseline-mateusz-20261014-artifacts/`; each local SHA256 matched its worker source:

| Artifact | SHA256 |
|---|---|
| result.json |8437e217957a1865879109234dd5bef6eb651297c5393c7a66e11098113886c0 |
| status.json |a0869330348bc7f6fd8e69575810211f2f7aa8e71bb13f5aa142f50d061df74b |
| plan.json |c5b4a7eff55a6a326153de823936ca18d41431647d643ec54af01b481eac19c4 |
| data-manifest.json |2dcffe95335f72e7ae84732f4706fc3a511b117fa3bde5e5b9cdc9779ca64370 |
| curve.json |8836ca81e7304f84b537fea88d2926ca9d178b98d37508e329d653a06d91032f |
| games.json |e28920d6fe43399c2b0716b64d8269da3aee5573c0128b9352984e989d1c88cc |
| games.pgn |294a2f9c313c73bb5d129e8f6aff65b76b3ba92feed7c9169308e9a349459cb8 |
| report.md |2258a3d19fbbe8146a8a0c554d2de4919781667fdfb13ef315e6c24fa7125885 |
