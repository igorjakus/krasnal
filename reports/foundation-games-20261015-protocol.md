# Foundation games — frozen protocol, 15 October 2026

Objective: play actual games, not interpret SF-move prediction accuracy as Elo.
No training. Worker: Adrian, one RTX3090; existing CPU annotation untouched.

## Systems (200 games each)

1. Foundation100M, legal policy argmax, faithful effort0 input.
2. Same checkpoint, one-ply all-legal eval-head selection by expected sigmoid-transformed CP bin representatives; White maximizes, Black minimizes. Exact terminal outcomes, policy tie-break. This is not calibrated WDL.
3. Stockfish30k-node unrestricted teacher alone, assistance control.
4. Human–SF interleave fine-tune, legal policy argmax with actual SF30k suggestions supplied before EVERY ply, including historical opponent plies. No effort token, matching its training encoder. This is an SF-assisted system, NOT an unaided neural engine. Its eval head was not trained during this fine-tune and is not used.

Foundation SHA256: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
UWR interleave source: `experiments/sf-interleave-100m-full-20261009-humanloss/model.pt`;
SHA256 `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`, 1227294188 bytes.
Copy bridges through Mac, to a new Adrian run; original is preserved. A partial file is never loaded; verify size and SHA256 before atomic publication.

## Games and uncertainty

- Frozen 100 distinct, non-transposing eight-ply opening prefixes, seeded20261015 from the existing historical human validation stream; no outcome/eval-based selection. Not an external standardized opening book. Same openings for every system.
- Five opponent UCI_Elo settings:1320,1700,2100,2500,2900;20 pairs each. Both colours per opening. Single-thread Stockfish17.1,32MiB hash,20k requested nodes/move. Teacher uses30k nodes/move and no strength limit. All neural systems condition on2600/2600 and use FP32.
- Fixed800-game total, no significance-based early stopping. Maximum six hours from saved plan creation, never renewed on resume. Modes run in the above order while the UWR checkpoint transfers.
- No resignations or evaluation adjudication. Normal mate/stalemate/dead-position/repetition/move-count rules with draw claims enabled. A480-ply cutoff or wall deadline is **unfinished**, not a draw; report worst/best score rating bounds.
- Fractional-score logistic performance-rating fit against opponent setting labels. Report per-setting W/D/L as primary data and check for ceiling/floor or obvious style-dependent mismatch.
- Approximate95% stratified opening-pair bootstrap,10000 fixed-seed resamples; colours remain grouped. Also report a conservative95% Hoeffding pair-score interval, inverted through the logistic model. This remains valid at zero wins/losses under independent opening-pair sampling and the assumed rating response. Infinite endpoints are retained, not clipped into fake finite ratings.
- Incomplete samples are explicitly provisional; complete-case results with censoring can be biased. No optional-stopping confidence claim. Confidence does not account for engine Elo calibration, opponent-style/model misspecification, or domain shift.

This is a **Stockfish-setting-relative performance rating at this node budget**, not FIDE/Lichess Elo. UCI_Elo calibration uses a different time control; these numbers cannot establish an absolute human rating. Official background: https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html

## Execution and provenance

Run: `experiments/foundation-games-20261015`, pinned sources in `code/`.
Container intended: `kt-foundation-games-20261015`.
Image: `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.
Engine SHA256: `38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.
Container ID: `615b93a981a3676405c717cc1f249ab9842487905ffd9f8c2a2ca765884b6634`.
Runner SHA256: `48c2e0ea226fb71fd49d7fb1fa9cf22392468d748ccbb8b308f202001c97dc39`.
Selector helper: `1226b87caaa347f2c7f89d7f96e49cdbd40350fefdcf14519bc6da5ad3ce38db`.
Model source: `e2d37be6dd8859b85bea58ac83c0ba2ef466703c14891ab00ef294c747319704`.
Tokens: `9410613365aeacc027293509f4f881319c05b40f2525a965b8f9f21f31ad84c6`.

Smoke: cached/uncached value agreement max1.79e-7, policy agreement, white/black score direction, known-rating/statistical endpoints, six short legal games foundation/control both colours. Interleave is hash/config/legality checked at first load after transfer; its results must not be claimed before that happens.
Artifacts: immutable `plan.json`; append-only `games.jsonl`,`games.pgn`; atomic `status.json`,`result.json`,`complete.json`; container logs. Result summary every40 games and at exit. No unit tests added.
