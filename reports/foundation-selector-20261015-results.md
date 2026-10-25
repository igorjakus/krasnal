# Foundation100M: enumerate legal continuations and select by eval

Completed2026-10-15, no training. Registered foundation SHA256
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Same frozen1000 SF30k gold positions as the small SV baseline; gold SHA256
`b347129b66483430289dde08f3fca0020c8b7b5158168fd6aad69f64f5ded319`.

## Main result: no terminal overrides, just heads

| Method | Matches frozen SF best move | SF top1 | Delta vs policy |
|---|---:|---:|---:|
| Policy head, legal argmax |469/1000 |46.9% |— |
| All legal continuations, expected eval CP |507/1000 |50.7% |+3.8pp |
| All legal continuations, expected bounded logistic eval |516/1000 |51.6% |+4.7pp |

Paired bounded-eval vs policy:174 positions improve from mismatch to match,
127 worsen,699 have unchanged correctness;506 select exactly the same move.
These are diagnostic results on this fixed set, not a general significance or
Elo claim. Positions can share game histories, so a naive independent-sample
significance calculation would be inappropriate.

## Exact terminal-rule variant

Four positions have a terminal legal continuation. Overriding model eval with
exact mate/draw outcome changes CP selector to50.9% and bounded selector to51.8%.
Raw policy stays46.9%. Thus terminal overrides account for only0.2pp, not the
main gain. Both variant results and pinned snapshots remain preserved.

## Interpretation and timing

This is the user's proposed operation: complete UCI history + candidate move ->
post-move eval distribution -> select highest expected value for side to move.
The eval head is White-relative: White maximizes, Black minimizes. No rollout
tree, minimax, new fine-tuning, Stockfish inputs, shortlist, or true eval access.
Bin midpoints and +/-3000cp open-tail representatives fixed before scoring;
bounded score is E[logistic(.00368208*CP)], not calibrated WDL. Policy tie-break.
Both methods were declared before seeing outcomes; no scale/tail tuning on eval.

Mean GPU-model prefix-plus-all-candidate time18.30ms (raw),18.95ms (terminal
variant). This includes prefix prefill/cache copy/candidate forward/score reduction,
not Python history reconstruction, legal generation or terminal-rule handling.
Both1000-position sweeps took approximately29s wall time on RTX3090. Cached vs
uncached smoke max eval probability difference1.79e-7, policy logits also match.

The10.7M FEN SV baseline gave30.7% on these gold positions, but comparison is
descriptive: foundation100M has a different architecture/input and much more
training. No games were run here; neither51.6% nor51.8% is a win rate or Elo.
This warrants a bounded game comparison before any claim about playing strength
or an automatic expensive pretraining run.

## Artifacts

Worker roots `experiments/foundation-selector-20261015` and
`experiments/foundation-selector-raw-20261015`. Both containers exited0.
Raw JSON includes every position, selected moves, candidate counts/ranks and
terminal candidates. Small JSON copied to Mac; SHA256 compared with worker.
Original JSON SHA256:
`3b5ebd8675675c4e9a41b250ac79deef652ea8584975922759ea4ee5a943ffac`.
Raw-head control JSON SHA256:
`518b1edc388e070beda9718af715d1f26a50db6acb0ba378fddac3209367bbbe`.
Inputs use training-format prefix BOS,Elo2600,Elo2600,effort0, then all moves.
See adjacent protocol for source/runtime hashes and exact limitations.
