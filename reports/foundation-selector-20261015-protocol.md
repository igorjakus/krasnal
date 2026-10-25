# Foundation eval-head one-ply selector — frozen protocol

User asked to enumerate all legal continuations of a UCI prefix and choose the
highest eval. Read-only experiment, no new training or modifications of the model.

- Registered foundation100M, SHA256
  `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
- Same frozen1000 SF30k gold positions used in the SV baseline. Reconstruct complete
  history from starting position; input BOS,Elo2600,Elo2600,effort0,history,candidate.
  Include effort0 **once in the training-format prefix**, avoiding the earlier
  historical baseline formatting bug. Never supply true evals or SF choices as input.
- All legal candidate moves evaluated, in chunks32. Prefix computed once and its
  KV state copied across candidate branches. Same absolute RoPE positions. Exact
  FP32 cached vs uncached smoke passed: max eval probability delta1.79e-7;
  policy logits also match normal GPT forward.
- Primary score: expected White-relative CP across29 predicted bins. Interior
  representatives are bin midpoints; open tail representatives fixed at +/-3000cp.
  White maximizes, Black minimizes. Policy probability is deterministic tie-break.
- Secondary, fixed in advance: expected logistic(.00368208*representative_cp).
  This bounded score is not calibrated WDL. No tuning on eval results.
- Baseline: same checkpoint/prefix, raw policy-head argmax among all legal moves.
- Value selectors use exact terminal results (mate/draw). Raw policy baseline has
  no terminal override; report terminal candidate positions for transparency.
- Report frozen SF top1 for all methods, paired gains/losses, candidate ranks and
  total prefix-plus-all-candidate inference timing. No Elo or game-strength claim.
- GPU was free on Adrian; CPU annotator untouched. One GPU, two CPUs, timeout15min.
  Runtime image pinned `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.
  Unique code snapshot `experiments/foundation-selector-20261015/code`.
  Script SHA256 `a4b27dbbe799c6a27492bbe16689e763131e7815ab92f67ff4f040fddd1d7a6f`;
  model/tokens sources SHA256 checked local/worker. Container
  `kt-foundation-selector-20261015`, ID
  `03acc28c75012b16a09603725a4a5dca366fbcf8d221dfbde3f121da58d11d81`.

No extra search depth, teacher scoring, policy reranking shortlist, model updates,
new scheduler or large48h training job in this experiment.

## Terminal-rule control

Initial evaluation has only4/1000 positions with any terminal continuation.
Added a verification run with `--raw-eval`, otherwise identical score definitions,
to measure the eval head without mate/draw overrides. Kept the original snapshot
and result intact; separate root `foundation-selector-raw-20261015`, script SHA256
`1226b87caaa347f2c7f89d7f96e49cdbd40350fefdcf14519bc6da5ad3ce38db`.
No score calibration, teacher tuning or training between the two evaluations.
