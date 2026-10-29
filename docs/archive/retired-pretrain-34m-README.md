# RETIRED — do not use this foundation

`runs/pretrain/model.pt` and the original SF-v1 `pretrain-model.pt` were the
**34,000,000-game** pretrain (~100.73M parameters), SHA-256
`52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84`.
They were mistaken for the newer 100M-game model. New training must reject
this hash, including copies under different filenames.

Use `runs/foundations/human-pretrain-100m-20260730/model.pt`, SHA-256
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
The registry is `configs/foundations.json` in the canonical local repository.

Retired checkpoints are preserved under `runs/archive/retired-pretrain-34m-20261009/`
or `experiments/archive/retired-pretrain-34m-20261009/`. They are for reproducing
historical results only, not defaults or fallback foundations. On Adrian the
move is deferred while `kt-curriculum-onpolicy-v1-20261008` is running, to avoid
breaking its already-pinned protocol. Its existing result remains a 34M-based
experiment even after the default foundation changes.
