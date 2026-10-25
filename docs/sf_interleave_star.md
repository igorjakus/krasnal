# Compute-matched scratch control: constant empty note

UWR jobs: smoke 19083 (after evaluator 19079 releases resources), training
19084 (after successful smoke), curve 19085 (after evaluator 19082 and smoke).
These are normal dependency waits; do not restart running experiments to free GPUs.
Mac collector: `com.krasnal.sf-star-20261009`, output
`reports/sf-interleave-star-full-20261009/RAPORT.md`.

Use `configs/sf-interleave-star-full-20261009.json`. This is a **scratch** control
for scratch SF, not a fine-tuned control. Same 7,035,726 original train games,
shuffled block order, seed 1234, 102,280,704 parameters, 1024-token windows,
global batch 32, LR schedule, human-only objective, and requested 59,631 steps.
Two passes do not double the number of unique games.

`scripts/train_sf_star.py` delegates to the existing pinned scratch trainer,
replacing every SF **input** identity (2112–4079) by the unused token 4080.
The exact same replacement applies to sampled validation. Original SF target
labels remain in the backing dataset solely to preserve identical zero-weight
loss operations; the wrapper asserts their loss weights are zero. They are not
fed to the network as inputs. No annotated corpus is duplicated or modified.
The config, source SHA-256 control manifest and original dataset manifest make
the transformation reusable. Dedicated evaluation replaces SF inputs and scores
**only human moves**, with the same output candidates as the other experiments.

The comparison reports both independent validation-selected checkpoints on the
same held-out test and common-step validation metrics. These answer different
questions: best model within the same budget versus learning at matched steps.
The 40-hour wall cutoff may produce unequal step counts; in that case the report
must flag it, not claim equal total compute. Sequence/batch/architecture FLOPs
are matched, not elapsed wall time. Continue a truncated run to the same step
budget before claiming a fully compute-matched final comparison.

Recovery MUST use `scripts/train_sf_star.py` (not the unmodified generic trainer).
Its `--resume-from` mode calls the generic resumable trainer with the input
replacement still installed. Pass the last checkpoint as positional argument
and as `--resume-from`, `--mode augmented --sampling shuffled-blocks`, the same
data/output, seed 1234, batch 2, grad-accum 16, steps 59631, LR 0.0003,
min-LR 0.00003, warmup 1000, human-weight 1, other-weight 0. Do not pass a
foundation ID: this run uses random initialization. Preserve all artifacts.

New worker scripts are deployed without overwriting pinned scripts used by
running jobs. `eval_sf_star_curve.py` is a deployed snapshot of the generalized
local scratch evaluation orchestrator, using `evaluate_star_interleave.py`.
