# Foundation 100M: move + Stockfish eval head audit

Read-only GPU audit on 2026-10-15; no new training launched. Checkpoint identity
`human-pretrain-100m-20260730`, SHA256
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
100,756,992 parameters, 14 layers/768 width/12 heads. Two heads: next-token move
policy and 29-class post-move White-perspective Stockfish evaluation. Eval is an
auxiliary supervised target, never an input. Historical July30 training/eval is
recorded in MEMORY.md; this audit reproduces the original reported metrics.

## Frozen historical evaluation reproduced

100 random 1024-token validation crops, batch1, NumPy seed0, FP32, inference only.
Same checkpoint and validation streams; not a fresh independent test. Crops can
overlap and begin mid-game. No optimizer or update, GPU audit completed in2.21s.

| Metric | Result |
|---|---:|
| Valid eval positions |94346 |
| Exact eval bucket accuracy |40.0589% |
| Always predict training-majority bucket14 |10.1509% |
| Correct within1 bucket |73.5315% |
| Correct within2 buckets |87.1706% |
| Correct bucket among top3 |78.5227% |
| Mean absolute bucket error |1.34985 |
| Eval cross entropy |1.57269 |
| Macro bucket recall |34.7671% |
| Historical next-token accuracy incl EOS |53.7100% /96414 labels |

Bins are **nonuniform**, with edges recorded in JSON, from ±10cp near equal to
very broad extreme bins. Bucket error is not centipawn MAE;40% does not mean
40% exact numeric scores, win rate or Elo. It is not comparable with the SV
baseline's30.7% Stockfish **move** top1. Policy head learns human next moves,
not necessarily Stockfish's best move.

## Full corpus audit

Existing `data/pretrain-eval/train.bin` and `train.evals.bin` contain exactly
6,959,677,374 aligned tokens each,100,000,000 BOS/EOS games,6,459,677,374 moves.
6,419,897,987 moves have labels:99.3842% coverage. Missing39,779,387 move evals
were masked, not filled with fake evaluations.62,677,799 games have evals on
every move;37,322,201 have at least one missing label. These counts came from
full CPU scans, not a sample. Foundation data is therefore not literally100M
fully annotated complete games. There are also20000 held-out validation games.

## Hardware discovery / pending large-run decision

User authorized max48h and max2GPUs, but asked to inspect this existing head
before deciding initialization. No48h run submitted. Bounded Slurm audit19171
completed on UWR opus1: **2xRTX3080 10GiB**, torch2.8.0+cu128, not2xRTX3090.
opus0 remains unverified: interactive probe19167 did not launch a step, was
cancelled; its stalled local SSH client was closed. Exact-node bounded hardware
probe19172 remained pending Resources and was cancelled to avoid a later surprise
allocation. Training jobs must not be pinned to nodes just to spread load.
Adrian RTX3090 24GiB was free; CPU annotator left untouched. UWR free storage at
inspection134GiB on99%-full shared filesystem: keep transfers/checkpoints bounded.

## Reproducibility

Audit container `kt-foundation-eval-audit-20261015`, ID
`d38bc21448405a367016c266ab22458eb59f159724b4e09696c388dca1f474a7`, exit0.
Unique pinned root on Adrian: `experiments/foundation-eval-audit-20261015`.
Exact source SHA256 checked local/worker before launch; no existing sources edited.
Script `audit_foundation_eval.py` SHA256
`4f0b6d7aef2a63f53ec8ca2d1ef8feb3126741419d5cde290265a585300ed337`.
Raw JSON on Mac: `foundation-eval-audit-20261015.json` (including confusion matrix,
class counts, prefix examples, eval boundaries, validation hashes).
