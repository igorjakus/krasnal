# Baseline audit — preliminary, do not claim +3.43 pp yet

The 100M foundation SHA-256 matches the original July30 checkpoint on both
workers: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
The June2024 test records SHA is unchanged:
`0cfd3bf2dcc42e156745576b132dd22555c54be31d4cae97c80c975e730f1885`.

**Confirmed input-format discrepancy:** original pretraining uses
`BOS, elo_white, elo_black, effort:0, human_moves, EOS`.
The SF interleave human-only baseline encoder omits `effort:0`.
This also affects older interleave baseline evaluations. It does not invalidate
the identical-format scratch SF versus scratch-star comparison, but it can
inflate the reported gain over the pretrained baseline.

An independent float32 CPU evaluation of 128 seeded random test games
(8,596 human moves, identical sample in both conditions) gives:

| Input | Human top1 | Human NLL |
| --- | ---: | ---: |
| Current baseline format, no effort0 | 52.0009% | 1.57559 |
| Pretraining-compatible format, effort0 supplied | 54.3625% | 1.40366 |

This is a paired sample, NOT a corrected full-test baseline. The treatment's
54.9458% was measured on all 9,666 games and must not be subtracted from these
sample percentages to estimate its gain.

Historical July30 metric was 53.710% on its own validation corpus and random
crop evaluation; it counts all non-conditioning labels, including EOS, whereas
the interleave result counts only human moves. An 8-crop probe gave 53.1631%
on the historical label definition, and 53.2921% on move-only labels. A full
100-crop historical reproduction completed and EXACTLY reproduced the original
53.710% (96,414 labels), eval-bucket accuracy40.059% and mean bucket error1.35.

CPU-only containers on Adrian:

- `kt-sf-baseline-audit-20261010`: paired 128-game check completed successfully.
- `kt-sf-baseline-audit-full-20261010`: full 9,666-game paired check in progress,
  result `experiments/sf-baseline-audit-20261010/audit-full.json`.
- `kt-sf-historical-audit-20261010`: original 100-crop evaluator completed,
  exit0, exact historical reproduction saved in `historical-reproduction.txt`.

Mac collector `com.krasnal.sf-baseline-audit-20261010` will fetch the full-test
diagnostic, verify remote output SHA and pinned checkpoint/record identities,
and write `WYNIK.md`. It does not overwrite the old canonical metrics.

No training weights, datasets, canonical evaluation results or active generator
were changed. Code resides in a separate worker snapshot. The GPU driver on
Adrian reports NVML driver/library mismatch, so this audit uses CPU, without
attempting to repair/reboot the host. CPU limits: full audit8, historical4,
alongside generation10, below the host's24 logical CPUs.
