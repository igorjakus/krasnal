# Mateusz/Lichess SV baseline

Status: finished.

10.7M parameters, scratch initialization; 20M available training positions.
Validation selects best checkpoint; SF30k is diagnostic, not a selection objective.

| Step | Exposures | Validation MAE | CB test MAE | SF30k top1 | Train phase seconds |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0.18983 | 0.21977 | — | 0.0 |
| 10000 | 2560000 | 0.11122 | 0.09445 | 0.179 | 477.2 |
| 20000 | 5120000 | 0.10566 | 0.09042 | — | 997.3 |
| 30000 | 7680000 | 0.10207 | 0.08660 | — | 1501.3 |
| 40000 | 10240000 | 0.10089 | 0.08401 | — | 2007.0 |
| 50000 | 12800000 | 0.09908 | 0.08286 | 0.257 | 2517.2 |
| 60000 | 15360000 | 0.09487 | 0.07823 | — | 3045.9 |
| 70000 | 17920000 | 0.09279 | 0.07531 | — | 3556.7 |
| 80000 | 20480000 | 0.09057 | 0.07422 | — | 4069.1 |
| 90000 | 23040000 | 0.08922 | 0.07222 | — | 4574.5 |
| 100000 | 25600000 | 0.08811 | 0.07129 | 0.307 | 5086.6 |

Single seed. No absolute Elo inferred from top1, loss or a bounded game screen.

Best-checkpoint final results (validation-selected):

```json
{
  "steps": 100000,
  "exposures": 25600000,
  "completed_updates": true,
  "train_phase_seconds": 5109.36389374733,
  "parameters": 10702848,
  "best_checkpoint_sha256": "5e2aef5361f497b850c8069d846addb6a7465028be189e0948a45ed74312b638",
  "latest_checkpoint_sha256": "66f3e7d0e54ffd44e58799dd8fe77ffd98e3edff878d59a42c95c79eba1d0032",
  "validation": {
    "soft_target_cross_entropy": 3.2305776916503905,
    "expected_value_mae": 0.08811311149597167,
    "n": 10000
  },
  "chessbench": {
    "soft_target_cross_entropy": 3.6911891235351564,
    "expected_value_mae": 0.07128863830566406,
    "n": 10000
  },
  "sf30k": {
    "sf30k_top1": 0.307,
    "n": 1000,
    "move_ms_mean": 19.789002694480587,
    "note": "Exact gold top1; no all-move regret or Elo claim"
  },
  "games": {
    "pilot": {
      "completed": 16,
      "wins": 13,
      "draws": 3,
      "losses": 0,
      "cutoffs": 0
    },
    "sf1600": {
      "completed": 16,
      "wins": 0,
      "draws": 5,
      "losses": 11,
      "cutoffs": 0
    }
  }
}
```
