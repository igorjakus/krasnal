# SV data comparison

Single-seed bounded pilot, not a DeepMind Elo reproduction.

| Data | Updates | SF30k top1 | CB value MAE | Mateusz value MAE | Training seconds |
|---|---:|---:|---:|---:|---:|
| chessbench | 4096 | 0.18 | 0.0876737045288086 | 0.1276831537246704 | 191.2 |
| mateusz | 4096 | 0.17 | 0.10128027839660644 | 0.11838002090454101 | 198.3 |

No measured Elo or games. Matched comparisons require equal completed updates.
Clocks removed from both inputs; best-depth Lichess labels still mix engine versions.
