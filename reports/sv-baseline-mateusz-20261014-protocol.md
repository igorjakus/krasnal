# Mateusz/Lichess SV baseline — 2026-10-14

User asked to train the baseline, not recurrence. New random-initialized 10.7M
board-state transformer, explicitly separate from the 100M PGN foundation and
old DAgger. Verify registered 100M foundation hash for provenance but load no
pretrained weights. Do not use or overwrite any old checkpoints or worker code.

## Frozen protocol

- Reuse pilot SV architecture, 71-token four-field FEN input, 128-bin HL-Gauss
  objective (sigma .75 bins), side-to-move labels and terminal-rule decision policy.
  Six 384-wide causal blocks, six heads; no MTP, recurrence, concepts, RL or PV.
- Data: 20M unique canonical positions from all ten pinned Mateusz Parquet files,
  2M/file, uniform reservoir then deterministic hash ordering; no depth filtering.
  Dedup SHA256/128 of canonical legal-EP FEN; ignore unavailable clocks.
- Exclude entire official ChessBench test, both pilot evaluation sets, complete
  SF30k gold histories plus legal children, and fixed game-opening positions.
- Validation: existing 10k Mateusz positions; minimum validation value MAE selects
  checkpoint. Existing 10k CB test and 1000 SF30k decisions are diagnostic only.
  Fixed evals are not used in the objective or learning-rate tuning.
- 100k updates x256 = 25.6M target exposures, epoch-shuffled without replacement.
  AdamW .9/.95, weight decay .1, clip1, LR3e-4, warmup1000, cosine floor3e-5.
- Phase budget: max90min including intermediate eval/checkpoint time. Global
  budget3h including preparation/final eval/games; deadline persisted once.
  Checkpoint every5k; learning-curve eval every10k; full decision eval at10k/50k/100k
  and selected best at the end. If capped, report actual counts, not target counts.
- Epoch permutation, optimizer, RNG, cursor, update, phase time and global deadline
  retained for bounded resumption. No implicit fresh restart or budget extension.
- Keep encoded data byte-sized on GPU, casting only selected batches. Two CPU
  threads for DuckDB/PyTorch; one GPU; leave CPU annotator and UWR untouched.
- Games: max10min total, 8 fixed openings with colour swaps vs the 1M-data pilot
  and SF UCI_Elo1600 /20k nodes. Ply/time cutoffs are unfinished, **not draws**.
  Small screen is not calibrated Elo. Record PGNs, engine/checkpoint hashes.

## Verification / safety

No unit tests. Executable smoke: encode/decode/EP/turn/castling equivalence to
pilot, target perspective/mates via pilot smoke, backward/optimizer/legal move;
GPU runtime smoke and worker data checks before full training. Sources and
config pinned in a new unique directory, exact-file SHA256 compared before launch.
Local checkout is dirty; preserve unrelated changes and do not commit wholesale.
No new scheduler, competing training job, foundation continuation or auto-scaling.

One seed, incomplete coverage of full316M source, heterogeneous teachers/budgets,
missing clocks and small game sample remain limitations. This baseline alone
does not demonstrate improvement over DeepMind.

Data attribution: Mateusz Grzyb,
[Lichess Stockfish Normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized),
CC BY4.0; original Lichess evaluations CC0. HF revision
`d83b3256d94a086f074a66db52f3a5b5d4c28847`; downloaded sources already exist on Adrian.

## Preflight provenance

- RTX3090 free, driver595.99.02; only `kt-sf-cpu-extra-20261009` active, CPU-only.
- Runtime image pinned to
  `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.
- Canonical HEAD `2662702d27bbdb2494c49e9a20a5935626e4461c`; dirty user changes
  preserved. Exact local/worker SHA256 checks agree:
  - `train_sv_baseline.py`: `b5bef4066f3cba756a070403b3e136d85fa98698b35ea5aebb212b2692659dcc`
  - unchanged pilot: `ca3dedd39c32f93989816ca7cfbedf372072b0274172f96878b138532933a022`
  - model blocks: `e2d37be6dd8859b85bea58ac83c0ba2ef466703c14891ab00ef294c747319704`
  - baseline config: `a1357b42d038056dd9cdb26d891d3612998c5b2100609f738c740bf85a62bebd`
- Local executable smoke passed encode/decode, EP/turn/castling, optimizer/backward
  and legal SV move checks. Full worker e2e smoke uses a separate `-smoke` root,
  10k positions, four updates and a 2-second game screen, not the full baseline budget.

### Smoke repair audit

First worker e2e smoke completed data prep, four GPU updates, checkpoint selection
and all1000 SF decisions. It then caught a PGN serialization bug: appending at
`game` after `Game.from_board(opening)` instead of at `game.end()`. No full baseline
training had started. Failed smoke outputs and plan are preserved in `-smoke`.
Old source SHA256 `6db9eb5403e04e829405d28ce09754841606e4310a8a84edc8bd5b8c6c3b916b`.
Exact source changes to new hash above: `node = game` -> `node = game.end()`;
e2e output suffix `-smoke` -> `-smoke2` (old artifacts not overwritten).
After repair, a GPU game smoke completed all32 four-ply opening/colour/opponent
cases, wrote PGNs and successfully parsed a PGN back without errors. Ply caps
were correctly retained as unfinished (`score=null`), not draws.
The detached full-run container repeats e2e in `-smoke2` before entering full prep.

## Dispatch

Detached container `kt-sv-baseline-mateusz-20261014`, ID
`d310a14aa9ab28a4cc9b67776acf07b22286542a105cd937ad217bc1aa3eaa48`.
Command runs bounded `--e2e-smoke` first and enters the full baseline **only on
exit0**. Each phase has an external timeout as well as the saved driver deadline.
No scheduled monitor or additional training jobs created. Worker root contains
`status.json`, `plan.json`, `data-manifest.json`, `updates.jsonl`, `curve.json`,
`latest.pt`, `best.pt`, fixed intermediate checkpoints, `result.json`, `report.md`,
and `games.json`/`games.pgn` as the respective stages complete.

Verified after launch: `-smoke2` completed the full prepare/train/checkpoint/eval/
game/report path and the `&&` driver entered the **full** root. Full status is
`preparing`, no container error/OOM; saved deadline `1792009677.8140197`
(2026-10-14 approximately22:28 CEST). Full-run training has not begun at this
handoff; the detached process will enter it automatically after preparing20M.
