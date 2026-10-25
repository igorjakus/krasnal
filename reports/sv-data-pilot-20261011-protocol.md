# SV data pilot — 2026-10-11

User-approved new small board-state transformer, trained from random initialization;
not another continuation of Krasnal DAgger. This explicitly follows the agreed
scratch SV baseline, rather than initializing this incompatible small architecture
from the mandatory 100M foundation. The current foundation is hash-verified and
recorded as lineage reference, **not loaded**; no retired checkpoint is used.

## Fixed protocol before dispatch

- Two serial arms: ChessBench SV vs Mateusz Grzyb's deduplicated Lichess evaluations.
- 1,000,000 unique training positions per arm, seed 42, identical initialized weights
  and fresh optimizer. Six 384-wide causal transformer blocks, six heads, no MTP,
  recurrence, concepts, PV, RL, search, or human pretrain. Existing pinned Block
  implementation uses RoPE/RMSNorm/SwiGLU; this is not an exact DeepMind architecture.
- Identical 71-token inputs: 64 board squares, turn, four castling flags, legal EP
  square and value readout. Remove clocks from **both** arms because Lichess stores
  four-field FENs. Dedup/split identity likewise ignores clocks.
- 128 value bins, HL-Gauss sigma 0.75 bins, mean per-position soft-label CE.
  Lichess cp/mate is converted White-relative -> side-to-move using the same
  logistic convention as ChessBench. Depth is recorded, not an input or a filter.
- 4,096 updates x 256 = 1,048,576 training exposures each. AdamW betas .9/.95,
  weight decay .1, gradient clip 1, LR 3e-4, warmup 100 then cosine to 3e-5.
- Train limit 45 minutes per arm; total run limit 6 hours including data download,
  preparation and evaluation. Deadline is persisted once; no silent reset/resume.
  Incomplete/capped arms do not constitute an equal-exposure result.
- Mateusz: revision pinned in config; uniform reservoir sampling across all 10
  files, not a potentially sorted first-N slice. Source files receive SHA256.
- ChessBench: uniform nonoverlapping record blocks from its shuffled train bag,
  using strict HTTP byte ranges and a pinned ETag. Preserve record-block indices
  and SHA256 of fetched payload/index ranges; do not download its full 36 GiB.
- Shared evaluation: 10k official ChessBench test positions, 10k held-out Mateusz
  positions, and existing 1000 SF30k curriculum gold decision positions. Exclude
  **all** official ChessBench test positions, held-out Mateusz positions, all gold
  histories and all legal children of gold positions from **both** train arms.
- Report value MAE / soft-target CE on both held-out sources, SF30k exact top1,
  complete update/exposure counts, train time and measured move latency. No games
  in this first pilot; no Elo or general strength claim from value loss/top1.

## Provenance and limits

Unique worker root `experiments/sv-data-pilot-20261011`, pinned `code/`, exact source
sync only; runtime image digest, Git dirty state, hashes and container ID archived.
Do not overwrite active annotation sources or interrupt its CPU-only container.
One training GPU, two CPU threads for loading/training, four allocated container CPUs.
Large data/checkpoints stay on Adrian; only small results/hashes come back to Mac.

This compares **data recipes**, not isolated teacher quality: distribution, engine
versions and search budgets differ. One seed and one small model are exploratory.
Mateusz evaluations mix engine versions/budgets; depth alone is not a quality proof.
No new unit tests: run executable smoke checks for actual Bag/Beam decode, target
perspective/mates, input identity, normalized loss, optimizer/backward and legal decisions.

## Preflight

- Adrian RTX 3090 free; only `kt-sf-cpu-extra-20261009` running, without GPU device
  requests. Driver 595.99.02; CUDA training smoke passed with bf16 + fused AdamW.
- Architecture: **10,702,848 parameters**. Five batch-256 GPU optimization steps
  including startup took 0.50 seconds (not an end-to-end runtime forecast).
- Local executable smoke passed, including real public ChessBench record decode.
- Foundation registry SHA256 matches mandatory 100M foundation.
- Runtime pinned image: `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.
- Canonical Git HEAD: `2662702d27bbdb2494c49e9a20a5935626e4461c`; checkout dirty,
  unrelated changes preserved. Only three exact files synced and SHA256 compared:
  - script: `ca3dedd39c32f93989816ca7cfbedf372072b0274172f96878b138532933a022`
  - model: `e2d37be6dd8859b85bea58ac83c0ba2ef466703c14891ab00ef294c747319704`
  - config: `216c6647fadefb9e0eb5a0839fd6823ef82ee3dba42f9b46b9b589c742c220e6`

## Dispatch

- Container `kt-sv-data-pilot-20261011`, ID
  `03658187c43cc047a65890dcaad13b57c1b1b53ce79edb2c050ec4c5acc4dfe8`.
- Started 2026-10-11 14:12 CEST. Persisted deadline `1791742322.8687944`
  (20:12 CEST). Detached Docker process; no session-bound training or new scheduler.
- Initial status verified `preparing`, container running, no OOM/error.
- Live state: worker `status.json`; logs: `docker logs kt-sv-data-pilot-20261011`.
  Results are written atomically to worker `results.json` and `report.md`.
