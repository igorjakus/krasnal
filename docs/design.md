# Design

## Goal

Show that a chess transformer that reasons in UCI before moving plays **stronger**
than the same model moving directly. The claim is relative, not state of the art.
The deliverable is the `<effort:0>` vs `<effort:4>` comparison at matched
wall-clock per move, measured as an Elo delta with error bars.

## Sequence format

```
<bos> <elo:6> <elo:7> <effort:2>
e2e4 e7e5 g1f3 b8c6
<think>
  <pv> f1b5 a7a6 b5a4 g8f6 <cp:18> </pv>
  <pv> f1c4 g8f6 d2d3      <cp:16> </pv>
</think>
f1b5
```

Move history, not FEN, is canonical: an 80-ply game yields 80 supervised targets
for ~85 tokens, where FEN-per-position yields one target for ~25. FEN token ids
are reserved for arbitrary-position setup (puzzles, UCI `position fen`) and are
trained as a small mixed fraction.

**Thinking is discarded from context after each move.** Keeping it would grow the
sequence ~35x and condition each move on stale analysis. Dropping it also keeps
the retained prefix stable, so the KV cache survives across moves and only the
thinking segment is evicted.

**Thinking is legality-masked.** A shadow board is maintained while generating a
PV, so every move inside `<think>` is legal by construction. Without this the
model can justify its choice with a hallucinated line.

## Vocabulary (2112)

Fixed and code-derived — no artifact file, no corpus dependency, no manifest to
validate. Moves are enumerated geometrically (queen and knight rays plus
promotions): 1968, the canonical count, covering every legal UCI string.

| block | count | notes |
| --- | --- | --- |
| structural | 7 | pad, bos, eos, think, /think, pv, /pv |
| elo | 9 | 200-point buckets, 1000..2400 |
| effort | 5 | 0 = no thinking; 4 = 5 PVs x 7 ply |
| score | 60 | 29 centipawn buckets + mate-in-1..8 + 15 delta buckets |
| fen | 28 | reserved |
| moves | 1968 | |
| pad | 50 | to a multiple of 64 |

Discrete score buckets rather than digits: the eval is where the lookahead signal
lives, and one token beats a multi-token noisy regression.

**The best line is scored absolutely; the rest by how much they lose.** Absolute
buckets are ~200 cp wide in the range that matters, so all five candidates of a
real position collapsed onto one token — the eval carried no signal about which
move to prefer. Delta buckets are fine (5 cp at the low end) because top-5 gaps
are almost always under 300 cp, and they match the stage-3 RL reward directly.

## Conditioning

Elo is kept, not filtered away. Conditioning is what lets low-rated games serve as
training data without dragging the policy down — the token absorbs the quality
variance. At play time both are clamped to the top bucket.

Conditioning tokens are never predicted. Because they only ever appear in a
prefix, the dataloader masks them by token identity and the on-disk format stays a
single flat `uint16` stream.

## Stages

1. **Pretrain** — human games, no thinking, `<effort:0>`, fixed-block packing.
2. **SFT** — Stockfish-annotated single decision points, mixed effort levels.
3. **GRPO** — only if stage 2 plateaus. Reward is negative centipawn loss against
   the annotator plus a legality/format term; single positions, no self-play.

## Rejected

- **MoE.** Decode at batch 1 is bandwidth-bound: 100M params in bf16 is 200 MB
  against 936 GB/s, so ~45 thinking tokens cost 15-30 ms per move against a ~3 s
  blitz budget. Three orders of magnitude of headroom. If it were slow the fix
  would be CUDA graphs, not routing.
- **flash-attn.** SDPA dispatches to the same kernel on CUDA and keeps MPS/CPU
  working for local debugging. Losing varlen packing costs padding, recovered with
  length-bucketed batching in stage 2.
- **Hydra.** Dataclasses and argparse until sweeps justify more.

## Hardware

One RTX 3090 (24 GB) and a 12-core / 24-thread i9-10920X, 31 GB RAM. Measured
annotation throughput with MultiPV 5, on 8 cores:

| nodes | positions/s | 2M positions, 20 workers |
| --- | --- | --- |
| 30k | 71 | ~2 h |
| 100k | 41 | ~7 h |
| 300k | 18 | ~14 h |

Default is 100k nodes. Stockfish 18 there is already well above 2700 Elo, so
deeper analysis buys accuracy a 100M model cannot absorb.

Positions are sampled uniformly over plies >= 10, capped at 2 per game. No
filtering on how sharp the position is: that needs the annotation before it can
decide, and it would skew the SFT distribution towards tactics.

Each position is analysed once at maximum effort; every lower effort level is a
truncation of the same record, so the ladder costs one engine pass, not five.
