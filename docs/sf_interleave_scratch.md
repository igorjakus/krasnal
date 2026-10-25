# Full interleave trained from scratch

Config: `configs/sf-interleave-scratch-full-20261009.json`. Run:
`experiments/sf-interleave-scratch-full-20261009-humanloss` on UWR.
Random seed 1234, no pretrained weights loaded. Same 14-layer/768-width
architecture and separate 4096-token vocabulary as the 100M-foundation
treatment (102,280,704 parameters, including unused 29-class eval head).

Same 7,035,726 training games and disjoint validation/test records. Two shuffled
block passes including the tail, global batch32, planned59,631steps. Human-only
loss, true SF as input at human prefixes; learning rate3e-4→3e-5, warmup1000,
gradient clip1, checkpoints and sampled validation human NLL every5000.
Learning rate differs intentionally from fine-tuning, so this is a performance
comparison, not a single-variable causal estimate of pretraining benefit.

UWR smoke19080 completed two finite updates, validation and checkpoint writes.
Full scratch job19081 uses the remaining GPU on opus3 without altering19078/19079.
Full validation evaluator19082 waits for19079 to release GPU resources, then
assesses saved checkpoints and selects by human validation NLL. It evaluates
the selected model on test once, against the already SHA-verified100M baseline.
The 40-hour training cutoff may stop before all requested steps; actual steps
and human-target exposure are saved explicitly. No ten-million-unique-game claim.

Mac collector: `com.krasnal.sf-scratch-20261009`, running `collect_sf_full.py`
with scratch config and output `reports/sf-interleave-scratch-full-20261009`.
No pretrained weights or historical artifacts are overwritten or deleted.
