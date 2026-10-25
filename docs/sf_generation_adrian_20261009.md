# Extra CPU Stockfish generation on Adrian

Validated at launch: real smoke completed 3 annotated games in 23.16 seconds;
full production then started 2016-01. Container ID
`56825549cd29c98813db788d1bd804c3b0c4867e849c21691db9aeb86f1b6062`.
Docker inspect confirmed GPU requests null, CPU cap10, RAM8GiB, no OOM.
Local targeted tests: 5 passed. This is startup verification, not completion
of the full 15-month schedule.

`COMPUTE.md` was read before dispatch. Lab007 `.104` answers on SSH, but its
key is not yet independently confirmed, so authentication was not bypassed.
Adrian is the additional trusted host; existing GPU curriculum is untouched.

Container `kt-sf-cpu-extra-20261009` is **CPU-only**, without GPU device requests,
capped at 10 logical CPUs, 8 GiB RAM and 256 processes. It uses 8 single-thread
Stockfish engines. Adrian had 24 logical CPUs, approximately 28 GiB available
RAM and 564 GiB free disk at dispatch. The audit-runtime image is
`sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.

Configuration: `configs/sf-generation-adrian-20261009.json`.
Sequential schedule: 2016-01 through 2017-03, 15 new source months, each up to
100,000 source games before filtering, up to 1.5M additional accepted games.
These months do not overlap UWR v1/v2 or the 33 queued missing v2 months.
Keep existing May/June2024 validation/test untouched. This is not a claim that
these games are unseen by the original human pretrain.

Output: `/home/ijakus/projects/krasnal-think/data/sf-interleave-v3-extra-20261009/raw/train`.
Cache: `data/source-cache/sf-adrian-20261009` on Adrian, not local `/tmp`.
Sources are retained, with a 30 GiB minimum-free-disk guard. The pinned code
snapshot is `experiments/sf-generation-adrian-20261009/code`; do not sync over it.
The engine was bridged UWR→Mac→Adrian with SHA-256
`38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`:
the exact same binary, 30k nodes, Threads1, Hash16 and source filters as UWR.

The container first runs a 3-source-game smoke into the run's separate `smoke/`
directory. Full generation starts only if smoke succeeds. Smoke outputs are
not production training shards. `generation.plan.json`, launch manifests and
final per-shard manifests record code/config/engine/output hashes and counts.
For interrupted runs, inspect `docker logs --tail 50 kt-sf-cpu-extra-20261009`
and `docker inspect`; fix only real errors. `docker start` reruns the same
pinned command: verified completed shards are skipped, partial IDs resumed.
Never start a second writer/container for the same months.

Current training datasets are immutable. Later, collect accepted completed
shards, validate manifests, deduplicate all training and held-out IDs, and build
a new version. The 4-hour monitor includes this container as well as UWR.
