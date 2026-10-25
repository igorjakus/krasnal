# UWR node outage and scratch recovery

Slurm reported opus2 and opus3 DOWN+NOT_RESPONDING (Not responding) on
2026-10-09 at 18:25 according to its own timestamps. Both scratch training
and CPU annotation were requeued by Slurm. This is an infrastructure failure,
not a non-finite loss or an out-of-memory error diagnosed in our scripts.
Do not force node state changes, bypass Slurm, or use unverified lab host keys.

The 100M-foundation fine-tune and its evaluation completed beforehand:
59,631 steps, 33,429.59 seconds (~9h17m), human-only loss, same pinned 7,035,726
training games. Validation selected the final checkpoint. Test (9,666 games,
652,712 human moves) top1 51.517974% → 54.945826% (+3.427852pp),
NLL 1.577478 → 1.380821. True Stockfish teacher forcing. All 16 local metric
files were verified against the transfer manifest; report:
`reports/sf-interleave-100m-full-20261009/RAPORT.md`.
This demonstrates improvement over the unchanged baseline, NOT yet SF versus
the compute-matched empty-note control.

Old pending/requeued scratch job 19081 and star job 19084 were held and canceled
because their original fresh-start scripts would reject existing output roots.
They were replaced, without touching any running process, by:

- **19096**, `sf-scratch-recover`, opus3: latest atomic checkpoint step-30000.pt;
  replay post-checkpoint lost updates, same optimizer, shuffled cursor, seed,
  batch32 and LR schedule. The old log reached step31701 but has no newer
  complete checkpoint. Original files are preserved.
- **19097**, `sf-star-recover`, opus2: interrupted initialization left plan.json
  and a control manifest but no atomic checkpoint. The recovery launcher will
  archive this incomplete start and recreate the identical seeded scratch run.
  It never feeds true SF into star training/validation/evaluation.

Both use `scripts/slurm/restart_sf_scratch.sh` and
`scripts/restart_sf_scratch.py`, MODE=sf/star. They are pending until node
availability returns. This launcher is also idempotent across future node
requeues: choose the latest complete checkpoint; never use .partial models.
No foundation weights are loaded for either scratch variant. Planned budget
remains 59,631 steps. Three local recovery tests passed; deployed script SHA
matches canonical source. Recovery adds lost-work wall time; distinguish this
from the matched number of optimizer updates and human exposures.

Evaluation 19082 is unchanged and requeued; star evaluation 19085 retains its
dependency. CPU arrays 19086 and 19087 retain ID-based resume, one writer/shard.
Current partials 2018-10 and 2023-05 passed JSON parsing after the interruption.
71 completed v2 shards exist; storage still has about 430GiB free.
Adrian CPU generator remains running normally, outside this Slurm outage.
The four-hour monitor knows the replacement IDs and must remain active.
