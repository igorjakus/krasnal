# krasnal-think

## Gdzie pracujemy

Mac: kod i dispatch. Adrian: magazyn danych/modeli oraz stały RTX 3090.
UWR (Cymes): dodatkowy compute przez Slurm. Laboratoria: potencjalny dodatkowy
compute przez Cymesa. `lab007-03.stud.ii` (`192.168.4.104`) odpowiada na SSH;
logowanie, zasoby i ścieżki jeszcze do potwierdzenia.
Zarządzamy tym przez Git + SSH + Docker/Slurm + rsync, bez osobnego frameworka.
Zasady przypinania kodu, checkpointów i synchronizacji: [AGENTS.md](AGENTS.md).

## Required foundation checkpoint

New experiments must use `human-pretrain-100m-20260730`, pinned in
[configs/foundations.json](configs/foundations.json), at
`runs/foundations/human-pretrain-100m-20260730/model.pt` on workers. This is the
100M-game pretrain with ~100.76M parameters. Verify its SHA-256 before a run.
`runs/pretrain/model.pt` is the **retired 34M-game model**, not this foundation;
it is being archived and must not be selected for new training. See
[AGENTS.md](AGENTS.md) for synchronization and active-worker constraints.

A chess transformer that reasons in UCI before moving. It writes candidate
principal variations and their evaluations inside a `<think>` block, then plays a
move. The question it exists to answer: **does that reasoning make it stronger
than the same model moving directly?**

Not an attempt at state of the art, and not an attempt at human-like play — for
the latter see [krasnal](https://github.com/KSI-ML-Research/krasnal).

```
<bos> <elo:8> <elo:8> e2e4 e7e5 g1f3 b8c6 <effort:2>
<think> <pv> f1b5 a7a6 b5a4 <delta:0> </pv> <pv> f1c4 g8f6 <delta:4> </pv> </think>
f1b5
```

## Usage

The reproducible experiment starts from a pinned human pretrain and follows
10 rounds of SFT → fixed evaluation → DAgger → fixed evaluation. Data generations
are disjoint, checkpoints resume automatically, and fresh data can extend the
same chain. The protocol lives in [configs/curriculum-clean.json](configs/curriculum-clean.json);
see [docs/curriculum.md](docs/curriculum.md) for worker execution and provenance.

```sh
uv run scripts/experiment.py plan
uv run scripts/experiment.py run
uv run scripts/experiment.py status
uv run scripts/experiment.py append --sources data/sft/new-unique-teacher.jsonl
```

```sh
uv sync
uv run pytest

uv run scripts/download_pretrain.py --target-games 100_000_000
uv run scripts/pretrain.py --size xlarge --out runs/new-pretrain-experiment

uv run scripts/annotate.py --positions 2_000_000 --workers 20
uv run scripts/sft.py runs/foundations/human-pretrain-100m-20260730/model.pt

uv run scripts/gauntlet.py runs/sft/model.pt --games 400
```

On a GPU box, via Docker:

```sh
docker build -t krasnal-think .
docker run --rm --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs \
    krasnal-think python scripts/pretrain.py --size xlarge
```

`--effort 0` moves directly without emitting an effort token. Higher levels set
a maximum thinking-token budget; they do not prescribe a PV count or depth. See
[docs/design.md](docs/design.md) and [THINKING_FORMAT.md](THINKING_FORMAT.md).
