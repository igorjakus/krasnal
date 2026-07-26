# krasnal-think

A chess transformer that reasons in UCI before moving. It writes candidate
principal variations and their evaluations inside a `<think>` block, then plays a
move. The question it exists to answer: **does that reasoning make it stronger
than the same model moving directly?**

Not an attempt at state of the art, and not an attempt at human-like play — for
the latter see [krasnal](https://github.com/KSI-ML-Research/krasnal).

```
<bos> <elo:8> <elo:8> <effort:2>
e2e4 e7e5 g1f3 b8c6
<think> <pv> f1b5 a7a6 b5a4 <cp:18> </pv> <pv> f1c4 g8f6 <cp:16> </pv> </think>
f1b5
```

## Usage

```sh
uv sync
uv run pytest

uv run scripts/download_pretrain.py --target-games 10_000_000
uv run scripts/pretrain.py --size xlarge
```

`--effort 0` disables thinking; higher levels buy more and deeper principal
variations. See [docs/design.md](docs/design.md).
