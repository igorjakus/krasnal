"""Bounded 50/40/10 replay pilot; no human validation examples as training."""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import sft
from krasnal.tokenization import reasoning as tokens
from krasnal.training.sft import batches, load, load_onpolicy


def main():
    flag = sys.argv.index("--replay-root")
    source = Path(sys.argv[flag + 1])
    del sys.argv[flag:flag + 2]
    gold = {tuple(json.loads(line)["h"].split()) for line in (source / "eval-gold.jsonl").open()}
    cache = None

    def stream(args, device, seed):
        nonlocal cache
        if cache is None:
            own, seen = [], set()
            for round_index in range(1, 5):
                path = source / f"round-{round_index:03d}/onpolicy/labeled.jsonl"
                kept = 0
                with path.open() as handle:
                    for line in handle:
                        record = json.loads(line)
                        history = tuple(record["h"].split())
                        if history in gold or history in seen:
                            continue
                        # Use validated existing encoder without writing a merged dataset.
                        policy = record["onpolicy"]
                        trace = [tokens.ID[t] for t in policy["trace"]]
                        target = tokens.move_id(policy["target"])
                        roots = [trace[i + 1] for i, t in enumerate(trace[:-1]) if t == tokens.PV]
                        if policy["version"] != 1 or trace[0] != tokens.effort_id(2) or trace[-1] != tokens.THINK_END or target not in roots:
                            raise ValueError("inconsistent on-policy record")
                        elo = tokens.elo_id(2600)
                        context = [tokens.BOS, elo, elo] + [tokens.move_id(m) for m in history] + trace
                        own.append((context, [-1] * (len(context) - 1) + [target]))
                        seen.add(history)
                        kept += 1
                        if kept == 4096:
                            break
            def safe_teacher():
                for context, target in load(args.annotations[0], efforts=(0, 2), elo=2600, seed=42, compact=True):
                    stop = next((i for i, t in enumerate(context) if t in (tokens.effort_id(0), tokens.effort_id(2))), len(context))
                    history = tuple(tokens.VOCAB[t] for t in context[:stop] if tokens.MOVE_START <= t < tokens.MOVE_START + len(tokens.MOVE_IDS))
                    if history not in gold:
                        yield context, target
            teacher = list(itertools.islice(safe_teacher(), 8192))
            if not own or not teacher:
                raise RuntimeError("empty replay component")
            cache = own, teacher
            print("replay pool", len(own), "onpolicy deduplicated;", len(teacher), "teacher", flush=True)
        own, teacher = cache
        human = np.memmap("data/pretrain-eval/train.bin", dtype=np.uint16, mode="r")
        rng = random.Random(seed)

        def human_example():
            for _ in range(1000):
                offset = rng.randrange(1024, len(human) - 1)
                target = int(human[offset])
                if not tokens.MOVE_START <= target < tokens.MOVE_START + len(tokens.MOVE_IDS):
                    continue
                prefix = human[max(0, offset - 1023):offset]
                starts = np.flatnonzero(prefix == tokens.BOS)
                if not len(starts):
                    continue
                context = prefix[int(starts[-1]):].astype(int).tolist()
                history = tuple(tokens.VOCAB[t] for t in context if tokens.MOVE_START <= t < tokens.MOVE_START + len(tokens.MOVE_IDS))
                if history in gold:
                    continue
                return context, [-1] * (len(context) - 1) + [target]
            raise RuntimeError("cannot draw valid human training prefix")

        def samples():
            counts = [0, 0, 0]
            for _ in range(73728):
                r = rng.random()
                kind = 0 if r < 0.5 else 1 if r < 0.9 else 2
                counts[kind] += 1
                yield rng.choice(own) if kind == 0 else rng.choice(teacher) if kind == 1 else human_example()
            print("sampled before length bucketing", counts, flush=True)

        yield from batches(samples(), args.batch_size, args.block_size, device, seed=seed)

    sft.training_stream = stream
    sft.main()


if __name__ == "__main__":
    main()
