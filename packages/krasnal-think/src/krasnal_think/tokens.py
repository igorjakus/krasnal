"""Fixed, code-derived vocabulary. No artifact file, no corpus dependency.

Layout: structural | elo | effort | eval | fen | moves, padded to a multiple of 64.
The move set is enumerated geometrically (queen + knight rays, plus promotions),
so every legal UCI string in any position maps to a token.
"""

from bisect import bisect_left

FILES = "abcdefgh"
SQUARES = [f + r for r in "12345678" for f in FILES]

ELO_EDGES = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]  # -> 9 buckets
EFFORT_LEVELS = 5
# Maximum number of tokens between <think> and </think>. Effort controls compute,
# not a fixed number or depth of principal variations.
EFFORT_BUDGET = {0: 0, 1: 12, 2: 24, 3: 36, 4: 50}
CP_EDGES = [
    -2500,
    -1600,
    -1200,
    -800,
    -600,
    -400,
    -300,
    -200,
    -150,
    -100,
    -75,
    -50,
    -25,
    -10,
    10,
    25,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1200,
    1600,
    2500,
]  # -> 29 buckets
# How much worse than the best move a variation is. Top-5 gaps are usually small,
# so these are far finer than CP_EDGES, which cannot separate candidate moves.
DELTA_EDGES = [5, 10, 15, 25, 40, 60, 90, 130, 180, 250, 350, 500, 700, 1000]  # 15 buckets
MAX_MATE = 8
MATE_CP = 10_000


def _knight_targets(fx: int, fy: int) -> list[tuple[int, int]]:
    return [
        (fx + dx, fy + dy)
        for dx, dy in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    ]


def _ray_targets(fx: int, fy: int) -> list[tuple[int, int]]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        for n in range(1, 8):
            out.append((fx + dx * n, fy + dy * n))
    return out


def enumerate_moves() -> list[str]:
    moves = []
    for fy in range(8):
        for fx in range(8):
            source = FILES[fx] + str(fy + 1)
            for tx, ty in _ray_targets(fx, fy) + _knight_targets(fx, fy):
                if not (0 <= tx < 8 and 0 <= ty < 8):
                    continue
                moves.append(source + FILES[tx] + str(ty + 1))
    for from_rank, to_rank in ((7, 8), (2, 1)):
        for fx in range(8):
            for tx in (fx - 1, fx, fx + 1):
                if not 0 <= tx < 8:
                    continue
                base = f"{FILES[fx]}{from_rank}{FILES[tx]}{to_rank}"
                moves.extend(base + piece for piece in "qrbn")
    return sorted(set(moves))


def _build() -> list[str]:
    vocab = ["<pad>", "<bos>", "<eos>", "<think>", "</think>", "<pv>", "</pv>"]
    vocab += [f"<elo:{i}>" for i in range(len(ELO_EDGES) + 1)]
    vocab += [f"<effort:{i}>" for i in range(EFFORT_LEVELS)]
    vocab += [f"<cp:{i}>" for i in range(len(CP_EDGES) + 1)]
    vocab += [f"<mate:{s}{n}>" for s in "+-" for n in range(1, MAX_MATE + 1)]
    vocab += [f"<delta:{i}>" for i in range(len(DELTA_EDGES) + 1)]
    # Reserved for the FEN prefix: 64 square slots emit one piece token each.
    vocab += ["<fen>", "</fen>", "<stm:w>", "<stm:b>", "<empty>"]
    vocab += [f"<{c}:{p}>" for c in "wb" for p in "pnbrqk"]
    vocab += [f"<castle:{c}>" for c in "KQkq"] + [f"<ep:{f}>" for f in FILES] + ["<ep:none>"]
    vocab += enumerate_moves()
    pad_to = (len(vocab) + 63) // 64 * 64
    vocab += [f"<unused:{i}>" for i in range(pad_to - len(vocab))]
    return vocab


VOCAB = _build()
ID = {token: i for i, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

PAD, BOS, EOS = ID["<pad>"], ID["<bos>"], ID["<eos>"]
THINK, THINK_END = ID["<think>"], ID["</think>"]
PV, PV_END = ID["<pv>"], ID["</pv>"]
MOVE_START = ID["a1a2"]
MOVE_IDS = range(MOVE_START, MOVE_START + len(enumerate_moves()))
# Keep every pretrained ID unchanged. Stockfish suggestions live after the
# original (padded) vocabulary, even when their UCI spelling matches a human move.
SF_MOVE_START = VOCAB_SIZE
SF_MOVE_IDS = range(SF_MOVE_START, SF_MOVE_START + len(MOVE_IDS))
INTERLEAVED_VOCAB_SIZE = (SF_MOVE_IDS.stop + 63) // 64 * 64


def sf_move_id(uci: str) -> int:
    return SF_MOVE_START + move_id(uci) - MOVE_START


def elo_id(rating: int) -> int:
    return ID[f"<elo:{bisect_left(ELO_EDGES, rating)}>"]


def effort_id(level: int) -> int:
    return ID[f"<effort:{level}>"]


def score_id(centipawns: int | None = None, mate: int | None = None) -> int:
    if mate is not None:
        n = min(abs(mate), MAX_MATE)
        return ID[f"<mate:{'+' if mate > 0 else '-'}{n}>"]
    assert centipawns is not None
    return ID[f"<cp:{bisect_left(CP_EDGES, centipawns)}>"]


def to_centipawns(centipawns: int | None, mate: int | None) -> int:
    if mate is not None:
        return (MATE_CP - abs(mate) * 100) * (1 if mate > 0 else -1)
    assert centipawns is not None
    return centipawns


def delta_id(best: int, candidate: int) -> int:
    """Bucket how much worse `candidate` is than `best`, both in centipawns."""
    return ID[f"<delta:{bisect_left(DELTA_EDGES, max(best - candidate, 0))}>"]


def move_id(uci: str) -> int:
    return ID[uci]


# Reuse existing IDs: old checkpoints and human move tokens stay compatible.
COMPACT_DELTA_BUCKETS = (0, 4, 8, 14)


def compact_delta_id(best: int, candidate: int) -> int:
    gap = max(best - candidate, 0)
    bucket = 0 if gap <= 5 else 4 if gap <= 40 else 8 if gap <= 180 else 14
    return ID[f"<delta:{bucket}>"]
