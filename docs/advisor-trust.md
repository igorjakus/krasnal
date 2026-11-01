# Advisor trust: kiedy model korzysta z podpowiedzi?

Pytania badawcze:

1. **Kalibracja.** Czy prawdopodobieństwo, które model daje ruchowi z podpowiedzi
   (`p_hint`), zgadza się z tym, jak często ludzie danego Elo faktycznie grają ten
   ruch (`agree`)? Różnica to `overtrust` (dodatnia oznacza nadmierne zaufanie).
2. **Wnioskowanie w kontekście.** Doradca jest losowany raz na partię i ukryty.
   Model widzi każdą podpowiedź oraz ruch, który człowiek potem zagrał, więc może
   policzyć, jak często doradca miał rację. Czy `p_hint` rośnie lub maleje z
   numerem ruchu (`by_ply`) zgodnie z jakością doradcy?
3. **Generalizacja.** Doradcy spoza treningu (`oracle50`, `oracle90`, `sfd4`): czy zaufanie
   wypada „pomiędzy” znanymi doradcami?

Oduczanie się samodzielnej gry (warunek `none`) to baseline z literatury
(causal confusion / copycat), a nie główny wkład.

## Doradcy

Kodowanie `A W A B ...` jak w SF-interleave. Podpowiedź jest tokenem z przestrzeni
SF, a brak podpowiedzi to stały token 4080 („gwiazdka”/kropka). Długość sekwencji
i pozycje RoPE nie zależą od doradcy. Loss liczony jest tylko na ruchach ludzkich
(`other_weight=0`).

| nazwa | co podpowiada | trening v1 | ewaluacja |
|---|---|---:|:---:|
| `none` | kropka (chroni przed oduczeniem się samodzielnej gry) | 15% | ✓ |
| `random` | losowy legalny ruch | 10% | ✓ |
| `sfd1` | słaby, szybki SF (depth 1) | 15% | ✓ |
| `sf30k` | SF 30k nodes (już policzony, pole `s`, za darmo) | 15% | ✓ |
| `oracle30`, `oracle70` | prawdziwy ruch z p=0.3/0.7, inaczej ruch SF ≠ człowiek, a gdy SF = człowiek, losowy legalny ≠ człowiek | 15/15% | ✓ |
| `oracle100` | zawsze prawdziwy ruch | 15% | ✓ |
| `oracle50`, `oracle90`, `sfd4` | — | held-out | ✓ |

`oracle100` jest w treningu celowo. Doradca jest losowany per partia, więc model
może go bezpiecznie kopiować dopiero wtedy, gdy z historii partii widzi, że ten
doradca zawsze ma rację. Na pierwszych ruchach partii musi ufać ostrożnie. Koszt:
w tych partiach loss szybko spada do zera, więc część obliczeń daje mało sygnału.

Losowanie (doradca, ruchy `random`/`oracle`) jest deterministyczne:
`sha256(seed, doradca, id partii)`. Implementacja jest w `src/krasnal/data/advisors.py`.

Doradcy `oracle` są syntetyczni, ale mają dokładnie znaną wiarygodność i są
naprawdę pomocni. Na teście SF-copying ludzie grali ruch SF 30k tylko w 39.2%
pozycji, a scratch-SF wygrał z gwiazdką zaledwie o +0.15 pp top-1. Dla modelu
100M nie ma kontroli „100M + gwiazdka”, więc nie wiadomo, ile z +3.43 pp pochodzi
z SF, a ile z dotrenowania na nowszych partiach. Dlatego istnieje config
`advisor-trust-v1-none.json`.

## Krok 0: ewaluacja istniejących checkpointów (bez treningu)

Dane testowe: `data/sf-interleave-v1/raw/test/2024-06.jsonl` (9 666 partii).

```sh
# 1. Dodatkowe podpowiedzi SF depth 1 i 4 dla testu (UWR, CPU)
ls data/sf-interleave-v1/raw/test/2024-06.jsonl > experiments/slurm/advisor-test-inputs.txt
sbatch --array=0 --export=ALL,MODE=all,INPUTS_FILE=experiments/slurm/advisor-test-inputs.txt,OUT_ROOT=data/advisor-trust-v1/test \
  scripts/slurm/advisor_annotate.sh

# 2. Ewaluacja SF-full 100M oraz scratch-SF we wszystkich warunkach (UWR, 1 GPU)
sbatch scripts/slurm/advisor_eval.sh
```

Wynik zapisuje się w `experiments/advisor-trust-v1/etap0/etap0.json`, a wiersze
per ruch w `*.rows.npz` (do wykresów i bootstrapu). Na co patrzeć:

- `overall.copy` i `p_hint` w funkcji jakości doradcy: czy krzywa jest płaska?
- `overtrust` per Elo (`by_elo`): czy model ufa SF tak samo przy 1200 i 2400?
- `random`, `sfd1` vs `none` (`vs_none.delta_nll`): czy zła podpowiedź szkodzi
  bardziej niż brak podpowiedzi?
- `oracle100`: jak bardzo model ufa podpowiedzi podejrzanie zawsze trafnej?

## Krok 1: trening mieszanki

```sh
# 1. Przypisanie doradcy per partia + SF depth 1 tylko dla ~15% partii
ls data/sf-interleave-v1/raw/train/*.jsonl > experiments/slurm/advisor-train-inputs.txt
ls data/sf-interleave-v1/raw/val/*.jsonl   > experiments/slurm/advisor-val-inputs.txt
sbatch --array=0-72%6 --export=ALL,MODE=assigned,INPUTS_FILE=experiments/slurm/advisor-train-inputs.txt,OUT_ROOT=data/advisor-trust-v1/raw/train \
  scripts/slurm/advisor_annotate.sh
sbatch --array=0 --export=ALL,MODE=assigned,INPUTS_FILE=experiments/slurm/advisor-val-inputs.txt,OUT_ROOT=data/advisor-trust-v1/raw/val \
  scripts/slurm/advisor_annotate.sh

# 2. Strumienie treningowe
sbatch scripts/slurm/advisor_build.sh

# 3. Fine-tune z foundation 100M (ten sam przepis co SF-full: 2 epoki, lr 1e-5)
.venv/bin/python scripts/run_advisor_trust.py configs/advisor-trust-v1.json --dry-run
sbatch scripts/slurm/train_advisor_trust.sh
#   na Adrianie (1 GPU): --world-size 1 --grad-accum 16

# 4. Ewaluacja nowego modelu we wszystkich warunkach
sbatch --export=ALL,MODELS="mixture=experiments/advisor-trust-v1/mixture/model.pt sf100m=experiments/sf-interleave-100m-full-20261009-humanloss/model.pt",OUT=experiments/advisor-trust-v1/eval/mixture.json \
  scripts/slurm/advisor_eval.sh
```

Kontrola bez podpowiedzi (100M + kropka) używa tych samych kroków z
`CONFIG=configs/advisor-trust-v1-none.json`, `ANN_ROOT=data/sf-interleave-v1/raw`
(surowe shardy, bez adnotacji) i `OUT_DIR=data/advisor-trust-v1/dataset-none`.
Ta kontrola nie potrzebuje Stockfisha.

Uwaga: `src/krasnal/data/` pasuje do wzorca `data/` w `.gitignore`, więc nowe pliki
w tym katalogu trzeba dodawać przez `git add -f`.

Wszystkie skrypty przeszły lokalny smoke test end-to-end: syntetyczne partie,
Stockfish 16, mały losowy GPT, adnotacja `all`/`assigned`, build, 20 kroków
treningu i ewaluacja. Na workerach nic jeszcze nie uruchomiono.
