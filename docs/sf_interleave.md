# Eksperyment `S -> ruch człowieka`

Eksperyment sprawdza, czy podanie najlepszego ruchu Stockfisha przy każdym
prefixie poprawia przewidywanie następnego ruchu z oryginalnej partii.

## Format danych

Dla oryginalnych ruchów `W B W B ...` zapisujemy:

```text
S(W) W S(B) B S(W) W S(B) B ...
```

`S(X)` jest liczony z pozycji przed ruchem `X`; pozycja jest aktualizowana
wyłącznie prawdziwym ruchem partii. Rekord JSONL przechowuje oba pełne ciągi,
ratingi, miesiąc źródłowy i stabilny identyfikator gry. Z niego budowane są dwa
strumienie treningowe: `augmented.bin` oraz identyczny zbiór gier w
`human.bin` jako kontrola.

Każdy shard ma manifest SHA-256 z parametrami AIX, Stockfisha i kodu. `build`
odrzuca powtórzone identyfikatory oraz jakiekolwiek nakładające się miesiące
między train/val/test. To zapobiega wyciekowi między splitami na poziomie całej
partii, nie tylko pojedynczej pozycji. Metadane pretraina nie zawierają jednak
identyfikatorów gier, więc możliwe powtórzenie względem wcześniejszego pretraina
jest jawnie opisane w manifeście jako nieznane.

## Ustalony kompromis Stockfisha

Właściwe shardy używają Stockfisha 17.1, `Threads=1`, `Hash=16 MB` i `30,000`
nodes na prefix. Na tych samych 20 partiach zgodność pierwszego ruchu wyniosła:

| porównanie | zgodność |
| --- | ---: |
| 1k vs 10k nodes | 57.4% |
| 10k vs 30k nodes | 72.4% |
| 30k vs 100k nodes | 76.2% |

30k jest głównym ustawieniem: wyraźnie odcina się od taniego 1k/10k, a dalsze
zwiększenie głębokości ma malejący zwrot względem przepustowości. Pełny wybór
etykiet jest zachowany w manifestach i można go później powtórzyć dla innego
limitu.

## Ewaluacja i trening

`evaluate_interleave.py --mode augmented` liczy NLL/top-1/top-5 wyłącznie na
ludzkich tokenach po prawdziwym tokenie `S`; Stockfish jest więc teacher-forced,
a model nie może poprawić wyniku przez własne błędne przewidywanie `S`. Osobno
raportowana jest predykcja samego `S`. `--mode human` jest kontrolą bez
interleavingu.

Trening powinien używać tego samego checkpointu i tych samych gier dla obu ramion:

```sh
python scripts/train_interleave.py runs/pretrain/model.pt \
  --data-dir data/sf-interleave-v1/train --mode human \
  --out experiments/sf-interleave-v1/control-human --steps 5000

python scripts/train_interleave.py runs/pretrain/model.pt \
  --data-dir data/sf-interleave-v1/train --mode augmented \
  --out experiments/sf-interleave-v1/treatment-sf-human --steps 10000
```

Treatment ma dwa razy więcej tokenów na ten sam ruch ludzki, dlatego domyślnie
porównujemy go przy około dwukrotnie większej liczbie kroków. Oba strumienie mają
większą wagę lossu dla ruchów ludzkich (`human_weight=2`) niż dla pozostałych
tokenów. Raport końcowy powinien zawierać wyniki bazowego pretraina, kontroli i
treatmentu na tym samym `val/records.jsonl` oraz potwierdzenie na
`test/records.jsonl`.
